//! DSSP secondary structure classification from hydrogen bond pairs.
//!
//! Takes H-bond pairs (from [`crate::analysis::bonds::hydrogen`]) and
//! classifies residues into Helix, Sheet, or Coil using the Kabsch &
//! Sander (1983) algorithm.
//!
//! ## Algorithm overview
//!
//! 1. **Helix detection**: An n-turn at residue i exists when `has_hbond(i+n,
//!    i)`. A minimal n-helix requires 2 consecutive n-turns. Priority: alpha
//!    (n=4) > 3_10 (n=3) > pi (n=5).
//! 2. **Bridge detection**: Parallel and antiparallel beta-bridges between
//!    residue pairs.
//! 3. **Ladder construction**: Consecutive bridges of the same type form
//!    ladders. All residues in a ladder are marked Sheet.
//! 4. **Priority**: Helix > Sheet.

use crate::analysis::bonds::hydrogen::HBond;
use crate::analysis::SSType;

/// Bridge type for beta-sheet detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BridgeType {
    /// Parallel beta-bridge.
    Parallel,
    /// Antiparallel beta-bridge.
    Antiparallel,
}

/// A single beta-bridge between two residues.
#[derive(Debug, Clone, Copy)]
struct Bridge {
    /// First residue index (always < j).
    i: usize,
    /// Second residue index.
    j: usize,
    /// Type of bridge.
    kind: BridgeType,
}

/// Classify secondary structure from H-bond pairs.
///
/// Detects helices (alpha i->i+4, 3_10 i->i+3, pi i->i+5 turn patterns)
/// and sheets (parallel/antiparallel bridge patterns with ladder
/// extension) using the Kabsch & Sander 1983 DSSP algorithm.
#[must_use]
pub fn classify(hbonds: &[HBond], n_residues: usize) -> Vec<SSType> {
    if n_residues < 2 {
        return vec![SSType::Coil; n_residues];
    }

    let has_hbond = |donor: usize, acceptor: usize| -> bool {
        hbonds
            .iter()
            .any(|h| h.donor == donor && h.acceptor == acceptor)
    };

    let mut ss = vec![SSType::Coil; n_residues];

    // Helix detection in priority order: alpha (n=4) > 3_10 (n=3) > pi (n=5).
    for &turn_size in &[4usize, 3, 5] {
        detect_helices(&mut ss, n_residues, turn_size, &has_hbond);
    }

    // Bridge detection and ladder/strand marking.
    let bridges = detect_bridges(n_residues, &has_hbond);
    let sheet_residues = build_ladders_and_mark(&bridges, n_residues);

    // Apply sheet assignments (helix takes priority).
    for (s, &is_sheet) in ss.iter_mut().zip(sheet_residues.iter()) {
        if is_sheet && *s != SSType::Helix {
            *s = SSType::Sheet;
        }
    }

    ss
}

/// Detect helices of a given turn size and mark residues.
///
/// An n-turn at residue i exists when `has_hbond(i+n, i)`. A minimal
/// n-helix requires 2 consecutive n-turns at i and i+1, marking
/// residues i+1 through i+n as Helix. Only assigns to residues that
/// are still Coil (preserving higher-priority helix assignments).
fn detect_helices(
    ss: &mut [SSType],
    n_residues: usize,
    turn_size: usize,
    has_hbond: &dyn Fn(usize, usize) -> bool,
) {
    let mut turns = vec![false; n_residues];
    for (i, turn) in turns
        .iter_mut()
        .enumerate()
        .take(n_residues.saturating_sub(turn_size))
    {
        if has_hbond(i + turn_size, i) {
            *turn = true;
        }
    }

    // Find runs of consecutive turns. A run of length >= 2 forms a helix.
    let mut run_start: Option<usize> = None;

    for (i, &is_turn) in turns.iter().enumerate() {
        if is_turn {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else {
            if let Some(start) = run_start {
                mark_helix_run(ss, start, i - 1, turn_size, n_residues);
            }
            run_start = None;
        }
    }

    // Handle run extending to end of chain.
    if let Some(start) = run_start {
        let last_turn = find_last_turn(&turns, start);
        mark_helix_run(ss, start, last_turn, turn_size, n_residues);
    }
}

/// Find the last consecutive true entry starting from `start`.
fn find_last_turn(turns: &[bool], start: usize) -> usize {
    let mut last = start;
    for (i, &t) in turns.iter().enumerate().skip(start) {
        if t {
            last = i;
        } else {
            break;
        }
    }
    last
}

/// Mark a helix run in the SS array. Only assigns Helix to Coil residues.
fn mark_helix_run(
    ss: &mut [SSType],
    first_turn: usize,
    last_turn: usize,
    turn_size: usize,
    n_residues: usize,
) {
    let run_len = last_turn - first_turn + 1;
    if run_len < 2 {
        return;
    }
    let helix_start = first_turn + 1;
    // Turn at position p marks residues p+1..=p+n-1. With consecutive
    // turns from first_turn to last_turn, the helix spans
    // first_turn+1 ..= last_turn+turn_size-1.
    let helix_end = (last_turn + turn_size).min(n_residues);
    for s in ss.iter_mut().take(helix_end).skip(helix_start) {
        if *s == SSType::Coil {
            *s = SSType::Helix;
        }
    }
}

/// Detect all beta-bridges in the structure.
fn detect_bridges(
    n_residues: usize,
    has_hbond: &dyn Fn(usize, usize) -> bool,
) -> Vec<Bridge> {
    let mut bridges = Vec::new();

    for i in 0..n_residues {
        for j in (i + 3)..n_residues {
            // Parallel bridge: Hbond(i-1,j) && Hbond(j,i+1)
            //               or Hbond(j-1,i) && Hbond(i,j+1)
            let parallel = (i > 0
                && i + 1 < n_residues
                && has_hbond(j, i - 1)
                && has_hbond(i + 1, j))
                || (j > 0
                    && j + 1 < n_residues
                    && has_hbond(i, j - 1)
                    && has_hbond(j + 1, i));

            // Antiparallel bridge: Hbond(i,j) && Hbond(j,i)
            //                   or Hbond(i-1,j+1) && Hbond(j-1,i+1)
            let antiparallel = (has_hbond(j, i) && has_hbond(i, j))
                || (i > 0
                    && j + 1 < n_residues
                    && j > 0
                    && i + 1 < n_residues
                    && has_hbond(j + 1, i - 1)
                    && has_hbond(i + 1, j - 1));

            if parallel {
                bridges.push(Bridge {
                    i,
                    j,
                    kind: BridgeType::Parallel,
                });
            }
            if antiparallel {
                bridges.push(Bridge {
                    i,
                    j,
                    kind: BridgeType::Antiparallel,
                });
            }
        }
    }

    bridges
}

/// Build ladders from bridges and mark all ladder residues as Sheet.
///
/// A ladder is a maximal sequence of consecutive bridges of the same type.
/// Parallel consecutive: bridge(i,j) and bridge(i+1,j+1).
/// Antiparallel consecutive: bridge(i,j) and bridge(i+1,j-1).
///
/// Even isolated bridges (ladders of length 1) mark both residues.
fn build_ladders_and_mark(bridges: &[Bridge], n_residues: usize) -> Vec<bool> {
    let mut is_sheet = vec![false; n_residues];

    if bridges.is_empty() {
        return is_sheet;
    }

    // Mark all bridge partners.
    for b in bridges {
        is_sheet[b.i] = true;
        is_sheet[b.j] = true;
    }

    // Build ladders: chains of consecutive bridges of the same type.
    let mut used = vec![false; bridges.len()];

    for idx in 0..bridges.len() {
        if used[idx] {
            continue;
        }
        used[idx] = true;

        let mut ladder = vec![bridges[idx]];
        extend_ladder(&mut ladder, bridges, &mut used);
        mark_ladder_residues(&ladder, &mut is_sheet);
    }

    is_sheet
}

/// Extend a ladder by finding consecutive bridges of the same type.
fn extend_ladder(
    ladder: &mut Vec<Bridge>,
    bridges: &[Bridge],
    used: &mut [bool],
) {
    loop {
        let last = ladder[ladder.len() - 1];
        let next_i = last.i + 1;
        let next_j = match last.kind {
            BridgeType::Parallel => last.j + 1,
            BridgeType::Antiparallel => {
                if last.j == 0 {
                    break;
                }
                last.j - 1
            }
        };

        let found = bridges.iter().enumerate().find(|(bi, b)| {
            !used[*bi] && b.kind == last.kind && b.i == next_i && b.j == next_j
        });

        if let Some((bi, b)) = found {
            used[bi] = true;
            ladder.push(*b);
        } else {
            break;
        }
    }
}

/// Mark all residues spanned by a ladder as Sheet.
fn mark_ladder_residues(ladder: &[Bridge], is_sheet: &mut [bool]) {
    if ladder.len() <= 1 {
        return; // Single bridges already marked by the initial pass.
    }

    let i_min = ladder.iter().map(|b| b.i).min().unwrap_or(0);
    let i_max = ladder.iter().map(|b| b.i).max().unwrap_or(0);
    let j_min = ladder.iter().map(|b| b.j).min().unwrap_or(0);
    let j_max = ladder.iter().map(|b| b.j).max().unwrap_or(0);

    for s in is_sheet.iter_mut().take(i_max + 1).skip(i_min) {
        *s = true;
    }
    for s in is_sheet.iter_mut().take(j_max + 1).skip(j_min) {
        *s = true;
    }
}

#[cfg(test)]
#[path = "dssp_tests.rs"]
mod tests;
