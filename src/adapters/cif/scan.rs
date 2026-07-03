//! Streaming mmCIF scanner that feeds [`EntityBuilder`] without
//! constructing the full DOM.
//!
//! Returns `None` for inputs containing constructs the scanner doesn't
//! handle confidently (unterminated quotes, missing required
//! `_atom_site` columns). Callers fall back to the DOM path
//! [`super::dom_build`].

use std::collections::{HashMap, HashSet};

use super::hint::resolve_hint;
use super::refuse::multi_block_error;
use super::scan_row::{
    finish, push_row, push_slots, AtomSiteCols, RowSlots, RowValues,
};
use crate::entity::molecule::{Completion, EntityBuilder, MoleculeEntity};
use crate::ops::error::AdapterError;

/// Result of a single-model scan parse.
pub(super) type ScanResult = Option<Result<Vec<MoleculeEntity>, AdapterError>>;
/// Result of an all-models scan parse.
pub(super) type ScanModelsResult =
    Option<Result<Vec<Vec<MoleculeEntity>>, AdapterError>>;

pub(super) fn parse_mmcif_scan(
    input: &str,
    completion: Completion,
) -> ScanResult {
    let mut scanner = Scanner::new(input);
    let pre = scanner.run_prepass()?;
    if let Some(err) = scanner.pending_error.take() {
        return Some(Err(err));
    }
    let cols = scanner.locate_atom_site()?;
    if let Some(err) = scanner.pending_error.take() {
        return Some(Err(err));
    }
    let target = match scanner.first_model_number(&cols) {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };

    let mut builder = EntityBuilder::with_completion(completion);
    register_hints(&pre, &mut builder);

    let mut any_atom = false;
    let result = scanner.run_rows(&cols, |slots, cols| {
        if let Some(target) = target {
            if slots.model(cols).is_some_and(|m| m != target) {
                return Ok(());
            }
        }
        if push_slots(&mut builder, slots, cols)? {
            any_atom = true;
        }
        Ok(())
    });
    if let Err(e) = result {
        return Some(Err(e));
    }
    if let Err(e) = scanner.verify_no_extra_block() {
        return Some(Err(e));
    }
    if !any_atom {
        return Some(Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        )));
    }
    Some(finish(builder))
}

pub(super) fn parse_mmcif_scan_all_models(input: &str) -> ScanModelsResult {
    let mut scanner = Scanner::new(input);
    let pre = scanner.run_prepass()?;
    if let Some(err) = scanner.pending_error.take() {
        return Some(Err(err));
    }
    let cols = scanner.locate_atom_site()?;
    if let Some(err) = scanner.pending_error.take() {
        return Some(Err(err));
    }

    let (mut buckets, order) = match collect_model_buckets(&mut scanner, &cols)
    {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };

    let mut out: Vec<Vec<MoleculeEntity>> = Vec::with_capacity(order.len());
    for model in &order {
        let Some(rows) = buckets.remove(model) else {
            continue;
        };
        match build_model(&pre, &rows, &cols) {
            Ok(Some(es)) => out.push(es),
            Ok(None) => {}
            Err(e) => return Some(Err(e)),
        }
    }
    if out.is_empty() {
        return Some(Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        )));
    }
    Some(Ok(out))
}

type ModelBuckets = (HashMap<i32, Vec<RowValues>>, Vec<i32>);

fn collect_model_buckets(
    scanner: &mut Scanner<'_>,
    cols: &AtomSiteCols,
) -> Result<ModelBuckets, AdapterError> {
    let mut buckets: HashMap<i32, Vec<RowValues>> = HashMap::new();
    let mut order: Vec<i32> = Vec::new();
    let default_model: i32 = 1;

    scanner.run_rows(cols, |slots, cols| {
        let row = RowValues::from_slots(slots, cols);
        let key = row.model.unwrap_or(default_model);
        buckets
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(row);
        Ok(())
    })?;
    scanner.verify_no_extra_block()?;

    if buckets.is_empty() {
        return Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        ));
    }
    order.sort_unstable();
    Ok((buckets, order))
}

fn build_model(
    pre: &PrePass,
    rows: &[RowValues],
    cols: &AtomSiteCols,
) -> Result<Option<Vec<MoleculeEntity>>, AdapterError> {
    let mut builder = EntityBuilder::new();
    register_hints(pre, &mut builder);
    let mut any_atom = false;
    for row in rows {
        if push_row(&mut builder, row, cols)? {
            any_atom = true;
        }
    }
    if any_atom {
        Ok(Some(finish(builder)?))
    } else {
        Ok(None)
    }
}

// Hint registration

#[derive(Default)]
struct PrePass {
    entity_types: Vec<(String, String)>,
    poly_types: HashMap<String, String>,
}

fn pick_two_cells(
    row: &[&str],
    a: Option<usize>,
    b: Option<usize>,
) -> Option<(String, String)> {
    let a = (*row.get(a?)?).to_owned();
    let b = (*row.get(b?)?).to_owned();
    Some((a, b))
}

fn absorb_singleton(tag: &str, value: &str, pre: &mut PrePass) {
    match tag {
        "_entity.id" => {
            pre.entity_types.push((value.to_owned(), String::new()));
        }
        "_entity.type" => {
            if let Some(last) = pre.entity_types.last_mut() {
                value.clone_into(&mut last.1);
            }
        }
        "_entity_poly.entity_id" => {
            let _ = pre.poly_types.insert(value.to_owned(), String::new());
        }
        "_entity_poly.type" => {
            if let Some(last) = pre.poly_types.values_mut().last() {
                value.clone_into(last);
            }
        }
        _ => {}
    }
}

fn register_hints(pre: &PrePass, builder: &mut EntityBuilder) {
    let mut seen: HashSet<String> = HashSet::new();
    for (id, etype) in &pre.entity_types {
        let poly = pre.poly_types.get(id).map(String::as_str);
        let hint = resolve_hint(etype, poly);
        builder.register_entity(id, hint);
        let _ = seen.insert(id.clone());
    }
    for (id, ptype) in &pre.poly_types {
        if seen.contains(id) {
            continue;
        }
        let hint = resolve_hint("polymer", Some(ptype));
        builder.register_entity(id, hint);
    }
}

// Scanner state machine

enum RowOutcome {
    /// A complete row of values landed in `self.scratch`.
    Row,
    /// The loop ended cleanly at this position.
    End,
    /// An unhandled construct mid-row; caller bails to DOM.
    Bail,
}

struct Scanner<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    seen_data_block: bool,
    at_line_start: bool,
    pending_error: Option<AdapterError>,
    /// Per-row borrowed cells, reused across rows to avoid per-row
    /// allocation. Slices point into `input`.
    scratch: Vec<&'a str>,
}

impl<'a> Scanner<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            seen_data_block: false,
            at_line_start: true,
            pending_error: None,
            scratch: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    /// First pass: collect `_entity` and `_entity_poly` rows, stopping
    /// at the start of the `_atom_site` loop. Refuses (`pending_error`)
    /// on a second `data_` block.
    fn run_prepass(&mut self) -> Option<PrePass> {
        let mut pre = PrePass::default();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                return None;
            }
            let b = self.bytes[self.pos];
            if b == b';' && self.at_line_start {
                let _ = self.scan_semicolon_text()?;
                continue;
            }
            if b == b'\'' || b == b'"' {
                let _ = self.scan_quoted(b)?;
                continue;
            }
            let kw_start = self.pos;
            let kw = self.scan_unquoted_token();
            if kw.is_empty() {
                self.pos += 1;
                continue;
            }
            let kw_lower = kw.to_ascii_lowercase();

            if kw_lower.starts_with("data_") {
                if self.seen_data_block {
                    self.pending_error = Some(multi_block_error(2));
                    return Some(PrePass::default());
                }
                self.seen_data_block = true;
                continue;
            }
            if kw_lower == "loop_" {
                let probe = self.peek_first_tag()?;
                let probe_lower = probe.to_ascii_lowercase();
                if probe_lower.starts_with("_atom_site.") {
                    self.pos = kw_start;
                    return Some(pre);
                }
                if probe_lower.starts_with("_entity_poly.") {
                    self.scan_entity_poly_loop(&mut pre.poly_types)?;
                } else if probe_lower.starts_with("_entity.") {
                    self.scan_entity_loop(&mut pre.entity_types)?;
                } else {
                    self.skip_loop()?;
                }
                continue;
            }
            if kw_lower.starts_with("save_") {
                let _ = self.scan_next_value()?;
                continue;
            }
            if kw_lower.starts_with('_') {
                let val = self.scan_next_value()?;
                absorb_singleton(&kw_lower, &val, &mut pre);
            }
        }
    }

    fn peek_first_tag(&mut self) -> Option<String> {
        self.skip_whitespace_and_comments();
        let save = self.pos;
        let t = self.scan_unquoted_token();
        self.pos = save;
        if t.is_empty() {
            None
        } else {
            Some(t.to_owned())
        }
    }

    fn scan_entity_loop(
        &mut self,
        out: &mut Vec<(String, String)>,
    ) -> Option<()> {
        let tags = self.scan_loop_tags()?;
        let id_idx = tags
            .iter()
            .position(|t| t.eq_ignore_ascii_case("_entity.id"));
        let type_idx = tags
            .iter()
            .position(|t| t.eq_ignore_ascii_case("_entity.type"));
        let stride = tags.len();
        loop {
            match self.read_loop_row(stride) {
                RowOutcome::Row => {
                    if let Some((id, etype)) =
                        pick_two_cells(&self.scratch, id_idx, type_idx)
                    {
                        out.push((id, etype));
                    }
                }
                RowOutcome::End => return Some(()),
                RowOutcome::Bail => return None,
            }
        }
    }

    fn scan_entity_poly_loop(
        &mut self,
        out: &mut HashMap<String, String>,
    ) -> Option<()> {
        let tags = self.scan_loop_tags()?;
        let id_idx = tags
            .iter()
            .position(|t| t.eq_ignore_ascii_case("_entity_poly.entity_id"));
        let type_idx = tags
            .iter()
            .position(|t| t.eq_ignore_ascii_case("_entity_poly.type"));
        let stride = tags.len();
        loop {
            match self.read_loop_row(stride) {
                RowOutcome::Row => {
                    if let Some((id, ptype)) =
                        pick_two_cells(&self.scratch, id_idx, type_idx)
                    {
                        let _ = out.insert(id, ptype);
                    }
                }
                RowOutcome::End => return Some(()),
                RowOutcome::Bail => return None,
            }
        }
    }

    fn skip_loop(&mut self) -> Option<()> {
        let tags = self.scan_loop_tags()?;
        let stride = tags.len();
        loop {
            match self.read_loop_row(stride) {
                RowOutcome::Row => {}
                RowOutcome::End => return Some(()),
                RowOutcome::Bail => return None,
            }
        }
    }

    fn scan_loop_tags(&mut self) -> Option<Vec<String>> {
        let mut tags: Vec<String> = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() || self.bytes[self.pos] != b'_' {
                break;
            }
            let t = self.scan_unquoted_token();
            if t.is_empty() {
                break;
            }
            tags.push(t.to_owned());
        }
        if tags.is_empty() {
            None
        } else {
            Some(tags)
        }
    }

    /// Read one loop row of `stride` borrowed values into `self.scratch`.
    fn read_loop_row(&mut self, stride: usize) -> RowOutcome {
        self.scratch.clear();
        for i in 0..stride {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                return if i == 0 {
                    RowOutcome::End
                } else {
                    RowOutcome::Bail
                };
            }
            if self.is_loop_terminator() {
                return if i == 0 {
                    RowOutcome::End
                } else {
                    RowOutcome::Bail
                };
            }
            let Some(value) = self.scan_value_borrowed() else {
                return RowOutcome::Bail;
            };
            self.scratch.push(value);
        }
        RowOutcome::Row
    }

    fn is_loop_terminator(&self) -> bool {
        if self.pos >= self.len() {
            return true;
        }
        let b = self.bytes[self.pos];
        if b == b'_' {
            return true;
        }
        if self.pos + 5 <= self.len() {
            let head = &self.bytes[self.pos..self.pos + 5];
            if head.eq_ignore_ascii_case(b"loop_")
                || head.eq_ignore_ascii_case(b"data_")
                || head.eq_ignore_ascii_case(b"save_")
            {
                return true;
            }
        }
        false
    }

    fn scan_next_value(&mut self) -> Option<String> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.len() {
            return None;
        }
        self.scan_value()
    }

    fn locate_atom_site(&mut self) -> Option<AtomSiteCols> {
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                return None;
            }
            let kw = self.scan_unquoted_token();
            if kw.is_empty() {
                self.pos += 1;
                continue;
            }
            if kw.eq_ignore_ascii_case("loop_") {
                let probe = self.peek_first_tag()?;
                if !probe.to_ascii_lowercase().starts_with("_atom_site.") {
                    self.skip_loop()?;
                    continue;
                }
                let tags = self.scan_loop_tags()?;
                return AtomSiteCols::from_tags(&tags);
            }
            if kw.eq_ignore_ascii_case("data_") || kw.starts_with("data_") {
                if self.seen_data_block {
                    self.pending_error = Some(multi_block_error(2));
                    return None;
                }
                self.seen_data_block = true;
                continue;
            }
            if kw.starts_with('_') {
                let _ = self.scan_next_value()?;
            }
        }
    }

    fn first_model_number(
        &mut self,
        cols: &AtomSiteCols,
    ) -> Result<Option<i32>, AdapterError> {
        let Some(model_col) = cols.pdb_model_num else {
            return Ok(None);
        };
        let save = self.pos;
        let mut min: Option<i32> = None;
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() || self.is_loop_terminator() {
                break;
            }
            match self.read_loop_row(cols.ncols) {
                RowOutcome::Row => {}
                RowOutcome::End => break,
                RowOutcome::Bail => {
                    self.pos = save;
                    return Err(AdapterError::InvalidFormat(
                        "Fast path: unable to scan _atom_site row for model \
                         number"
                            .into(),
                    ));
                }
            }
            if let Some(s) = self.scratch.get(model_col) {
                if let Ok(n) = s.parse::<i32>() {
                    min = Some(min.map_or(n, |cur| cur.min(n)));
                }
            }
        }
        self.pos = save;
        Ok(min)
    }

    fn run_rows<F>(
        &mut self,
        cols: &AtomSiteCols,
        mut visit: F,
    ) -> Result<(), AdapterError>
    where
        F: FnMut(&RowSlots<'_, 'a>, &AtomSiteCols) -> Result<(), AdapterError>,
    {
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                if let Some(err) = self.pending_error.take() {
                    return Err(err);
                }
                break;
            }
            if self.is_loop_terminator() {
                break;
            }
            self.scan_row(cols)?;
            let slots = RowSlots::new(&self.scratch);
            visit(&slots, cols)?;
        }
        Ok(())
    }

    /// Tokenize one `_atom_site` row into `self.scratch` as borrowed cells.
    fn scan_row(&mut self, cols: &AtomSiteCols) -> Result<(), AdapterError> {
        self.scratch.clear();
        for _ in 0..cols.ncols {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                return Err(AdapterError::InvalidFormat(
                    "Unexpected end of CIF data in _atom_site loop".into(),
                ));
            }
            let v = self.scan_value_borrowed().ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "Fast path: unable to tokenize _atom_site value".into(),
                )
            })?;
            self.scratch.push(v);
        }
        Ok(())
    }

    fn verify_no_extra_block(&mut self) -> Result<(), AdapterError> {
        if let Some(err) = self.pending_error.take() {
            return Err(err);
        }
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.len() {
                return Ok(());
            }
            let kw = self.scan_unquoted_token();
            if kw.is_empty() {
                self.pos += 1;
                continue;
            }
            let kw_lower = kw.to_ascii_lowercase();
            if kw_lower.starts_with("data_") {
                return Err(multi_block_error(2));
            }
            if kw_lower == "loop_" {
                let _ = self.scan_loop_tags();
                continue;
            }
            if kw_lower.starts_with('_') {
                let _ = self.scan_next_value();
            }
        }
    }

    // Token primitives

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' => self.pos += 1,
                b'\n' => {
                    self.pos += 1;
                    self.at_line_start = true;
                }
                b'#' => {
                    while self.pos < self.len() && self.bytes[self.pos] != b'\n'
                    {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn scan_unquoted_token(&mut self) -> &'a str {
        self.at_line_start = false;
        let start = self.pos;
        while self.pos < self.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_whitespace() || c == b'#' {
                break;
            }
            self.pos += 1;
        }
        // SAFETY: input was &str; byte slice respects char boundaries.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }

    /// Tokenize one value as a borrowed slice into the input buffer.
    ///
    /// CIF values are taken verbatim: unquoted tokens, the text between
    /// matching single/double quotes, and `;`-delimited multiline blocks all
    /// reproduce a contiguous span of the source (CIF has no in-string escape
    /// sequence; an embedded quote not followed by whitespace is literal), so
    /// every value borrows without copying.
    fn scan_value_borrowed(&mut self) -> Option<&'a str> {
        if self.pos >= self.len() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b == b';' && self.at_line_start {
            return self.scan_semicolon_text();
        }
        if b == b'\'' || b == b'"' {
            return self.scan_quoted(b);
        }
        Some(self.scan_unquoted_token())
    }

    fn scan_value(&mut self) -> Option<String> {
        self.scan_value_borrowed().map(str::to_owned)
    }

    /// CIF rule: the closing quote must be followed by whitespace, `#`,
    /// or EOF. A quote followed by anything else is part of the string.
    fn scan_quoted(&mut self, quote: u8) -> Option<&'a str> {
        self.at_line_start = false;
        let start = self.pos + 1;
        let mut cursor = start;
        while cursor < self.len() {
            if self.bytes[cursor] == quote {
                let next = self.bytes.get(cursor + 1).copied();
                if next.is_none_or(|b| b.is_ascii_whitespace() || b == b'#') {
                    let val = &self.input[start..cursor];
                    self.pos = cursor + 1;
                    return Some(val);
                }
            }
            cursor += 1;
        }
        None
    }

    fn scan_semicolon_text(&mut self) -> Option<&'a str> {
        self.at_line_start = false;
        let start = self.pos + 1;
        let mut cursor = start;
        while cursor < self.len() {
            while cursor < self.len() && self.bytes[cursor] != b'\n' {
                cursor += 1;
            }
            if cursor >= self.len() {
                return None;
            }
            cursor += 1;
            if cursor < self.len() && self.bytes[cursor] == b';' {
                let end = cursor - 1;
                let text = &self.input[start..end];
                self.pos = cursor + 1;
                return Some(text);
            }
        }
        None
    }
}
