# Use PowerShell on Windows (bash resolves to WSL on some machines)
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Run all checks (what CI runs)
check: fmt-check clippy test doc doc-python

# Format check (nightly)
fmt-check:
    cargo +nightly fmt --check

# Format (nightly)
fmt:
    cargo +nightly fmt

# Clippy with all targets
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run tests (pure-Rust + non-Python features). Portable: needs no Python.
# NOT --all-features, because `extension-module` cannot link a standalone test
# binary. The Python bindings are tested separately by `test-python`.
test:
    cargo test --features serde,specta,c-api

# Run the Python-binding tests. Requires a shared, embeddable CPython; pin it
# explicitly so pyo3 doesn't auto-grab an inconsistent interpreter (e.g. a
# pixi/uv env) whose libpython isn't on the runtime path. `auto-initialize`
# gives the in-process tests an interpreter to attach to.
test-python python="python3":
    PYO3_PYTHON=`which {{python}}` cargo test --features python,serde,specta,c-api,pyo3/auto-initialize

# Build docs and check for warnings
doc $RUSTDOCFLAGS="-D warnings":
    cargo doc --no-deps --document-private-items

# Build docs for the Python/c-api surface and check for warnings. The
# featureless `doc` recipe never compiles the pyo3 surface, so intra-doc
# links in those modules go unchecked unless we build with the feature set.
doc-python $RUSTDOCFLAGS="-D warnings":
    cargo doc --no-deps --document-private-items --features python,serde,specta,c-api

# Dependency audit
deny:
    cargo deny check

# Check for unused dependencies
machete:
    cargo machete

# Check file lengths (max 800 lines). A file opts out by placing the sentinel
# comment `foldit:allow-long-file` within its first 10 lines (mirrors the
# repo-wide scripts/check_file_lengths.py convention).
file-lengths:
    python3 -c "import os, sys; SENTINEL='foldit:allow-long-file'; files = [os.path.join(r,f) for r,_,fs in os.walk('src') for f in fs if f.endswith('.rs') and 'target' not in r]; bad = []; [bad.append((f,len(L))) for f in files for L in [open(f,encoding='utf-8',errors='replace').readlines()] if len(L) > 800 and not any(SENTINEL in ln for ln in L[:10])]; [print(f'ERROR: {f} has {n} lines (max 800)') for f,n in bad]; sys.exit(1 if bad else 0)"

# One-time repo setup (hooks + commit template)
setup:
    git config core.hooksPath .githooks
    git config commit.template .gitmessage
    @echo "Done. Hooks and commit template activated."

# Count clippy errors
errors:
    cargo clippy --all-targets --all-features 2>&1 | rg '^error' | wc -l

# Count clippy warnings
warnings:
    cargo clippy --all-targets --all-features 2>&1 | rg '^warning' | wc -l

# Clippy violations per-rule, per-module (optionally filter by dir)
# Usage: just lint [dir]
# Examples: just lint              (all modules)
#           just lint adapters     (only src/adapters/)
#           just lint types        (only src/types/)
lint dir="":
    #!/usr/bin/env bash
    cargo clippy --all-targets --all-features --message-format=json 2>/dev/null \
    | python3 -c "
    import sys, json
    filt = '{{dir}}'
    seen = set()
    rows = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get('reason') != 'compiler-message':
            continue
        m = msg['message']
        code = m.get('code')
        if not code or not code.get('code'):
            continue
        rule = code['code']
        spans = m.get('spans', [])
        primary = next((s for s in spans if s.get('is_primary')), None)
        if not primary:
            continue
        f = primary['file_name']
        # skip non-source files (Cargo.toml metadata lints, etc.)
        if not f.startswith('src/'):
            continue
        if filt and not f.startswith('src/' + filt):
            continue
        # deduplicate across targets (lib vs test emit the same diagnostic)
        key = (f, primary.get('line_start'), rule)
        if key in seen:
            continue
        seen.add(key)
        # module = first two path components under src/
        parts = f.split('/')
        if len(parts) >= 3:
            mod_name = parts[1] + '/' + parts[2].replace('.rs', '')
        elif len(parts) == 2:
            mod_name = parts[1].replace('.rs', '')
        else:
            mod_name = f
        rows.append((mod_name, rule))
    if not rows:
        print('No violations found.')
        sys.exit(0)
    # Count per (module, rule)
    from collections import Counter
    counts = Counter(rows)
    by_mod = {}
    for (mod_name, rule), n in counts.items():
        by_mod.setdefault(mod_name, []).append((rule, n))
    total = sum(counts.values())
    for mod_name in sorted(by_mod):
        rules = sorted(by_mod[mod_name], key=lambda x: -x[1])
        mod_total = sum(n for _, n in rules)
        print(f'\n  {mod_name} ({mod_total})')
        for rule, n in rules:
            print(f'    {n:3d}  {rule}')
    print(f'\n  total: {total}')
    "

# Run everything including optional tools
check-all: check deny machete file-lengths
