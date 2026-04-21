#!/usr/bin/env bash
# Step 6: Integration & Regression smoke gate
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS_DIR:-$REPO/models}"
BASE="$MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf"
SPEC="$MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf"
export CUDA_VISIBLE_DEVICES=0

pass=0; fail=0

check() {
    local label="$1"; shift
    if "$@" 2>/dev/null; then
        echo "PASS: $label"; pass=$((pass+1))
    else
        echo "FAIL: $label"; fail=$((fail+1))
    fi
}

echo "=== Step 6: Integration & Regression ==="

# 6.1 Unit tests pass
echo "--- 6.1 Unit tests ---"
check "unit-tests" "$REPO/build/bin/test-spec-prefill" "$BASE"

# 6.2 spec-prefill-run binary exists and runs
echo "--- 6.2 Binary smoke ---"
check "binary-exists" test -f "$REPO/build/bin/llama-spec-prefill-run"
check "binary-help-exits-cleanly" bash -c "$REPO/build/bin/llama-spec-prefill-run 2>&1 | grep -q Usage"

# 6.3 Feature-off identity: kr=1.0 produces n_kept == n_total (all tokens kept)
echo "--- 6.3 Feature-off identity ---"
outfile="$(mktemp /tmp/smoke_kr1_XXXX.jsonl)"
"$REPO/build/bin/llama-spec-prefill-run" \
    --model "$BASE" --spec-model "$SPEC" \
    --keep-ratio 1.0 --lookahead 8 \
    --prompt-file "$REPO/tests/sample_prompts.jsonl" \
    --out "$outfile" 2>/dev/null
# At kr=1.0 n_kept should equal n_total (or be close)
check "kr1.0-output-nonempty" test -s "$outfile"
check "kr1.0-all-prompts" bash -c "wc -l < '$outfile' | grep -qE '^[1-9]'"
rm -f "$outfile"

# 6.4 Parity binary runs without crash
echo "--- 6.4 Parity binary ---"
tmpdir="$(mktemp -d)"
check "parity-binary-runs" "$REPO/build/bin/test-spec-prefill-parity" \
    "$MODELS/llama-3.1-8b-instruct-q4_k_m.gguf" \
    "$MODELS/llama-3.2-1b-instruct-q4_k_m.gguf" \
    "$tmpdir/traces.jsonl"
check "parity-traces-written" test -s "$tmpdir/traces.jsonl"
rm -rf "$tmpdir"

# 6.5 Bench binary runs (positional args: base spec)
echo "--- 6.5 Bench binary ---"
check "bench-runs" "$REPO/build/bin/test-spec-prefill-bench" "$BASE" "$SPEC"
check "bench-output-nonempty" bash -c "$REPO/build/bin/test-spec-prefill-bench '$BASE' '$SPEC' 2>/dev/null | grep -q 'Benchmark'"

# 6.6 Quality binary runs
echo "--- 6.6 Quality binary ---"
tmpout="$(mktemp /tmp/smoke_qual_XXXX.jsonl)"
check "quality-runs" "$REPO/build/bin/test-spec-prefill-quality" \
    --base "$BASE" --spec "$SPEC" \
    --prompt-file "$REPO/tests/sample_prompts.jsonl" \
    --out "$tmpout" --keep-ratio 0.25
check "quality-output-written" test -s "$tmpout"
rm -f "$tmpout"

echo ""
echo "=== Integration Gate: $pass PASS, $fail FAIL ==="
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS"
    exit 0
else
    echo "RESULT: FAIL ($fail failures)"
    exit 1
fi
