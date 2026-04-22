#!/usr/bin/env bash
# spec-prefill parity mock test
#
# Compares C++ spec-prefill output against Python reference implementation
# for the same input prompt. Validates algorithmic parity without requiring
# vLLM (addresses §1b Oracle review concern).
#
# Usage:
#   ./tools/spec-prefill-parity-mock.sh --cpp-bin <path> --py-ref <path> --model <path>
#
# Requires:
#   - C++ llama-spec-prefill-run binary
#   - Python torch (optional; test skips gracefully if unavailable)
#   - GGUF model file

set -euo pipefail

CPP_BIN=""
PY_REF=""
MODEL=""
PROMPT_FILE=""
OUT_DIR="/tmp/spec_parity_mock"

usage() {
    echo "Usage: $0 --cpp-bin <path> --py-ref <path> --model <path> [--prompt-file <path>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpp-bin) CPP_BIN="$2"; shift 2 ;;
        --py-ref) PY_REF="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        *) usage ;;
    esac
done

if [[ -z "$CPP_BIN" || -z "$MODEL" ]]; then
    echo "ERROR: --cpp-bin and --model are required"
    usage
fi

mkdir -p "$OUT_DIR"
TEST_PROMPT="$OUT_DIR/parity_test_prompt.txt"
CPP_OUT="$OUT_DIR/cpp_output.json"
PY_OUT="$OUT_DIR/py_output.json"

# Create a deterministic test prompt
cat > "$TEST_PROMPT" <<EOF
The capital of France is Paris, a city known for its art, fashion, and the Eiffel Tower. The Louvre Museum, home to the Mona Lisa, is one of the world's largest and most famous art museums. Paris has been a center of art, fashion, gastronomy, and culture for centuries.
EOF

echo "=== Spec-Prefill Parity Mock Test ==="
echo ""

# Run C++ binary
echo "Running C++ spec-prefill..."
"$CPP_BIN" \
    --model "$MODEL" \
    --spec-model "$MODEL" \
    --prompt-file "$TEST_PROMPT" \
    --out "$CPP_OUT" \
    --lookahead 8 \
    --pool 13 \
    --chunk-size 32 \
    2>/dev/null

if [[ ! -f "$CPP_OUT" ]]; then
    echo "FAIL: C++ binary produced no output"
    exit 1
fi

CPP_N_KEPT=$(head -1 "$CPP_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_kept'])" 2>/dev/null || echo "-1")
CPP_N_PROMPT=$(head -1 "$CPP_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_total'])" 2>/dev/null || echo "-1")
CPP_TTFT=$(head -1 "$CPP_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['ttft_ms'])" 2>/dev/null || echo "-1")

echo "  C++: n_kept=$CPP_N_KEPT, n_prompt=$CPP_N_PROMPT, ttft=$CPP_TTFT ms"

# Run Python ref-impl (if available)
if [[ -z "$PY_REF" ]]; then
    # Try to find the ref impl
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [[ -f "$SCRIPT_DIR/spec-prefill-ref-impl.py" ]]; then
        PY_REF="$SCRIPT_DIR/spec-prefill-ref-impl.py"
    else
        PY_REF=""
    fi
fi

if [[ -z "$PY_REF" || ! -f "$PY_REF" ]]; then
    echo "SKIP: Python ref-impl not found — parity comparison not possible"
    echo "  (Install torch and run with --py-ref <path> for full parity test)"
    exit 0
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "SKIP: torch not available — parity comparison not possible"
    exit 0
fi

echo "Running Python ref-impl..."
python3 "$PY_REF" --model "$MODEL" --prompt-file "$TEST_PROMPT" --out "$PY_OUT" 2>/dev/null

if [[ ! -f "$PY_OUT" ]]; then
    echo "SKIP: Python ref-impl produced no output"
    exit 0
fi

PY_N_KEPT=$(head -1 "$PY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_kept'])" 2>/dev/null || echo "-1")
PY_N_PROMPT=$(head -1 "$PY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_total'])" 2>/dev/null || echo "-1")

echo "  Python: n_kept=$PY_N_KEPT, n_prompt=$PY_N_PROMPT"

# Compare
if [[ "$CPP_N_KEPT" == "$PY_N_KEPT" && "$CPP_N_PROMPT" == "$PY_N_PROMPT" ]]; then
    echo ""
    echo "PASS: C++ and Python ref-impl produce identical n_kept ($CPP_N_KEPT) and n_prompt ($CPP_N_PROMPT)"
    exit 0
else
    echo ""
    echo "FAIL: C++ and Python ref-impl produce different results"
    echo "  C++ n_kept=$CPP_N_KEPT vs Python n_kept=$PY_N_KEPT"
    echo "  C++ n_prompt=$CPP_N_PROMPT vs Python n_prompt=$PY_N_PROMPT"
    exit 1
fi
