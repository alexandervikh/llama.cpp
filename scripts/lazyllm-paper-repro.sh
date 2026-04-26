#!/bin/bash
# LazyLLM paper-reproduction benchmark script.
# Reproduces TTFT speedup results from arXiv:2407.14057 (Fu et al., 2024).
#
# Usage: ./scripts/lazyllm-paper-repro.sh [model.gguf] [output_dir]
#
# Paper target: Llama-2-7B, pruning_layers=[8,16,24], keep_ratios=[0.7,0.5,0.3]
# Reported TTFT speedup: 2.34x (7B), 2.16x (13B)
# POC gate: >=2.0x at 4k context (within 15% of paper's 2.34x)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$MEO_DIR/build"
BINARY="$BUILD_DIR/bin/llama-lazyllm-run"

MODEL=${1:-/home/coder/llama.cpp/models/llama-3.1-8b-instruct-q4_k_m.gguf}
OUT_DIR=${2:-$MEO_DIR/quality-results}
DATE=$(date +%Y%m%d-%H%M%S)
GIT_SHA=$(cd "$MEO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/lazyllm-paper-repro-${DATE}.md"

# Paper config for 32-layer models (Llama-2-7B, Llama-3.1-8B both have 32 layers)
PRUNING_LAYERS="8 16 24"
KEEP_RATIOS="0.7 0.5 0.3"
POOL_SIZE=13
REPEAT=5

# Detect GPU
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "CPU only")
MODEL_NAME=$(basename "$MODEL")

echo "==================================================================="
echo "LazyLLM Paper Reproduction — $(date)"
echo "Model:  $MODEL_NAME"
echo "GPU:    $GPU_INFO"
echo "Config: layers=[$PRUNING_LAYERS] ratios=[$KEEP_RATIOS] pool=$POOL_SIZE"
echo "==================================================================="

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Build with: cd $MEO_DIR && cmake --build build --target llama-lazyllm-run -j$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Available models:"
    ls /home/coder/llama.cpp/models/*.gguf 2>/dev/null | head -10
    exit 1
fi

# Generate a synthetic long prompt
gen_prompt() {
    local n_words=$1
    python3 -c "
import sys
words = 'The quick brown fox jumps over the lazy dog. The cat sat on the mat. A stitch in time saves nine. All that glitters is not gold. Better late than never. Every cloud has a silver lining. Fortune favours the bold. Great minds think alike. Haste makes waste. Ignorance is bliss. '.split()
result = []
import itertools
for i, w in zip(range($n_words), itertools.cycle(words)):
    result.append(w)
print(' '.join(result))
"
}

# Run benchmark at multiple context lengths
declare -A RESULTS_BASELINE
declare -A RESULTS_LAZYLLM
declare -A RESULTS_SPEEDUP

for N_TOKENS in 500 1000 2000 4000 8000; do
    N_CTX=$((N_TOKENS + 200))

    echo ""
    echo "--- Context: ~${N_TOKENS} tokens (n_ctx=${N_CTX}) ---"

    PROMPT=$(gen_prompt $((N_TOKENS * 2)))  # over-generate, tokenizer will trim
    TMP_PROMPT=$(mktemp /tmp/lazyllm_prompt_XXXX.txt)
    echo "$PROMPT" > "$TMP_PROMPT"

    OUTPUT=$("$BINARY" \
        --model "$MODEL" \
        --prompt "@$TMP_PROMPT" \
        --pruning-layers $PRUNING_LAYERS \
        --keep-ratios $KEEP_RATIOS \
        --pool-size $POOL_SIZE \
        --repeat $REPEAT \
        --n-ctx $N_CTX \
        2>/dev/null)

    rm -f "$TMP_PROMPT"

    echo "$OUTPUT" | grep -E "Prompt:|run|Median|Speedup|Gate"

    # Extract results
    BL=$(echo "$OUTPUT" | grep "Median baseline" | grep -oP '[\d.]+(?= ms)' | head -1)
    LZ=$(echo "$OUTPUT" | grep "Median LazyLLM"  | grep -oP '[\d.]+(?= ms)' | head -1)
    SP=$(echo "$OUTPUT" | grep "Speedup ratio"    | grep -oP '[\d.]+(?=x)' | head -1)

    RESULTS_BASELINE[$N_TOKENS]=${BL:-"N/A"}
    RESULTS_LAZYLLM[$N_TOKENS]=${LZ:-"N/A"}
    RESULTS_SPEEDUP[$N_TOKENS]=${SP:-"N/A"}
done

# Generate report
cat > "$REPORT" << EOF
# LazyLLM Paper Reproduction Report

## Setup

| Field | Value |
|---|---|
| Date | $(date) |
| Model | $MODEL_NAME |
| GPU | $GPU_INFO |
| Build | git=$GIT_SHA |
| Config | pruning_layers=[$PRUNING_LAYERS], keep_ratios=[$KEEP_RATIOS], pool_size=$POOL_SIZE |
| Repeats | $REPEAT timed runs, median reported |

## TTFT Speedup Results

| Context Tokens | Baseline TTFT (ms) | LazyLLM TTFT (ms) | Speedup |
|---|---|---|---|
EOF

for N_TOKENS in 500 1000 2000 4000 8000; do
    BL=${RESULTS_BASELINE[$N_TOKENS]:-"N/A"}
    LZ=${RESULTS_LAZYLLM[$N_TOKENS]:-"N/A"}
    SP=${RESULTS_SPEEDUP[$N_TOKENS]:-"N/A"}
    echo "| ${N_TOKENS} | $BL | $LZ | ${SP}× |" >> "$REPORT"
done

cat >> "$REPORT" << EOF

## Paper Comparison

| Metric | Paper (Llama-2-7B) | This run ($MODEL_NAME) |
|---|---|---|
| TTFT speedup @ 4k | 2.34× | ${RESULTS_SPEEDUP[4000]:-"N/A"}× |
| Gate threshold | ≥ 2.0× | $(python3 -c "s=${RESULTS_SPEEDUP[4000]:-0}; print('PASS' if s >= 2.0 else 'FAIL (target=2.0x)')" 2>/dev/null || echo "see above") |

## Configuration Notes

- Flash Attention: **DISABLED** (required for kq_soft_max tensor extraction, Phase 1 path)
- Position encoding: sequential 0..n_kept-1 (RoPE workaround; original positions = Phase 3 TODO)
- Architecture: Llama-2-7B has 32 layers; Llama-3.1-8B also has 32 layers — same pruning schedule applies
- Quantization: Q4_K_M (reduces prefill compute; F16 would give larger absolute speedup)

## Verdict

- **Paper gate (TTFT ≥ 2.0× at 4k context)**: $(python3 -c "s=${RESULTS_SPEEDUP[4000]:-0}; print('PASS' if s >= 2.0 else 'FAIL')" 2>/dev/null || echo "see results")
- **Quality gate**: Run \`python3 tools/lazyllm-eval.py --mode quality\` for F1 evaluation
EOF

echo ""
echo "==================================================================="
echo "Report saved to: $REPORT"
echo "==================================================================="
cat "$REPORT"
