#!/usr/bin/env bash
# LazyLLM — paper-reproduction one-command runner.
# Reference: arXiv:2407.14057 (Fu et al., Apple, 2024)
#
# Usage:
#   ./scripts/lazyllm-paper-reproduction.sh [--run-id NAME] [--skip-build] [--skip-datasets]
#   [--skip-ttft] [--skip-quality] [--models-dir DIR] [--out-dir DIR]
#
# Required models (GGUF format, see §1.1 of LAZYLLM_PLAN.md):
#   models/llama-2-7b-f16.gguf
#   models/llama-2-7b-chat-f16.gguf
#   models/xgen-7b-8k-base-f16.gguf
#   models/llama-2-13b-q8_0.gguf   (Q8_0 if 13B-FP16 doesn't fit on GPU)
#
# Paper config (32-layer models):
#   pruning_layers = {8, 16, 24}  (= L/4, L/2, 3L/4)
#   keep_ratios    = {0.7, 0.5, 0.3}
#   pool_kernel_size = 13
#   greedy decoding (temp=0)
#   n_ctx = 4096
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

# ── defaults ──────────────────────────────────────────────────────────────────
RUN_ID=$(date +%Y%m%d-%H%M%S)
MODELS_DIR="$REPO/models"
OUT_DIR="$REPO/results/$RUN_ID"
SKIP_BUILD=0
SKIP_DATASETS=0
SKIP_TTFT=0
SKIP_QUALITY=0
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --models-dir)    MODELS_DIR="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --skip-build)    SKIP_BUILD=1; shift ;;
        --skip-datasets) SKIP_DATASETS=1; shift ;;
        --skip-ttft)     SKIP_TTFT=1; shift ;;
        --skip-quality)  SKIP_QUALITY=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4 || echo "CPU only")

echo "=================================================================="
echo "LazyLLM Paper Reproduction Run"
echo "  Run ID  : $RUN_ID"
echo "  Git SHA : $GIT_SHA"
echo "  GPU     : $GPU_INFO"
echo "  Out dir : $OUT_DIR"
echo "  Date    : $(date)"
echo "=================================================================="
tee -a "$LOG" <<EOF
run_id=$RUN_ID
git_sha=$GIT_SHA
date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gpu=$GPU_INFO
EOF

# ── paper hyperparameters ─────────────────────────────────────────────────────
PRUNING_LAYERS="8 16 24"
KEEP_RATIOS="0.7 0.5 0.3"
POOL_SIZE=13
N_CTX=4096
N_GPU_LAYERS=99    # offload all layers
REPEAT=5           # timed reps (drop first cold rep)
SEED=42

# ── per-subset max_new_tokens (LongBench official) ────────────────────────────
declare -A MAX_NEW_TOKENS=(
    [narrativeqa]=128 [qasper]=128 [multifieldqa_en]=64
    [hotpotqa]=32 [2wikimqa]=32 [musique]=32
    [gov_report]=512 [qmsum]=512 [multi_news]=512
    [trec]=64 [triviaqa]=32 [samsum]=128
    [passage_count]=32 [passage_retrieval_en]=32
    [lcc]=64 [repobench-p]=64
)

ALL_SUBSETS=(
    narrativeqa qasper multifieldqa_en
    hotpotqa 2wikimqa musique
    gov_report qmsum multi_news
    trec triviaqa samsum
    passage_count passage_retrieval_en
    lcc repobench-p
)

# ── models to test ─────────────────────────────────────────────────────────────
declare -A MODELS=(
    [llama-2-7b]="$MODELS_DIR/llama-2-7b-f16.gguf"
    [llama-2-7b-chat]="$MODELS_DIR/llama-2-7b-chat-f16.gguf"
    [xgen-7b-8k]="$MODELS_DIR/xgen-7b-8k-base-f16.gguf"
    [llama-2-13b]="$MODELS_DIR/llama-2-13b-q8_0.gguf"
)

# ── phase 0: build ─────────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo ""
    echo ">>> Phase 0: Building binaries..."
    cmake -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_GRAPHS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=OFF \
        2>&1 | tee -a "$LOG"
    cmake --build build -j"$NPROC" \
        --target llama-lazyllm-run llama-bench llama-cli llama-quantize \
        2>&1 | tee -a "$LOG"
    echo "Build complete."
fi

BINARY_LZ="$REPO/build/bin/llama-lazyllm-run"
BINARY_BENCH="$REPO/build/bin/llama-bench"
if [[ ! -x "$BINARY_LZ" ]]; then
    echo "ERROR: $BINARY_LZ not found. Run with --skip-build=0 or build manually." | tee -a "$LOG"
    exit 1
fi

# ── phase 1: datasets ──────────────────────────────────────────────────────────
if [[ $SKIP_DATASETS -eq 0 ]]; then
    echo ""
    echo ">>> Phase 1: Downloading all 16 LongBench subsets (200 each)..."
    python3 "$REPO/scripts/get-longbench.py" \
        --all-tasks --n 200 \
        --out "$REPO/datasets/longbench/" \
        2>&1 | tee -a "$LOG"
    echo "Datasets ready."
fi

# ── helper: check model exists ─────────────────────────────────────────────────
check_model() {
    local name="$1" path="$2"
    if [[ ! -f "$path" ]]; then
        echo "  SKIP $name: model not found at $path" | tee -a "$LOG"
        return 1
    fi
    return 0
}

# ── phase 2: TTFT sweeps ───────────────────────────────────────────────────────
if [[ $SKIP_TTFT -eq 0 ]]; then
    echo ""
    echo ">>> Phase 2: TTFT sweeps (llama-bench --lazyllm)..."
    mkdir -p "$OUT_DIR/ttft"
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        check_model "$model_name" "$model_path" || continue

        echo "  Running TTFT sweep for $model_name..."
        "$BINARY_BENCH" \
            -m "$model_path" \
            -ngl "$N_GPU_LAYERS" \
            --lazyllm \
            --lazyllm-layers $PRUNING_LAYERS \
            --lazyllm-ratios $KEEP_RATIOS \
            -p 2048,4096,8192,16384 \
            -n 0 \
            -r "$REPEAT" \
            -o md \
            2>>"$LOG" \
            > "$OUT_DIR/ttft/ttft-${model_name}.md" \
            || echo "  WARN: bench failed for $model_name" | tee -a "$LOG"

        echo "  Saved: $OUT_DIR/ttft/ttft-${model_name}.md"
    done
fi

# ── phase 3: quality sweeps ────────────────────────────────────────────────────
if [[ $SKIP_QUALITY -eq 0 ]]; then
    echo ""
    echo ">>> Phase 3: Quality sweeps (llama-lazyllm-run on LongBench)..."
    mkdir -p "$OUT_DIR/quality"

    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        check_model "$model_name" "$model_path" || continue

        echo "  Model: $model_name"
        for subset in "${ALL_SUBSETS[@]}"; do
            jsonl="$REPO/datasets/longbench/${subset}.jsonl"
            if [[ ! -f "$jsonl" ]]; then
                echo "    SKIP $subset: dataset not found" | tee -a "$LOG"
                continue
            fi
            mnt="${MAX_NEW_TOKENS[$subset]:-64}"
            outcsv="$OUT_DIR/quality/quality-${model_name}-${subset}.csv"
            echo "    $subset (max_new=$mnt)..."
            "$BINARY_LZ" \
                --model "$model_path" \
                --prompts-file "$jsonl" \
                --pruning-layers $PRUNING_LAYERS \
                --keep-ratios $KEEP_RATIOS \
                --pool-size "$POOL_SIZE" \
                --n-ctx "$N_CTX" \
                --max-tokens "$mnt" \
                --truncation middle \
                --n-gpu-layers "$N_GPU_LAYERS" \
                --repeat 1 \
                --out-csv "$outcsv" \
                2>>"$LOG" \
                || echo "    WARN: lazyllm-run failed for $model_name/$subset" | tee -a "$LOG"
        done
    done
fi

# ── phase 4: score ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 4: Scoring..."
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    [[ ! -f "$model_path" ]] && continue

    summary="$OUT_DIR/quality/longbench_summary_${model_name}.md"
    echo "  Scoring $model_name → $summary"
    python3 "$REPO/scripts/score-longbench.py" \
        --csv-dir "$OUT_DIR/quality/" \
        --model "$model_name" \
        --model-filter "$model_name" \
        --out "$summary" \
        2>>"$LOG" \
        || echo "  WARN: scoring failed for $model_name" | tee -a "$LOG"
done

# ── phase 5: render final report ───────────────────────────────────────────────
REPORT="$OUT_DIR/LAZYLLM_REPORT.md"
echo ""
echo ">>> Phase 5: Rendering final report → $REPORT"

{
cat << HEADER
# LazyLLM Paper Reproduction Report

**Date**: $(date -u +%Y-%m-%d)
**Git SHA**: $GIT_SHA
**Run ID**: $RUN_ID
**GPU**: $GPU_INFO

## Configuration

| Parameter | Value |
|-----------|-------|
| pruning_layers | {$PRUNING_LAYERS} |
| keep_ratios | {$KEEP_RATIOS} |
| pool_kernel_size | $POOL_SIZE |
| n_ctx | $N_CTX |
| n_gpu_layers | $N_GPU_LAYERS |
| repeats (TTFT) | $REPEAT |
| decoding | greedy (temp=0) |
| truncation | middle (head+tail) |

## TTFT Speedup Results

HEADER

for model_name in "${!MODELS[@]}"; do
    ttft_file="$OUT_DIR/ttft/ttft-${model_name}.md"
    if [[ -f "$ttft_file" ]]; then
        echo "### $model_name"
        echo ""
        cat "$ttft_file"
        echo ""
    fi
done

cat << QUALITY
## Quality Results

QUALITY

for model_name in "${!MODELS[@]}"; do
    summary="$OUT_DIR/quality/longbench_summary_${model_name}.md"
    if [[ -f "$summary" ]]; then
        echo "### $model_name"
        echo ""
        cat "$summary"
        echo ""
    fi
done

cat << FOOTER
## Acceptance Gates

See LAZYLLM_PLAN.md §7 for full gate definitions.

| Gate | Threshold | Notes |
|------|-----------|-------|
| G1 Identity | bit-identical logits at kr=1.0 | run \`ctest -L lazyllm\` |
| G4 TTFT speedup 7B | ≥ 2.0× | see TTFT tables above |
| G5 TTFT speedup 13B | ≥ 1.9× | see TTFT tables above |
| G6 Quality aggregate | ≤ 2 pp drop | see quality summary above |
| G7 Multi-doc QA F1 | ≤ 3 pp drop | see quality summary above |

FOOTER
} > "$REPORT"

echo "=================================================================="
echo "Done! Report: $REPORT"
echo "Log:    $LOG"
echo "=================================================================="
