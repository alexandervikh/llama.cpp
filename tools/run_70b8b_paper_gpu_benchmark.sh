#!/usr/bin/env bash
# Paper-style TTFT sweep: Llama-3.1-70B-Instruct (base) + Llama-3.1-8B-Instruct (draft).
# Intended for multi-GPU hosts (e.g. 4×24GB): set TENSOR_SPLIT to match your GPU count.
#
# Usage:
#   BUILD=build ./tools/run_70b8b_paper_gpu_benchmark.sh
#   SKIP_32K=1 TENSOR_SPLIT=0.5,0.5 ./tools/run_70b8b_paper_gpu_benchmark.sh   # 2 GPUs
#   CPU-only smoke (very slow for 70B; not recommended):
#     CPU_SMOKE=1 ./tools/run_70b8b_paper_gpu_benchmark.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="${BUILD:-build}"
BASE="${BASE:-$ROOT/models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$ROOT/models/llama-3.1-8b-instruct-q4_k_m.gguf}"
GPU_LAYERS="${SPEC_PREFILL_GPU_LAYERS:-99}"
# Even split across 4 GPUs; override for 2- or 3-GPU boxes.
TENSOR_SPLIT="${TENSOR_SPLIT:-0.25,0.25,0.25,0.25}"
OUT_GPU="${OUT_GPU:-$BUILD/ttft_70b8b_paper_gpu}"
CONTEXTS="${CONTEXTS:-2k,4k,8k,16k,32k}"
CLI="$BUILD/bin/llama-cli"
SPEC_RUN="$BUILD/bin/llama-spec-prefill-run"
PY="tools/spec-prefill-ttft-bench.py"

SKIP_FLAGS=()
if [[ "${SKIP_32K:-0}" == "1" ]]; then
  SKIP_FLAGS+=(--skip-32k)
fi

if [[ "${CPU_SMOKE:-0}" == "1" ]]; then
  echo "=== CPU smoke: 8B self-spec, 2k only (not paper 70B+8B) ==="
  PYTHONUNBUFFERED=1 python3 "$PY" \
    --model "$DRAFT" --spec-model "$DRAFT" \
    --llama-cli "$CLI" --spec-run "$SPEC_RUN" \
    -o "${OUT_GPU}_cpu_smoke" --gpu-layers 0 --only-2k --paper --spec-kr 1.0 0.25
  exit 0
fi

echo "=== GPU: 70B + 8B draft, paper kr grid, contexts: $CONTEXTS ==="
PYTHONUNBUFFERED=1 python3 "$PY" \
  --model "$BASE" --spec-model "$DRAFT" \
  --llama-cli "$CLI" --spec-run "$SPEC_RUN" \
  -o "$OUT_GPU" --gpu-layers "$GPU_LAYERS" \
  --tensor-split "$TENSOR_SPLIT" \
  --contexts "$CONTEXTS" \
  "${SKIP_FLAGS[@]}" \
  --paper

echo "Results: $OUT_GPU/ttft_results.json"
