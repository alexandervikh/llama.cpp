#!/usr/bin/env bash
# Paper-aligned TTFT sweeps (hyperparams: lookahead=8, pool=13, chunk=32; kr includes 0.10 from Fig. 3).
# Default models: Llama-3.1-8B + Llama-3.2-1B (same *family* as paper's Llama stack; 70B/405B not required here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="${BUILD:-build}"
BASE="${BASE:-$ROOT/models/llama-3.1-8b-instruct-q4_k_m.gguf}"
DRAFT="${DRAFT:-$ROOT/models/llama-3.2-1b-instruct-q4_k_m.gguf}"
GPU_LAYERS="${SPEC_PREFILL_GPU_LAYERS:-99}"
OUT_GPU="${OUT_GPU:-$ROOT/build/paper_ttft_gpu}"
OUT_CPU="${OUT_CPU:-$ROOT/build/paper_ttft_cpu}"

CLI="$BUILD/bin/llama-cli"
SPEC_RUN="$BUILD/bin/llama-spec-prefill-run"
PY="tools/spec-prefill-ttft-bench.py"

echo "=== GPU (CUDA): paper kr grid, 2k+8k (skip 32k for runtime) ==="
PYTHONUNBUFFERED=1 python3 "$PY" \
  --model "$BASE" --spec-model "$DRAFT" \
  --llama-cli "$CLI" --spec-run "$SPEC_RUN" \
  -o "$OUT_GPU" --gpu-layers "$GPU_LAYERS" --skip-32k --paper

echo ""
echo "=== CPU (quick smoke): 1B self-spec, 2k ctx only, kr 1.0 + 0.25 ==="
if [[ "${SKIP_CPU_BENCH:-0}" == "1" ]]; then
  echo "SKIP_CPU_BENCH=1 — skipping CPU (set to 0 to run)."
  exit 0
fi
CPU_BASE="${CPU_BASE:-$ROOT/models/llama-3.2-1b-instruct-q4_k_m.gguf}"
PYTHONUNBUFFERED=1 python3 "$PY" \
  --model "$CPU_BASE" --spec-model "$CPU_BASE" \
  --llama-cli "$CLI" --spec-run "$SPEC_RUN" \
  -o "$OUT_CPU" --gpu-layers 0 --skip-32k --only-2k \
  --spec-kr 1.0 0.25
