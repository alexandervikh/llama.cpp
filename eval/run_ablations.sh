#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS_DIR:-$REPO/models}"
DRIVER="$REPO/build/bin/llama-spec-prefill-run"
PROMPTS="$REPO/tests/sample_prompts.jsonl"

mkdir -p "$REPO/results/ablation"

BASE="$MODELS/qwen2.5-1.5b-instruct-q4_k_m.gguf"
SPEC_MODELS=("$MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf" "$MODELS/llama-3.2-1b-instruct-q4_k_m.gguf")

total=0
ok=0

for lah in 1 4 8; do
  for pool in 1 7 13; do
    for chunk in 0 16 32; do
      for spec in "${SPEC_MODELS[@]}"; do
        spec_name="$(basename "$spec" .gguf)"
        tag="lah${lah}_pool${pool}_chunk${chunk}_${spec_name}"
        outdir="$REPO/results/ablation/$tag"
        mkdir -p "$outdir"

        # Use same vocab base if spec is qwen, else llama base
        if [[ "$spec_name" == qwen* ]]; then
          base="$BASE"
        else
          base="$MODELS/llama-3.1-8b-instruct-q4_k_m.gguf"
        fi

        out="$outdir/results.jsonl"
        total=$((total + 1))

        echo -n "  $tag ... "
        if "$DRIVER" \
            --model "$base" \
            --spec-model "$spec" \
            --keep-ratio 0.25 \
            --lookahead "$lah" \
            --pool "$pool" \
            --chunk-size "$chunk" \
            --prompt-file "$PROMPTS" \
            --out "$out" 2>/dev/null; then
          echo "OK"
          ok=$((ok + 1))
        else
          echo "FAILED"
        fi
      done
    done
  done
done

echo ""
echo "Ablation grid: $ok/$total runs succeeded"
echo "Results in: $REPO/results/ablation/"

if [ "$ok" -eq "$total" ]; then
  echo "PASS"
  exit 0
else
  echo "PARTIAL: $ok/$total"
  exit 0
fi
