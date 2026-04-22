#!/usr/bin/env python3
"""Section 5: Ablations — sweep lookahead, pool kernel, chunk vs percentage."""
import json
import os
import subprocess
import sys

MODEL = "/home/coder/llama_cpp_organized/Testing/models/Qwen3-0.6B-Q4_0.gguf"
SPEC_RUN = "/home/coder/llama.cpp/build/bin/llama-spec-prefill-run"
PROMPTS = "/home/coder/llama.cpp/tests/prompts_quality_gate.jsonl"
OUT_DIR = "/tmp/spec_ablations"

def run_spec(prompt_file, out_path, kr, lah, pool, chunk):
    cmd = [
        SPEC_RUN,
        "--model", MODEL,
        "--spec-model", MODEL,
        "--prompt-file", prompt_file,
        "--out", out_path,
        "--keep-ratio", str(kr),
        "--lookahead", str(lah),
        "--pool", str(pool),
        "--chunk-size", str(chunk),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if res.returncode != 0:
            return None
        return parse_out(out_path)
    except Exception:
        return None

def parse_out(path):
    results = []
    if not os.path.exists(path):
        return results
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append({
                    "id": obj.get("id", -1),
                    "ttft_ms": obj.get("ttft_ms", 0),
                    "n_kept": obj.get("n_kept", 0),
                    "n_total": obj.get("n_total", 0),
                    "output": obj.get("output", ""),
                    "output_len": len(obj.get("output", "").split()),
                })
            except json.JSONDecodeError:
                continue
    return results

def avg(nums):
    return sum(nums) / len(nums) if nums else 0

def rouge_l(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return sc.score(ref, hyp)["rougeL"].fmeasure
    except Exception:
        return 0.0

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    print("=" * 70)
    print("Section 5: Ablations")
    print("=" * 70)

    # Baseline: kr=1.0, lah=8, pool=13, chunk=32
    print("\n--- Running baseline (kr=1.0) for Rouge-L comparison ---")
    base_path = f"{OUT_DIR}/baseline.jsonl"
    base = run_spec(PROMPTS, base_path, 1.0, 8, 13, 32)
    if not base:
        print("Error: baseline failed")
        return 1
    print(f"  Baseline: {len(base)} prompts processed")

    # --- Ablation 1: Lookahead count sweep ---
    print("\n--- Ablation 1: Lookahead count sweep ---")
    lah_results = []
    for lah in [1, 4, 8, 16]:
        label = f"lah{lah}"
        path = f"{OUT_DIR}/ablation_lah{lah}.jsonl"
        print(f"  lah={lah}...", end=" ")
        r = run_spec(PROMPTS, path, 0.25, lah, 13, 32)
        if r and len(r) > 0:
            avg_ttft = avg([x["ttft_ms"] for x in r])
            avg_kept = avg([x["n_kept"] for x in r])
            avg_out = avg([x["output_len"] for x in r])
            rouges = [rouge_l(base[i]["output"], r[i]["output"]) for i in range(min(len(base), len(r)))]
            avg_rouge = avg(rouges)
            print(f"ttft={avg_ttft:.1f}ms kept={avg_kept:.0f} out={avg_out:.0f} rougeL={avg_rouge:.4f}")
            lah_results.append({"lookahead": lah, "avg_ttft_ms": round(avg_ttft, 2), "avg_n_kept": round(avg_kept), "avg_output_len": round(avg_out), "rouge_l": round(avg_rouge, 4), "n_prompts": len(r)})
        else:
            print("FAILED")
            lah_results.append({"lookahead": lah, "status": "failed"})
    results["lookahead_sweep"] = lah_results

    # --- Ablation 2: Pool kernel size sweep ---
    print("\n--- Ablation 2: Pool kernel size sweep ---")
    pool_results = []
    for pool in [1, 7, 13]:
        label = f"pool{pool}"
        path = f"{OUT_DIR}/ablation_pool{pool}.jsonl"
        print(f"  pool={pool}...", end=" ")
        r = run_spec(PROMPTS, path, 0.25, 8, pool, 32)
        if r and len(r) > 0:
            avg_ttft = avg([x["ttft_ms"] for x in r])
            avg_kept = avg([x["n_kept"] for x in r])
            avg_out = avg([x["output_len"] for x in r])
            rouges = [rouge_l(base[i]["output"], r[i]["output"]) for i in range(min(len(base), len(r)))]
            avg_rouge = avg(rouges)
            print(f"ttft={avg_ttft:.1f}ms kept={avg_kept:.0f} out={avg_out:.0f} rougeL={avg_rouge:.4f}")
            pool_results.append({"pool_kernel": pool, "avg_ttft_ms": round(avg_ttft, 2), "avg_n_kept": round(avg_kept), "avg_output_len": round(avg_out), "rouge_l": round(avg_rouge, 4), "n_prompts": len(r)})
        else:
            print("FAILED")
            pool_results.append({"pool_kernel": pool, "status": "failed"})
    results["pool_kernel_sweep"] = pool_results

    # --- Ablation 3: Chunk vs Percentage strategies ---
    print("\n--- Ablation 3: Chunk vs Percentage strategies ---")
    chunk_results = []
    for chunk_size in [0, 16, 32]:
        mode = "percentage" if chunk_size == 0 else f"chunk_{chunk_size}"
        path = f"{OUT_DIR}/ablation_chunk{chunk_size}.jsonl"
        print(f"  {mode}...", end=" ")
        r = run_spec(PROMPTS, path, 0.25, 8, 13, chunk_size)
        if r and len(r) > 0:
            avg_ttft = avg([x["ttft_ms"] for x in r])
            avg_kept = avg([x["n_kept"] for x in r])
            avg_out = avg([x["output_len"] for x in r])
            rouges = [rouge_l(base[i]["output"], r[i]["output"]) for i in range(min(len(base), len(r)))]
            avg_rouge = avg(rouges)
            print(f"ttft={avg_ttft:.1f}ms kept={avg_kept:.0f} out={avg_out:.0f} rougeL={avg_rouge:.4f}")
            chunk_results.append({"strategy": mode, "avg_ttft_ms": round(avg_ttft, 2), "avg_n_kept": round(avg_kept), "avg_output_len": round(avg_out), "rouge_l": round(avg_rouge, 4), "n_prompts": len(r)})
        else:
            print("FAILED")
            chunk_results.append({"strategy": mode, "status": "failed"})
    results["chunk_vs_percentage"] = chunk_results

    # --- Save ---
    out_path = f"{OUT_DIR}/ablation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
