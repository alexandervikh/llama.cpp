#!/usr/bin/env python3
"""Section 4: Performance Evaluation — TTFT & throughput benchmarks."""
import json
import os
import random
import subprocess
import sys


MODEL = "/home/coder/llama_cpp_organized/Testing/models/Qwen3-0.6B-Q4_0.gguf"
LLAMA_BENCH = "/home/coder/llama.cpp/build/bin/llama-bench"
SPEC_RUN = "/home/coder/llama.cpp/build/bin/llama-spec-prefill-run"
OUT_DIR = "/tmp/spec_perf"

def generate_prompt(token_count, seed=42):
    rng = random.Random(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "in", "a", "field", "of", "green", "grass", "under", "the", "sun",
        "and", "the", "sky", "is", "blue", "birds", "are", "singing",
        "the", "river", "flows", "through", "the", "valley", "mountains",
        "stand", "tall", "trees", "sway", "in", "the", "wind", "flowers",
        "bloom", "in", "spring", "leaves", "fall", "in", "autumn",
        "winter", "brings", "snow", "summer", "brings", "warmth",
        "the", "ocean", "waves", "crash", "against", "the", "shore",
        "fish", "swim", "in", "the", "deep", "sea", "whales", "sing",
    ]
    tokens = []
    while len(tokens) < token_count:
        tokens.append(rng.choice(words))
    return " ".join(tokens[:token_count])

def run_llama_bench(n_prompt, n_gen, label):
    cmd = [
        LLAMA_BENCH,
        "-m", MODEL,
        "-ngl", "99",
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-o", "json",
        "--no-warmup",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if res.returncode != 0:
            return None
        data = json.loads(res.stdout)
        for d in data:
            if d.get("n_prompt") == n_prompt and d.get("n_gen") == n_gen:
                return {"label": label, "type": d.get("type", "?"), "tps": d.get("avg_ts"), "tokens": n_gen}
    except Exception:
        pass
    return None

def run_spec_prefill(prompt, kr, lah, pool, chunk, label):
    out_path = f"{OUT_DIR}/{label}.jsonl"
    pf = f"{OUT_DIR}/{label}_prompt.jsonl"
    with open(pf, 'w') as f:
        f.write(json.dumps({"id": 0, "prompt": prompt}) + "\n")
    cmd = [
        SPEC_RUN,
        "--model", MODEL,
        "--spec-model", MODEL,
        "--prompt-file", pf,
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
        with open(out_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return {
                    "label": label,
                    "ttft_ms": obj.get("ttft_ms", 0),
                    "n_kept": obj.get("n_kept", 0),
                    "n_total": obj.get("n_total", 0),
                    "output_len": len(obj.get("output", "").split()),
                }
    except Exception:
        pass
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"throughput": [], "ttft": [], "overhead": []}

    print("=" * 70)
    print("Section 4: Performance Evaluation")
    print("=" * 70)

    # --- Throughput benchmarks at different context lengths ---
    print("\n--- Throughput Benchmarks (llama-bench) ---")
    for pp, label in [(512, "pp512"), (2048, "pp2k"), (8192, "pp8k"), (32768, "pp32k")]:
        r = run_llama_bench(pp, 0, f"baseline-{label}")
        if r:
            print(f"  {label}: {r['tps']:.1f} t/s (prompt={pp} tokens)")
            results["throughput"].append({"mode": "prompt", "context": label, "tokens": pp, "tps": round(r["tps"], 2)})
        r2 = run_llama_bench(0, 128, f"baseline-tg128-{label}")
        if r2:
            print(f"  tg128: {r2['tps']:.1f} t/s (generation=128 tokens)")
            results["throughput"].append({"mode": "generation", "context": label, "tokens": 128, "tps": round(r2["tps"], 2)})

    # --- TTFT benchmarks via spec-prefill-run JSONL output ---
    print("\n--- TTFT Benchmarks (spec-prefill-run JSONL, quality gate prompts) ---")
    PROMPTS_FILE = "/home/coder/llama.cpp/tests/prompts_quality_gate.jsonl"
    spec_params_ttft = [
        {"kr": 1.0, "lah": 8, "pool": 13, "chunk": 32, "label": "ttft-kr1.0-lah8"},
        {"kr": 0.5, "lah": 8, "pool": 13, "chunk": 32, "label": "ttft-kr0.5-lah8"},
        {"kr": 0.25, "lah": 8, "pool": 13, "chunk": 32, "label": "ttft-kr0.25-lah8"},
        {"kr": 0.1, "lah": 8, "pool": 13, "chunk": 32, "label": "ttft-kr0.1-lah8"},
    ]

    for sp in spec_params_ttft:
        r = run_spec_prefill("", sp["kr"], sp["lah"], sp["pool"], sp["chunk"], sp["label"])
        out_path = f"{OUT_DIR}/{sp['label']}.jsonl"
        pf = f"{OUT_DIR}/{sp['label']}_prompt.jsonl"
        cmd = [
            SPEC_RUN,
            "--model", MODEL,
            "--spec-model", MODEL,
            "--prompt-file", PROMPTS_FILE,
            "--out", out_path,
            "--keep-ratio", str(sp["kr"]),
            "--lookahead", str(sp["lah"]),
            "--pool", str(sp["pool"]),
            "--chunk-size", str(sp["chunk"]),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if res.returncode != 0:
                print(f"    {sp['label']}: FAILED ({res.stderr[:200]})")
                continue
            avg_ttft = 0
            n_kept_avg = 0
            n_total_avg = 0
            count = 0
            with open(out_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    avg_ttft += obj.get("ttft_ms", 0)
                    n_kept_avg += obj.get("n_kept", 0)
                    n_total_avg += obj.get("n_total", 0)
                    count += 1
            if count > 0:
                avg_ttft /= count
                n_kept_avg /= count
                n_total_avg /= count
                print(f"    {sp['label']}: avg_ttft={avg_ttft:.1f}ms avg_kept={n_kept_avg:.0f}/{n_total_avg:.0f} (n={count})")
                results["ttft"].append({
                    "context": "quality_gate", "tokens": 0,
                    "keep_ratio": sp["kr"], "ttft_ms": round(avg_ttft, 2),
                    "n_kept": round(n_kept_avg), "n_total": round(n_total_avg), "count": count,
                })
        except Exception as e:
            print(f"    {sp['label']}: ERROR ({e})")

    # --- Save ---
    out_path = f"{OUT_DIR}/perf_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
