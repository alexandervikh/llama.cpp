#!/usr/bin/env python3
"""Section 4: Performance Evaluation — TTFT benchmarks at different context lengths.

Measures Time-To-First-Token (TTFT) for baseline llama-cli vs spec-prefill-run
at 2k, 8k, and 32k prompt context lengths.

Uses llama-cli --perf for internal timings and llama-spec-prefill-run for
spec-prefill TTFT (captured in JSONL output).
"""

import json
import os
import random
import re
import subprocess
import sys


MODEL_PATH = "/home/coder/llama_cpp_organized/Testing/models/Qwen3-0.6B-Q4_0.gguf"
LLAMA_CLI = "llama-cli"
SPEC_RUN = "llama-spec-prefill-run"
OUTPUT_DIR = "/tmp/spec_perf_ttft"


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
        "dolphins", "play", "in", "the", "surf", "seagulls", "fly",
        "above", "the", "water", "the", "sun", "sets", "horizon",
        "painting", "sky", "orange", "pink", "and", "purple", "stars",
        "appear", "night", "moon", "rises", "silver", "light",
    ]
    prompt_tokens = []
    while len(prompt_tokens) < token_count:
        prompt_tokens.append(rng.choice(words))
    return " ".join(prompt_tokens[:token_count])


def run_llama_cli_ttft(prompt, ctx_size, label):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-c", str(ctx_size),
        "-n", "0",
        "--perf",
        "-p", prompt,
        "-ot", "1",
        "--threads", "4",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "timeout", "error": "timeout after 120s"}

    stderr = res.stderr
    result = {"label": label, "status": "ok", "stderr": stderr}

    m = re.search(r'prompt eval time\s*=\s*([\d.]+)\s*ms', stderr)
    if m:
        result["prompt_eval_ms"] = float(m.group(1))
    else:
        m = re.search(r'prompt.*time\s*=\s*([\d.]+)', stderr)
        if m:
            result["prompt_eval_ms"] = float(m.group(1))

    m = re.search(r'eval time\s*=\s*([\d.]+)\s*ms', stderr)
    if m:
        result["eval_ms"] = float(m.group(1))

    return result


def run_spec_prefill_ttft(prompt, ctx_size, kr, lah, pool, chunk):
    out_path = f"{OUTPUT_DIR}/ttft_{ctx_size}k_kr{kr}.jsonl"
    prompt_path = f"{OUTPUT_DIR}/prompt_{ctx_size}k.jsonl"

    with open(prompt_path, 'w') as f:
        f.write(json.dumps({"id": 0, "prompt": prompt}) + "\n")

    cmd = [
        SPEC_RUN,
        "--model", MODEL_PATH,
        "--spec-model", MODEL_PATH,
        "--prompt-file", prompt_path,
        "--out", out_path,
        "--keep-ratio", str(kr),
        "--lookahead", str(lah),
        "--pool", str(pool),
        "--chunk-size", str(chunk),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"label": f"spec-ctx{ctx_size}k-kr{kr}", "status": "timeout"}

    if res.returncode != 0:
        return {"label": f"spec-ctx{ctx_size}k-kr{kr}", "status": "error", "stderr": res.stderr[:500]}

    result = {"label": f"spec-ctx{ctx_size}k-kr{kr}", "status": "ok"}
    with open(out_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result["ttft_ms"] = float(obj.get("ttft_ms", 0))
                result["n_kept"] = int(obj.get("n_kept", 0))
                result["n_total"] = int(obj.get("n_total", 0))
                result["output_len"] = len(obj.get("output", "").split())
            except json.JSONDecodeError:
                continue

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    contexts = [
        (2048, "2k"),
        (8192, "8k"),
        (32768, "32k"),
    ]

    spec_params = [
        {"kr": 1.0, "lah": 8, "pool": 13, "chunk": 32},
        {"kr": 0.5, "lah": 8, "pool": 13, "chunk": 32},
        {"kr": 0.25, "lah": 8, "pool": 13, "chunk": 32},
    ]

    results = {"ttft": [], "overhead": []}

    print("=" * 70)
    print("Section 4: Performance Evaluation — TTFT Benchmarks")
    print("=" * 70)

    for ctx_size, ctx_label in contexts:
        print(f"\n--- Context: {ctx_label} ({ctx_size} tokens) ---")

        prompt = generate_prompt(ctx_size)

        print(f"  Baseline llama-cli TTFT ({ctx_label})...", end=" ")
        base = run_llama_cli_ttft(prompt, ctx_size, f"baseline-{ctx_label}")
        if base["status"] == "ok" and "prompt_eval_ms" in base:
            print(f"{base['prompt_eval_ms']:.1f} ms")
            base["prompt_eval_ms"] = round(base["prompt_eval_ms"], 2)
            results["ttft"].append({"context": ctx_label, "baseline_ms": base["prompt_eval_ms"]})
        else:
            print(f"FAILED ({base['status']})")
            results["ttft"].append({"context": ctx_label, "baseline_ms": None, "error": base.get("status")})

        for sp in spec_params:
            spec = run_spec_prefill_ttft(prompt, ctx_size, sp["kr"], sp["lah"], sp["pool"], sp["chunk"])
            label = f"spec-{ctx_label}-kr{sp['kr']}"
            if spec["status"] == "ok" and "ttft_ms" in spec:
                print(f"  {label}: {spec['ttft_ms']:.1f} ms (kept={spec.get('n_kept', '?')}/{spec.get('n_total', '?')}, output={spec.get('output_len', '?')})")
                results["ttft"].append({
                    "context": ctx_label,
                    "keep_ratio": sp["kr"],
                    "ttft_ms": round(spec["ttft_ms"], 2),
                    "n_kept": spec.get("n_kept"),
                    "n_total": spec.get("n_total"),
                })
                if "prompt_eval_ms" in base:
                    overhead_pct = ((spec["ttft_ms"] - base["prompt_eval_ms"]) / base["prompt_eval_ms"] * 100) if base["prompt_eval_ms"] > 0 else 0
                    results["overhead"].append({
                        "context": ctx_label,
                        "keep_ratio": sp["kr"],
                        "baseline_ms": base["prompt_eval_ms"],
                        "spec_ttft_ms": spec["ttft_ms"],
                        "overhead_pct": round(overhead_pct, 2),
                    })
            else:
                print(f"  {label}: FAILED ({spec['status']})")
                results["ttft"].append({"context": ctx_label, "keep_ratio": sp["kr"], "status": spec["status"]})

    out_path = f"{OUTPUT_DIR}/ttft_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 70)
    print("Overhead Summary (spec-prefill TTFT vs baseline TTFT)")
    print("=" * 70)
    for o in results["overhead"]:
        ctx = o["context"]
        kr = o["keep_ratio"]
        overhead = o["overhead_pct"]
        print(f"  {ctx} | kr={kr:.2f} | baseline={o['baseline_ms']:.0f}ms | spec={o['spec_ttft_ms']:.0f}ms | overhead={overhead:+.1f}%")

    print(f"\nSummary saved to: {out_path}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
