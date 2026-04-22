#!/usr/bin/env python3
"""QPS benchmarks for spec-prefill -- concurrent request throughput."""
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "/home/coder/llama_cpp_organized/Testing/models/Qwen3-0.6B-Q4_0.gguf"
SPEC_RUN = "/home/coder/llama.cpp/build/bin/llama-spec-prefill-run"
PROMPTS_FILE = "/home/coder/llama.cpp/tests/prompts_quality_gate.jsonl"
OUT_DIR = "/tmp/spec_qps"


def run_single(prompt_file, out_path, kr, lah, pool, chunk):
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
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=120)
        elapsed = time.time() - t0
        if res.returncode != 0:
            return {"error": res.stderr[:300], "elapsed_s": elapsed, "ok": False}
        with open(out_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return {
                    "ok": True,
                    "elapsed_s": elapsed,
                    "ttft_ms": obj.get("ttft_ms", 0),
                    "n_kept": obj.get("n_kept", 0),
                    "n_total": obj.get("n_total", 0),
                    "output_tokens": len(obj.get("output", "").split()),
                }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed_s": 120, "ok": False}
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.time() - t0, "ok": False}
    return {"error": "no output", "elapsed_s": time.time() - t0, "ok": False}


def parse_prompts():
    prompts = []
    with open(PROMPTS_FILE, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append((i, obj.get("prompt", line)))
            except json.JSONDecodeError:
                prompts.append((i, line))
    return prompts


def run_qps_round(n_concurrent, prompts_subset, kr, lah, pool, chunk):
    tmp_dir = f"{OUT_DIR}/tmp_{n_concurrent}_{int(time.time()*1000)}"
    os.makedirs(tmp_dir, exist_ok=True)

    pf_paths = []
    for idx, prompt_text in prompts_subset:
        pf = f"{tmp_dir}/prompt_{idx}.jsonl"
        with open(pf, 'w') as f:
            f.write(json.dumps({"id": idx, "prompt": prompt_text}) + "\n")
        pf_paths.append(pf)

    start = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        for i, pf in enumerate(pf_paths):
            out_path = f"{tmp_dir}/out_{i}.jsonl"
            futures.append(executor.submit(run_single, pf, out_path, kr, lah, pool, chunk))

        results = []
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_time = time.time() - start

    return results, wall_time


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"concurrency_sweep": []}

    print("=" * 70)
    print("QPS Benchmark: Concurrent Spec-Prefill Throughput")
    print("=" * 70)

    prompts = parse_prompts()
    print(f"\nLoaded {len(prompts)} prompts from quality gate file")

    concurrency_levels = [1, 2, 4, 8]
    kr, lah, pool, chunk = 0.25, 8, 13, 32

    for n in concurrency_levels:
        print(f"\n--- Concurrency: {n} concurrent requests ---")
        print(f"  Prompts: {len(prompts)}, params: kr={kr} lah={lah} pool={pool} chunk={chunk}")

        results_list, wall_time = run_qps_round(n, prompts, kr, lah, pool, chunk)

        ok_results = [r for r in results_list if r.get("ok")]
        failed = [r for r in results_list if not r.get("ok")]

        if ok_results:
            total_requests = len(ok_results)
            total_tokens_in = sum(r.get("n_total", 0) for r in ok_results)
            total_tokens_out = sum(r.get("output_tokens", 0) for r in ok_results)
            avg_latency = sum(r.get("elapsed_s", 0) for r in ok_results) / total_requests
            avg_ttft = sum(r.get("ttft_ms", 0) for r in ok_results) / total_requests
            qps = total_requests / wall_time
            throughput = total_tokens_in / wall_time
            gen_throughput = total_tokens_out / wall_time

            print(f"  Wall time:     {wall_time:.2f}s")
            print(f"  QPS:           {qps:.2f} req/s")
            print(f"  Avg latency:   {avg_latency:.2f}s")
            print(f"  Avg TTFT:      {avg_ttft:.1f}ms")
            print(f"  Throughput:    {throughput:.0f} tok/s (input)")
            print(f"  Gen throughput:{gen_throughput:.0f} tok/s (output)")
            print(f"  Succeeded:     {len(ok_results)}/{total_requests}")
            if failed:
                print(f"  Failed:        {len(failed)} ({failed[0].get('error', '?')})")

            results["concurrency_sweep"].append({
                "concurrency": n,
                "wall_time_s": round(wall_time, 3),
                "qps": round(qps, 3),
                "avg_latency_s": round(avg_latency, 3),
                "avg_ttft_ms": round(avg_ttft, 2),
                "input_throughput_tps": round(throughput, 2),
                "gen_throughput_tps": round(gen_throughput, 2),
                "succeeded": len(ok_results),
                "failed": len(failed),
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
            })
        else:
            print(f"  All {len(results_list)} requests failed!")
            results["concurrency_sweep"].append({
                "concurrency": n,
                "status": "all_failed",
                "error": results_list[0].get("error", "unknown") if results_list else "no results",
            })

    out_path = f"{OUT_DIR}/qps_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
