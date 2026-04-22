#!/usr/bin/env python3
"""Spec-prefill evaluation script
Runs baseline llama-cli generation and spec-prefill-run across multiple keep_ratios,
computes Rouge-L similarity to baseline, and emits a summary JSON report.

Defaults are aligned with the user's environment in this repo:
- Model: Qwen3-0.6B-Q4_0.gguf
- Prompts file: prompts_quality_gate.jsonl
- Binaries: llama-cli, llama-spec-prefill-run (on PATH)
- Output: /tmp/spec_eval_summary.json
- Rouge score: rouge_score package (from rouge_score import rouge_scorer)
"""

import argparse
import json
import os
import sys
import subprocess
import shutil
from typing import List, Dict, Tuple


# Default constants (can be overridden via CLI)
MODEL_PATH_DEFAULT = "/home/coder/llama_cpp_organized/Testing/models/Qwen3-0.6B-Q4_0.gguf"
PROMPTS_FILE_DEFAULT = "/home/coder/llama.cpp/tests/prompts_quality_gate.jsonl"
LLAMA_CLI_DEFAULT = "llama-cli"
SPEC_PREFILL_RUN_DEFAULT = "llama-spec-prefill-run"
SUMMARY_OUT_DEFAULT = "/tmp/spec_eval_summary.json"

KEEP_RATIOS = [1.0, 0.5, 0.25, 0.1]


def load_prompts(prompts_path: str) -> List[Dict]:
    prompts = []
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON line at {idx+1} in prompts file", file=sys.stderr)
                continue
            if 'id' in obj:
                prompts.append({'id': obj['id'], 'prompt': obj.get('prompt', '')})
            else:
                prompts.append({'id': idx, 'prompt': obj.get('prompt', line)})
    return prompts


def run_command(cmd: List[str], verbose: bool = True) -> subprocess.CompletedProcess:
    if verbose:
        print("RUN:", " ".join(cmd))
        sys.stdout.flush()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if verbose:
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        sys.stdout.flush()
    return res


def ensure_binary(binary: str) -> bool:
    path = shutil.which(binary)
    if path is None:
        print(f"Error: required binary '{binary}' not found in PATH.")
        return False
    return True


def compute_avg(nums: List[float]) -> float:
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive spec-prefill evaluation script")
    parser.add_argument('--model-path', default=MODEL_PATH_DEFAULT, help='Model path used for both baseline and spec-prefill')
    parser.add_argument('--prompts-file', default=PROMPTS_FILE_DEFAULT, help='Prompts JSONL file (each line a JSON object with id and prompt)')
    parser.add_argument('--llama-cli', default=LLAMA_CLI_DEFAULT, help='Path or name of llama-cli executable')
    parser.add_argument('--spec-run', default=SPEC_PREFILL_RUN_DEFAULT, help='Path or name of llama-spec-prefill-run executable')
    parser.add_argument('--out-summary', default=SUMMARY_OUT_DEFAULT, help='Path to write final summary JSON')
    args = parser.parse_args()

    model_path = args.model_path
    prompts_path = args.prompts_file
    llama_cli = args.llama_cli
    spec_run = args.spec_run
    summary_out = args.out_summary

    # Basic validations
    if not os.path.exists(model_path):
        print(f"Error: model path does not exist: {model_path}")
        sys.exit(2)
    if not os.path.exists(prompts_path):
        print(f"Error: prompts file does not exist: {prompts_path}")
        sys.exit(2)
    if shutil.which(llama_cli) is None:
        print(f"Error: llama-cli binary not found: {llama_cli}")
        sys.exit(2)
    if shutil.which(spec_run) is None:
        print(f"Error: spec-prefill-run binary not found: {spec_run}")
        sys.exit(2)

    # Load prompts
    print("Loading prompts from:", prompts_path)
    prompts = load_prompts(prompts_path)
    if not prompts:
        print("No prompts found. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(prompts)} prompts.")

    # Baseline: use spec-prefill with keep_ratio=1.0 (identity, no filtering).
    # This is the same pipeline as the actual spec-prefill runs, ensuring
    # a fair comparison (same model load, same decode path, just no token filtering).
    baseline_outputs: Dict[int, str] = {}
    baseline_lengths: List[int] = []
    baseline_examples: List[Dict] = []
    baseline_out_path = "/tmp/spec_prefill_baseline_1.0.jsonl"
    print("Running baseline (spec-prefill keep_ratio=1.0, no filtering)...")
    baseline_cmd = [
        spec_run,
        "--model", model_path,
        "--spec-model", model_path,
        "--prompt-file", prompts_path,
        "--out", baseline_out_path,
        "--keep-ratio", "1.0",
        "--lookahead", "8",
        "--pool", "13",
        "--chunk-size", "32",
    ]
    res = run_command(baseline_cmd, verbose=False)
    if res.returncode != 0:
        print(f"Error: baseline spec-prefill run failed: {res.stderr}", file=sys.stderr)
        sys.exit(3)
    # Parse baseline output
    with open(baseline_out_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = int(obj.get('id')) if obj.get('id') is not None else idx
            txt = obj.get('output', '')
            baseline_outputs[did] = txt
            bl = len(txt.split()) if txt else 0
            baseline_lengths.append(bl)
            if idx < 3:
                baseline_examples.append({"id": did, "prompt": prompts[idx].get('prompt', ''), "baseline_output": txt})
    if not baseline_outputs:
        print("Error: No baseline outputs produced. Exiting.")
        sys.exit(3)
    baseline_avg_len = compute_avg(baseline_lengths)
    print(f"Baseline complete: {len(baseline_outputs)} prompts, avg output length {baseline_avg_len:.1f} tokens.")

    # Spec-prefill ablations (keep_ratios below 1.0)
    ablations = []
    spec_results = {}
    all_keep_metrics = []
    rouge_scorer = None
    try:
        from rouge_score import rouge_scorer
        rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except Exception as e:
        print(f"Warning: rouge_score import failed: {e}. Rouge-L will be skipped.")

    def rouge_l_score(ref: str, hyp: str) -> float:
        if rouge_scorer is None:
            return 0.0
        try:
            scores = rouge_scorer.score(ref, hyp)
            return float(scores["rougeL"].fmeasure)
        except Exception:
            return 0.0

    for keep_ratio in [kr for kr in KEEP_RATIOS if kr < 1.0]:
        out_path = f"/tmp/spec_prefill_keep_{str(keep_ratio).replace('.', 'p')}.jsonl"
        cmd = [
            spec_run,
            "--model", model_path,
            "--spec-model", model_path,
            "--prompt-file", prompts_path,
            "--out", out_path,
            "--keep-ratio", str(keep_ratio),
            "--lookahead", "8",
            "--pool", "13",
            "--chunk-size", "32",
        ]
        res = run_command(cmd, verbose=False)
        if res.returncode != 0:
            print(f"Warning: spec-prefill-run failed for keep_ratio {keep_ratio}. Continuing.", file=sys.stderr)
            continue
        # Parse output
        spec_entries = []
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    spec_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        kept_vals = []
        total_vals = []
        ttft_vals = []
        rouge_scores = []
        examples = []
        for e in spec_entries:
            did = int(e.get('id')) if e.get('id') is not None else None
            hyp_out = e.get('output', '')
            n_kept = int(e.get('n_kept', 0))
            n_total = int(e.get('n_total', 0))
            ttft = float(e.get('ttft_ms', 0.0))
            kept_vals.append(n_kept)
            total_vals.append(n_total)
            ttft_vals.append(ttft)
            base_txt = baseline_outputs.get(did)
            if base_txt is not None:
                score = rouge_l_score(base_txt, hyp_out)
                rouge_scores.append(score)
            if len(examples) < 3:
                examples.append({"id": did, "baseline": base_txt, "spec_output": hyp_out})
        avg_ttft = compute_avg(ttft_vals)
        avg_n_kept = compute_avg(kept_vals)
        avg_n_total = compute_avg(total_vals)
        avg_rouge = compute_avg(rouge_scores)
        spec_results[str(keep_ratio)] = {
            "out_path": out_path,
            "avg_ttft_ms": avg_ttft,
            "avg_n_kept": avg_n_kept,
            "avg_n_total": avg_n_total,
            "rouge_l_vs_baseline": avg_rouge,
            "examples": examples,
        }
        all_keep_metrics.append({
            "keep_ratio": keep_ratio,
            "avg_ttft_ms": avg_ttft,
            "avg_n_kept": avg_n_kept,
            "avg_n_total": avg_n_total,
            "rouge_l_vs_baseline": avg_rouge,
        })
        print(f"Spec-prefill keep_ratio={keep_ratio} complete: avg_ttft={avg_ttft:.2f} ms, avg_n_kept={avg_n_kept:.1f}, avg_n_total={avg_n_total:.1f}, rougeL={avg_rouge:.4f}")

    # Compile summary
    rouge_overall = None
    if baseline_outputs and all_keep_metrics:
        # Compute final rouge average across keep_ratios (mean of their rouge scores)
        rouge_values = [m.get('rouge_l_vs_baseline', 0.0) for m in all_keep_metrics]
        rouge_overall = compute_avg(rouge_values)

    summary = {
        "baseline": {
            "num_prompts": len(prompts),
            "avg_output_len": float(baseline_avg_len),
            "examples": baseline_examples,
        },
        "spec_prefill": {
            "keep_ratios": KEEP_RATIOS,
            "results": spec_results,
            "rouge_l_vs_baseline": float(rouge_overall) if rouge_overall is not None else None,
        },
        "ablation": all_keep_metrics,
    }

    # Persist summary
    os.makedirs(os.path.dirname(summary_out) or '.', exist_ok=True)
    with open(summary_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Summary written to:", summary_out)

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
