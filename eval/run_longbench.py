#!/usr/bin/env python3
"""Run speculative prefill driver across LongBench tasks and keep-ratios."""
import argparse, subprocess, json, os, sys

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "gov_report", "passage_retrieval_en"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="Path to llama-spec-prefill-run binary")
    ap.add_argument("--base", required=True, help="Base model GGUF path")
    ap.add_argument("--spec", required=True, help="Spec model GGUF path")
    ap.add_argument("--keep-ratios", nargs="+", type=float, default=[0.1, 0.25, 1.0])
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--n-per-task", type=int, default=20, help="Prompts per task")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra args to driver")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic prompts if LongBench unavailable")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def try_load_task(task):
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            rows = []
            for i, r in enumerate(ds):
                if i >= args.n_per_task:
                    break
                prompt = r.get("context", "") + "\n\n" + r.get("input", "")
                answer = r.get("answers", [""])[0] if r.get("answers") else ""
                rows.append({"id": i, "prompt": prompt, "answer": answer})
            return rows
        except Exception as e:
            print(f"  Could not load {task}: {e}")
            return None

    def synthetic_prompts(task, n):
        passage = (
            f"This is a synthetic long-context document for testing the {task} task. "
            "It contains multiple paragraphs of text designed to evaluate how well the "
            "speculative prefill algorithm retains relevant information while filtering tokens. "
            "The document discusses various topics including history, science, technology, "
            "and culture to provide diverse content for quality evaluation. "
        ) * 20
        return [{"id": i, "prompt": f"Document: {passage}\n\nQuestion: Summarize the main topics. Answer:", "answer": ""} for i in range(n)]

    results_summary = []

    for task in TASKS:
        print(f"\n=== Task: {task} ===")
        rows = try_load_task(task)
        if rows is None:
            print(f"  Using synthetic prompts for {task}")
            rows = synthetic_prompts(task, args.n_per_task)

        prompts_path = os.path.join(args.out, f"{task}_prompts.jsonl")
        with open(prompts_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(rows)} prompts to {prompts_path}")

        for kr in args.keep_ratios:
            out_path = os.path.join(args.out, f"{task}_kr{kr}.jsonl")
            cmd = [
                args.driver,
                "--model", args.base,
                "--spec-model", args.spec,
                "--keep-ratio", str(kr),
                "--lookahead", "8",
                "--pool", "13",
                "--prompt-file", prompts_path,
                "--out", out_path,
            ] + list(args.extra)
            print(f"  Running kr={kr}... ", end="", flush=True)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    print(f"OK -> {out_path}")
                    results_summary.append({"task": task, "kr": kr, "status": "ok", "out": out_path})
                else:
                    print(f"FAILED: {r.stderr[-200:]}")
                    results_summary.append({"task": task, "kr": kr, "status": "failed", "stderr": r.stderr[-200:]})
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                results_summary.append({"task": task, "kr": kr, "status": "timeout"})

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    ok = sum(1 for r in results_summary if r["status"] == "ok")
    total = len(results_summary)
    print(f"\nCompleted: {ok}/{total} runs succeeded")
    expected = len(TASKS) * len(args.keep_ratios)
    if ok == expected:
        print(f"PASS: all {expected} output files created")
    else:
        print(f"PARTIAL: {ok}/{expected} output files created")

if __name__ == "__main__":
    main()
