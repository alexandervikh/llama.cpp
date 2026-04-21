#!/usr/bin/env python3
"""Score LongBench outputs: compare kr variants against kr=1.0 baseline per task."""
import json, argparse, os, sys
from rouge_score import rouge_scorer

def load_outputs(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows[r["id"]] = r.get("output", "")
            except Exception:
                pass
    return rows

def rouge_l_score(ref_outputs, hyp_outputs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for id_, ref in ref_outputs.items():
        if id_ not in hyp_outputs:
            continue
        hyp = hyp_outputs[id_]
        scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "gov_report", "passage_retrieval_en"])
    ap.add_argument("--keep-ratios", nargs="+", type=float, default=[0.1, 0.25])
    args = ap.parse_args()

    print("=" * 70)
    print(f"{'Task':<25} {'KR':<6} {'RougeL':<10} {'vs Baseline':<12} {'Status'}")
    print("=" * 70)

    task_results = {}
    overall_pass = True

    for task in args.tasks:
        baseline_path = os.path.join(args.results_dir, f"{task}_kr1.0.jsonl")
        baseline = load_outputs(baseline_path)
        if not baseline:
            print(f"{task:<25} {'---':<6} {'N/A':<10} {'N/A':<12} SKIP (no baseline)")
            continue

        base_self = rouge_l_score(baseline, baseline)
        task_results[task] = {"baseline_rouge": base_self, "ratios": {}}

        for kr in args.keep_ratios:
            test_path = os.path.join(args.results_dir, f"{task}_kr{kr}.jsonl")
            test = load_outputs(test_path)
            if not test:
                print(f"{task:<25} {kr:<6} {'N/A':<10} {'N/A':<12} SKIP (no output)")
                continue

            test_score = rouge_l_score(baseline, test)
            ratio = test_score / base_self if base_self > 0 else 0.0
            threshold = 0.70 if kr >= 0.25 else 0.50
            status = "PASS" if ratio >= threshold else "FAIL"
            if status == "FAIL":
                overall_pass = False

            print(f"{task:<25} {kr:<6} {test_score:<10.4f} {ratio:<12.4f} {status}")
            task_results[task]["ratios"][kr] = {"rouge": test_score, "ratio": ratio, "status": status}

    print("=" * 70)

    # Summary
    kr_results = {}
    for kr in args.keep_ratios:
        ratios = [task_results[t]["ratios"].get(kr, {}).get("ratio", None)
                  for t in task_results if kr in task_results[t].get("ratios", {})]
        ratios = [r for r in ratios if r is not None]
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            threshold = 0.70 if kr >= 0.25 else 0.50
            status = "PASS" if mean_ratio >= threshold else "FAIL"
            print(f"Mean ratio at kr={kr}: {mean_ratio:.4f} -> {status}")
            kr_results[kr] = {"mean_ratio": mean_ratio, "status": status}

    print("=" * 70)
    print("Overall:", "PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())
