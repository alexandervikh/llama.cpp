#!/usr/bin/env python3
"""Score quality gate outputs with Rouge-L. Compares kr=0.25/0.10 against kr=1.0 baseline."""
import json, argparse, sys
from rouge_score import rouge_scorer

def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row["output"]
    return rows

def score(baseline_outputs, test_outputs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for id_, base_text in baseline_outputs.items():
        if id_ not in test_outputs:
            continue
        test_text = test_outputs[id_]
        s = scorer.score(base_text, test_text)
        scores.append(s["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="JSONL output at kr=1.0")
    ap.add_argument("--test", required=True, help="JSONL output at kr=0.25 or kr=0.10")
    ap.add_argument("--threshold", type=float, default=0.70, help="Min fraction of baseline Rouge-L")
    ap.add_argument("--label", default="test")
    args = ap.parse_args()

    baseline = load_jsonl(args.baseline)
    test = load_jsonl(args.test)

    base_self_score = score(baseline, baseline)
    test_score = score(baseline, test)

    ratio = test_score / base_self_score if base_self_score > 0 else 0.0

    print(f"=== Quality Gate: {args.label} ===")
    print(f"Baseline self Rouge-L : {base_self_score:.4f}")
    print(f"Test Rouge-L          : {test_score:.4f}")
    print(f"Ratio (test/baseline) : {ratio:.4f}  (threshold: {args.threshold:.2f})")

    if ratio >= args.threshold:
        print(f"PASS: {ratio:.4f} >= {args.threshold:.2f}")
        sys.exit(0)
    else:
        print(f"FAIL: {ratio:.4f} < {args.threshold:.2f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
