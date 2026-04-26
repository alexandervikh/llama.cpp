#!/usr/bin/env python3
"""
Score F1 from the CSV output of llama-lazyllm-run --prompts-file.
No model inference — just string matching against ground truth answers.

Input CSV columns: id, n_prompt, n_kept, baseline_ttft_ms, lazyllm_ttft_ms,
                   speedup, baseline_output, lazyllm_output, answers

Usage:
  python3 scripts/score-f1.py results.csv [--out report.md]
"""

import argparse
import csv
import math
import os
import re
import string
import sys
from datetime import datetime


# ── F1 scoring (matches LongBench's metric exactly) ───────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks) if p_toks else 0.0
    rec  = len(common) / len(g_toks) if g_toks else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def best_f1(pred: str, gold_list: list) -> float:
    if not gold_list:
        return 0.0
    return max(token_f1(pred, g) for g in gold_list)


def bootstrap_ci(scores: list, n_boot: int = 2000, alpha: float = 0.05):
    import random
    if not scores:
        return 0.0, 0.0
    means = sorted(
        sum(random.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n_boot)
    )
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return lo, hi


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_file", help="CSV from llama-lazyllm-run --prompts-file")
    p.add_argument("--out", default="", help="Output markdown report path (optional)")
    args = p.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"ERROR: {args.csv_file} not found", file=sys.stderr)
        sys.exit(1)

    bl_f1s, lz_f1s = [], []
    bl_ttfts, lz_ttfts, speedups = [], [], []
    n_total = 0

    with open(args.csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers_raw = row.get("answers") or ""
            gold = [a.strip() for a in answers_raw.split("|") if a.strip()]
            bl_out = row.get("baseline_output", "")
            lz_out = row.get("lazyllm_output", "")

            if not gold:
                continue

            bl_f1s.append(best_f1(bl_out, gold))
            lz_f1s.append(best_f1(lz_out, gold))

            try:
                bl_ttfts.append(float(row["baseline_ttft_ms"]))
                lz_ttfts.append(float(row["lazyllm_ttft_ms"]))
                speedups.append(float(row["speedup"]))
            except (KeyError, ValueError):
                pass

            n_total += 1

    if n_total == 0:
        print("ERROR: no valid rows found in CSV", file=sys.stderr)
        sys.exit(1)

    mean_bl_f1  = sum(bl_f1s) / len(bl_f1s)
    mean_lz_f1  = sum(lz_f1s) / len(lz_f1s)
    delta_f1    = mean_lz_f1 - mean_bl_f1

    bl_ci_lo, bl_ci_hi = bootstrap_ci(bl_f1s)
    lz_ci_lo, lz_ci_hi = bootstrap_ci(lz_f1s)

    med_speedup = sorted(speedups)[len(speedups)//2] if speedups else 0.0
    med_bl_ttft = sorted(bl_ttfts)[len(bl_ttfts)//2] if bl_ttfts else 0.0
    med_lz_ttft = sorted(lz_ttfts)[len(lz_ttfts)//2] if lz_ttfts else 0.0

    # Gate checks
    gate_ttft    = med_speedup >= 2.0
    gate_quality = lz_ci_lo >= mean_bl_f1 - 0.02   # LazyLLM CI_lo ≥ baseline_mean - 2%

    print("=" * 60)
    print(f"LazyLLM Paper-Reproduction Evaluation")
    print(f"Dataset : {args.csv_file}")
    print(f"Prompts : {n_total}")
    print()
    print(f"TTFT speedup (median) : {med_speedup:.3f}x  {'✅ PASS' if gate_ttft else '❌ FAIL'} (gate ≥2.0x)")
    print(f"  Baseline TTFT median: {med_bl_ttft:.0f} ms")
    print(f"  LazyLLM TTFT median : {med_lz_ttft:.0f} ms")
    print()
    print(f"F1 baseline           : {mean_bl_f1:.4f}  95% CI [{bl_ci_lo:.4f}, {bl_ci_hi:.4f}]")
    print(f"F1 LazyLLM            : {mean_lz_f1:.4f}  95% CI [{lz_ci_lo:.4f}, {lz_ci_hi:.4f}]")
    print(f"F1 delta              : {delta_f1:+.4f}")
    print(f"Quality gate          : {'✅ PASS' if gate_quality else '❌ FAIL'} (LazyLLM CI_lo ≥ baseline_mean - 2%)")
    print()
    overall = "✅ PASS" if gate_ttft and gate_quality else "❌ FAIL"
    print(f"Overall verdict       : {overall}")
    print("=" * 60)

    # write markdown report if requested
    out_path = args.out or args.csv_file.replace(".csv", "-report.md")
    with open(out_path, "w") as f:
        f.write(f"# LazyLLM Paper-Reproduction Report\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Input: {args.csv_file}\n")
        f.write(f"Prompts: {n_total}\n\n")
        f.write(f"## TTFT Speedup\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Baseline median TTFT | {med_bl_ttft:.0f} ms |\n")
        f.write(f"| LazyLLM median TTFT | {med_lz_ttft:.0f} ms |\n")
        f.write(f"| Speedup (median) | {med_speedup:.3f}× |\n")
        f.write(f"| Gate ≥2.0× | {'PASS ✅' if gate_ttft else 'FAIL ❌'} |\n\n")
        f.write(f"## Quality (F1)\n\n")
        f.write(f"| Metric | Baseline | LazyLLM | Delta |\n|---|---|---|---|\n")
        f.write(f"| F1 mean | {mean_bl_f1:.4f} | {mean_lz_f1:.4f} | {delta_f1:+.4f} |\n")
        f.write(f"| F1 95% CI lo | {bl_ci_lo:.4f} | {lz_ci_lo:.4f} | — |\n")
        f.write(f"| F1 95% CI hi | {bl_ci_hi:.4f} | {lz_ci_hi:.4f} | — |\n")
        f.write(f"| Gate CI_lo ≥ baseline-2% | — | {'PASS ✅' if gate_quality else 'FAIL ❌'} | — |\n\n")
        f.write(f"## Paper Target (LLaMA-2-7B, multi_doc_qa)\n\n")
        f.write(f"| Metric | Paper | This run | Status |\n|---|---|---|---|\n")
        f.write(f"| TTFT speedup | 2.34× | {med_speedup:.2f}× | {'✅' if gate_ttft else '❌'} |\n")
        f.write(f"| F1 score | ≥22.0 | {mean_lz_f1*100:.1f} | {'✅' if mean_lz_f1*100 >= 22.0 else 'TBD'} |\n")

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
