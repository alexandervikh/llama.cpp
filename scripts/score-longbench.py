#!/usr/bin/env python3
"""
Score LongBench evaluation results and produce a markdown summary.

Reads CSV files produced by llama-lazyllm-run (one per subset, named
quality-<model>-<subset>.csv) and computes per-subset metrics with 95%
bootstrap confidence intervals, plus paper-gate checks.

Usage:
  python3 scripts/score-longbench.py \
      --csv-dir results/<run-id>/ \
      --out results/<run-id>/longbench_summary.md \
      [--model <name>]
"""

import argparse
import csv
import os
import re
import random
import string
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Subset configuration
# ---------------------------------------------------------------------------

ALL_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]

MULTI_DOC_QA = {"hotpotqa", "2wikimqa", "musique"}

METRIC_MAP = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "triviaqa": "f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "samsum": "rouge_l",
    "trec": "accuracy",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc": "edit_sim",
    "repobench-p": "edit_sim",
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = re.compile(r"[%s]" % re.escape(string.punctuation))
_WHITESPACE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = _PUNCTUATION.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    # count with multiplicity
    from collections import Counter
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    overlap = sum((pred_c & gold_c).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_f1(pred: str, gold: str) -> float:
    return _token_f1(pred, gold)


def compute_f1_multi(pred: str, golds: list) -> float:
    """Best F1 over multiple reference answers."""
    if not golds:
        return 0.0
    return max(_token_f1(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# Rouge-L (LCS-based, no external deps)
# ---------------------------------------------------------------------------

def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use two-row DP to keep memory linear
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l_multi(pred: str, golds: list) -> float:
    if not golds:
        return 0.0
    return max(compute_rouge_l(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# Exact-match accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(pred: str, golds: list) -> float:
    pred_n = _normalize(pred)
    return 1.0 if any(_normalize(g) == pred_n for g in golds) else 0.0


# ---------------------------------------------------------------------------
# Edit similarity (1 - normalised Levenshtein)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def compute_edit_sim(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    denom = max(len(pred), len(gold))
    if denom == 0:
        return 1.0
    return 1.0 - _edit_distance(pred, gold) / denom


def compute_edit_sim_multi(pred: str, golds: list) -> float:
    if not golds:
        return 0.0
    return max(compute_edit_sim(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(scores: list, n_boot: int = 1000, alpha: float = 0.05):
    """Return (mean, lower, upper) with (1-alpha)*100% bootstrap CI."""
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    mean = sum(scores) / n
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Score dispatcher
# ---------------------------------------------------------------------------

def score_row(pred: str, golds: list, metric: str) -> float:
    if metric == "f1":
        return compute_f1_multi(pred, golds)
    elif metric == "rouge_l":
        return compute_rouge_l_multi(pred, golds)
    elif metric == "accuracy":
        return compute_accuracy(pred, golds)
    elif metric == "edit_sim":
        return compute_edit_sim_multi(pred, golds)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_answers(raw: str) -> list:
    """Parse the pipe-separated answers field from the CSV."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|")]
    return [p for p in parts if p]


def load_csv(path: str) -> list:
    """Load a quality CSV and return list of row dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def score_csv(path: str, subset: str) -> dict:
    """
    Score one CSV file.

    Returns a dict with keys:
      baseline_scores, lazyllm_scores, ttft_speedups, metric, n
    """
    metric = METRIC_MAP.get(subset, "f1")
    rows = load_csv(path)
    baseline_scores, lazyllm_scores, ttft_speedups = [], [], []

    for row in rows:
        golds = parse_answers(row.get("answers", ""))
        baseline_out = row.get("baseline_output", "")
        lazyllm_out = row.get("lazyllm_output", "")

        baseline_scores.append(score_row(baseline_out, golds, metric))
        lazyllm_scores.append(score_row(lazyllm_out, golds, metric))

        try:
            b_ttft = float(row.get("baseline_ttft_ms") or 0)
            l_ttft = float(row.get("lazyllm_ttft_ms") or 0)
            if l_ttft > 0:
                ttft_speedups.append(b_ttft / l_ttft)
        except (ValueError, ZeroDivisionError):
            pass

    return {
        "metric": metric,
        "n": len(rows),
        "baseline_scores": baseline_scores,
        "lazyllm_scores": lazyllm_scores,
        "ttft_speedups": ttft_speedups,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_csvs(csv_dir: str) -> dict:
    """
    Find quality-*-<subset>.csv files in csv_dir.
    Returns {subset_name: filepath}.
    """
    found = {}
    pattern = re.compile(r"^quality-.*?-(.+)\.csv$")
    for fname in os.listdir(csv_dir):
        m = pattern.match(fname)
        if m:
            subset = m.group(1)
            found[subset] = os.path.join(csv_dir, fname)
    return found


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}"


def _delta_str(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}"


def _ci_str(lo: float, hi: float, mean: float) -> str:
    return f"[{_delta_str(lo - mean)}, {_delta_str(hi - mean)}]"


def _speedup_str(speedups: list) -> str:
    if not speedups:
        return "—"
    med = sorted(speedups)[len(speedups) // 2]
    return f"{med:.2f}×"


def render_markdown(
    results: dict,
    model: str,
    csv_dir: str,
) -> str:
    lines = []
    lines.append("# LongBench Evaluation Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {model or 'unknown'}")
    lines.append("")
    lines.append("## Per-Subset Results")
    lines.append("")
    lines.append("| Subset | Metric | Baseline | LazyLLM | Delta | CI | TTFT Speedup |")
    lines.append("|--------|--------|----------|---------|-------|----|--------------|")

    all_baseline, all_lazyllm = [], []
    all_speedups = []
    multi_doc_baseline, multi_doc_lazyllm = [], []

    for subset in ALL_SUBSETS:
        if subset not in results:
            lines.append(f"| {subset} | — | — | — | — | — | — |")
            continue

        r = results[subset]
        b_mean, b_lo, b_hi = bootstrap_ci(r["baseline_scores"])
        l_mean, l_lo, l_hi = bootstrap_ci(r["lazyllm_scores"])
        delta = l_mean - b_mean

        # CI on delta via bootstrap on paired differences
        diffs = [ls - bs for ls, bs in zip(r["lazyllm_scores"], r["baseline_scores"])]
        d_mean, d_lo, d_hi = bootstrap_ci(diffs)

        speedup = _speedup_str(r["ttft_speedups"])

        lines.append(
            f"| {subset} | {r['metric']} "
            f"| {_pct(b_mean)} "
            f"| {_pct(l_mean)} "
            f"| {_delta_str(delta)} "
            f"| [{_delta_str(d_lo)}, {_delta_str(d_hi)}] "
            f"| {speedup} |"
        )

        all_baseline.extend(r["baseline_scores"])
        all_lazyllm.extend(r["lazyllm_scores"])
        all_speedups.extend(r["ttft_speedups"])
        if subset in MULTI_DOC_QA:
            multi_doc_baseline.extend(r["baseline_scores"])
            multi_doc_lazyllm.extend(r["lazyllm_scores"])

    # Aggregate stats
    b_agg = sum(all_baseline) / len(all_baseline) if all_baseline else 0.0
    l_agg = sum(all_lazyllm) / len(all_lazyllm) if all_lazyllm else 0.0
    delta_agg = l_agg - b_agg
    med_speedup = sorted(all_speedups)[len(all_speedups) // 2] if all_speedups else 0.0

    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| | Baseline | LazyLLM | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Mean score (all subsets) "
        f"| {_pct(b_agg)} "
        f"| {_pct(l_agg)} "
        f"| {_delta_str(delta_agg)} |"
    )
    speedup_display = f"{med_speedup:.2f}×" if all_speedups else "—"
    lines.append(
        f"| Median TTFT speedup | — | — | {speedup_display} |"
    )

    # Paper gates
    lines.append("")
    lines.append("## Paper Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Value | Status |")
    lines.append("|------|-----------|-------|--------|")

    # Gate 1: aggregate quality drop ≤ 2.0 pp
    gate1_val = delta_agg * 100
    gate1_pass = gate1_val >= -2.0
    gate1_status = "PASS ✓" if gate1_pass else "FAIL ✗"
    lines.append(
        f"| Aggregate quality drop | ≤ 2.0 pp "
        f"| {gate1_val:+.1f} pp | {gate1_status} |"
    )

    # Gate 2: multi-doc QA F1 drop ≤ 3.0 pp
    if multi_doc_baseline:
        md_b = sum(multi_doc_baseline) / len(multi_doc_baseline)
        md_l = sum(multi_doc_lazyllm) / len(multi_doc_lazyllm)
        md_delta = (md_l - md_b) * 100
    else:
        md_delta = 0.0
    gate2_pass = md_delta >= -3.0
    gate2_status = "PASS ✓" if gate2_pass else "FAIL ✗"
    lines.append(
        f"| Multi-doc QA F1 drop | ≤ 3.0 pp "
        f"| {md_delta:+.1f} pp | {gate2_status} |"
    )

    # Gate 3: median TTFT speedup ≥ 2.0×
    gate3_pass = med_speedup >= 2.0
    gate3_status = "PASS ✓" if gate3_pass else "FAIL ✗"
    lines.append(
        f"| Median TTFT speedup | ≥ 2.0× "
        f"| {med_speedup:.2f}× | {gate3_status} |"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-subset CSV
# ---------------------------------------------------------------------------

def write_per_subset_csv(results: dict, out_path: str):
    fieldnames = [
        "subset", "metric", "n",
        "baseline_mean", "lazyllm_mean", "delta",
        "ci_lo", "ci_hi", "median_ttft_speedup",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset in ALL_SUBSETS:
            if subset not in results:
                continue
            r = results[subset]
            b_mean, _, _ = bootstrap_ci(r["baseline_scores"])
            l_mean, _, _ = bootstrap_ci(r["lazyllm_scores"])
            diffs = [ls - bs for ls, bs in zip(r["lazyllm_scores"], r["baseline_scores"])]
            d_mean, d_lo, d_hi = bootstrap_ci(diffs)
            speedups = sorted(r["ttft_speedups"])
            med_sp = speedups[len(speedups) // 2] if speedups else float("nan")
            writer.writerow({
                "subset": subset,
                "metric": r["metric"],
                "n": r["n"],
                "baseline_mean": f"{b_mean * 100:.2f}",
                "lazyllm_mean": f"{l_mean * 100:.2f}",
                "delta": f"{(l_mean - b_mean) * 100:.2f}",
                "ci_lo": f"{d_lo * 100:.2f}",
                "ci_hi": f"{d_hi * 100:.2f}",
                "median_ttft_speedup": f"{med_sp:.3f}",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Score LongBench CSV results and write a markdown summary."
    )
    p.add_argument("--csv-dir", required=True,
                   help="Directory containing quality-*-<subset>.csv files")
    p.add_argument("--out", required=True,
                   help="Output markdown summary path (e.g. results/longbench_summary.md)")
    p.add_argument("--model", default="",
                   help="Optional model name for the report header")
    args = p.parse_args()

    csv_dir = args.csv_dir
    if not os.path.isdir(csv_dir):
        print(f"ERROR: --csv-dir {csv_dir!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    csv_files = discover_csvs(csv_dir)
    if not csv_files:
        print(f"WARNING: no quality-*-<subset>.csv files found in {csv_dir}", file=sys.stderr)

    results = {}
    for subset, path in sorted(csv_files.items()):
        if subset not in METRIC_MAP and subset not in ALL_SUBSETS:
            print(f"  Skipping unknown subset: {subset}", file=sys.stderr)
            continue
        print(f"  Scoring {subset} ({path}) ...")
        try:
            results[subset] = score_csv(path, subset)
        except Exception as e:
            print(f"  WARNING: failed to score {subset}: {e}", file=sys.stderr)

    missing = [s for s in ALL_SUBSETS if s not in results]
    if missing:
        print(f"\nWARNING: missing subsets (no CSV found): {', '.join(missing)}", file=sys.stderr)

    md = render_markdown(results, model=args.model, csv_dir=csv_dir)

    # Print to stdout
    print("\n" + md)

    # Write markdown
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote summary -> {out_path}")

    # Write per-subset CSV alongside the markdown
    out_dir = os.path.dirname(os.path.abspath(out_path))
    csv_out = os.path.join(out_dir, "per_subset_scores.csv")
    write_per_subset_csv(results, csv_out)
    print(f"Wrote per-subset scores -> {csv_out}")


if __name__ == "__main__":
    main()
