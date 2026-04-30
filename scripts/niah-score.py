#!/usr/bin/env python3
"""
Score Needle-in-a-Haystack (NIAH) results for LazyLLM evaluation.

Reads a CSV produced by llama-lazyllm-run (--out-csv) cross-referenced with
the JSONL prompt file produced by niah-build.py, and computes per-cell
pass rates for baseline and LazyLLM.

A prediction "passes" if the needle number appears verbatim in the
model's generated output.

Usage:
  python3 scripts/niah-score.py \\
    --prompts niah_prompts.jsonl \\
    --csv niah_results.csv \\
    --out niah-report.md

Output: markdown table and per-cell pass/fail heatmap description.
Exits with code 1 if any acceptance gate fails.

Acceptance gates (from LAZYLLM_TEST_PLAN.md):
  - Baseline pass rate ≥ 95% across all cells (sanity)
  - LazyLLM pass rate ≥ 90% at kr=0.5
  - LazyLLM pass rate ≥ 80% at kr=0.3
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── gates ─────────────────────────────────────────────────────────────────────

GATE_BASELINE_PASS_RATE  = 0.95
GATE_LAZYLLM_KR05        = 0.90
GATE_LAZYLLM_KR03        = 0.80


# ── scoring ────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, answer: str) -> bool:
    """True if the answer string appears verbatim in the prediction."""
    return answer.strip() in prediction


def parse_args():
    p = argparse.ArgumentParser(description="Score NIAH results for LazyLLM")
    p.add_argument("--prompts", required=True, help="JSONL prompt file from niah-build.py")
    p.add_argument("--csv", required=True, help="CSV results file from llama-lazyllm-run")
    p.add_argument("--out", default="niah-report.md", help="Output markdown report path")
    p.add_argument("--kr", type=float, default=0.5,
                   help="Keep ratio used in this run (for gate selection)")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any gate fails (default: report only)")
    return p.parse_args()


def load_prompts(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_csv_results(path: str) -> list[dict]:
    """
    Expected CSV columns: prompt_idx, baseline_output, lazyllm_output
    (as produced by llama-lazyllm-run --out-csv with generation mode)
    Falls back gracefully if columns differ.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def score_run(prompts: list[dict], results: list[dict], kr: float) -> dict:
    """
    Returns a nested dict:
      {(length, depth): {"baseline": [bool, ...], "lazyllm": [bool, ...]}}
    """
    cells: dict = defaultdict(lambda: {"baseline": [], "lazyllm": []})

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        answer = prompt["answers"][0] if prompt["answers"] else ""
        length = prompt["length"]
        depth  = prompt["depth"]

        bl_out = result.get("baseline_output", result.get("baseline", ""))
        lz_out = result.get("lazyllm_output",  result.get("lazyllm",  ""))

        cells[(length, depth)]["baseline"].append(exact_match(bl_out, answer))
        cells[(length, depth)]["lazyllm"].append(exact_match(lz_out, answer))

    return dict(cells)


def rate(bools: list[bool]) -> float:
    return sum(bools) / len(bools) if bools else 0.0


def render_report(cells: dict, kr: float, out_path: str) -> bool:
    """Write markdown report. Returns True if all gates pass."""
    all_lengths = sorted({k[0] for k in cells})
    all_depths  = sorted({k[1] for k in cells})

    # Aggregate pass rates
    all_bl_passes  = []
    all_lz_passes  = []
    for v in cells.values():
        all_bl_passes.extend(v["baseline"])
        all_lz_passes.extend(v["lazyllm"])

    overall_bl = rate(all_bl_passes)
    overall_lz = rate(all_lz_passes)

    gate_bl     = overall_bl >= GATE_BASELINE_PASS_RATE
    lz_threshold = GATE_LAZYLLM_KR05 if kr >= 0.45 else GATE_LAZYLLM_KR03
    gate_lz     = overall_lz >= lz_threshold

    gates_pass = gate_bl and gate_lz

    lines = [
        "# NIAH Evaluation Report",
        "",
        f"Keep ratio: {kr:.2f}  ",
        f"Cells evaluated: {len(cells)}  ",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | LazyLLM | Gate |",
        "|--------|----------|---------|------|",
        f"| Pass rate | {overall_bl:.1%} | {overall_lz:.1%} | "
        f"BL≥{GATE_BASELINE_PASS_RATE:.0%} {'✅' if gate_bl else '❌'}  "
        f"LZ≥{lz_threshold:.0%} {'✅' if gate_lz else '❌'} |",
        "",
    ]

    # Per-length summary
    lines += ["## By Context Length", "", "| Length | BL pass rate | LZ pass rate |",
              "|--------|-------------|-------------|"]
    for length in all_lengths:
        bl_vals = []
        lz_vals = []
        for depth in all_depths:
            k = (length, depth)
            if k in cells:
                bl_vals.extend(cells[k]["baseline"])
                lz_vals.extend(cells[k]["lazyllm"])
        lines.append(f"| {length} | {rate(bl_vals):.1%} | {rate(lz_vals):.1%} |")
    lines.append("")

    # Per-depth summary
    lines += ["## By Depth", "", "| Depth | BL pass rate | LZ pass rate |",
              "|-------|-------------|-------------|"]
    for depth in all_depths:
        bl_vals = []
        lz_vals = []
        for length in all_lengths:
            k = (length, depth)
            if k in cells:
                bl_vals.extend(cells[k]["baseline"])
                lz_vals.extend(cells[k]["lazyllm"])
        lines.append(f"| {depth:.2f} | {rate(bl_vals):.1%} | {rate(lz_vals):.1%} |")
    lines.append("")

    # Full heatmap table — baseline
    lines += ["## Heatmap — Baseline pass rate", "", "| depth \\ length | "
              + " | ".join(str(l) for l in all_lengths) + " |",
              "|" + "---|" * (len(all_lengths) + 1)]
    for depth in all_depths:
        row = [f"{depth:.2f}"]
        for length in all_lengths:
            k = (length, depth)
            v = rate(cells[k]["baseline"]) if k in cells else float("nan")
            row.append(f"{v:.0%}" if v == v else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Full heatmap table — LazyLLM
    lines += ["## Heatmap — LazyLLM pass rate", "", "| depth \\ length | "
              + " | ".join(str(l) for l in all_lengths) + " |",
              "|" + "---|" * (len(all_lengths) + 1)]
    for depth in all_depths:
        row = [f"{depth:.2f}"]
        for length in all_lengths:
            k = (length, depth)
            v = rate(cells[k]["lazyllm"]) if k in cells else float("nan")
            row.append(f"{v:.0%}" if v == v else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"**All gates pass**: {'YES ✅' if gates_pass else 'NO ❌'}")

    report = "\n".join(lines)
    Path(out_path).write_text(report)
    print(report)
    return gates_pass


def main():
    args = parse_args()

    prompts = load_prompts(args.prompts)
    results = load_csv_results(args.csv)

    if len(prompts) != len(results):
        print(f"WARNING: {len(prompts)} prompts but {len(results)} result rows; "
              f"scoring first {min(len(prompts), len(results))} pairs.", file=sys.stderr)
        n = min(len(prompts), len(results))
        prompts = prompts[:n]
        results = results[:n]

    cells = score_run(prompts, results, args.kr)
    ok = render_report(cells, args.kr, args.out)
    print(f"\nReport saved to: {args.out}")

    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
