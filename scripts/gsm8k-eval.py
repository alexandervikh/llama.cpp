#!/usr/bin/env python3
"""
GSM8K evaluation for LazyLLM accuracy verification.
arXiv:2110.14168 (Cobbe et al., OpenAI)

Measures whether LazyLLM preserves math reasoning accuracy relative to
baseline on grade-school math word problems.

GSM8K is a SHORT-CONTEXT benchmark (~1.5k tokens per 8-shot prompt).
LazyLLM should have near-zero effect here because pruning keeps most
tokens on short inputs — any accuracy gap signals a correctness bug.

Usage:
  python3 scripts/gsm8k-eval.py \\
    --binary ./build/bin/llama-lazyllm-run \\
    --model models/llama-3.1-8b-instruct-q8_0.gguf \\
    --n-examples 250 \\
    --pruning-layers 8 16 24 \\
    --keep-ratios 0.5 0.5 0.5 \\
    --out gsm8k-report.md

Requires: pip install datasets
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# ── 8-shot examples (fixed from the GSM8K paper appendix) ─────────────────────

FEW_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he gave some to Denny. Now he has 12. So he gave 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8."""


def build_prompt(question: str) -> str:
    return FEW_SHOT_EXAMPLES + f"\n\nQ: {question}\nA:"


# ── answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output."""
    # Look for "The answer is N" pattern
    m = re.search(r"[Tt]he answer is\s+([\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fall back to last number in text
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    # Normalize: strip commas, compare as float
    try:
        return abs(float(pred.replace(",", "")) - float(gold.replace(",", ""))) < 1e-3
    except ValueError:
        return pred.strip() == gold.strip()


# ── dataset loading ───────────────────────────────────────────────────────────

def load_gsm8k(n: int, seed: int = 42) -> list[dict]:
    """Load n examples from the GSM8K test split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    import random
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = [ds[i] for i in indices[:n]]
    return [{"question": ex["question"], "answer": ex["answer"].split("####")[-1].strip()}
            for ex in selected]


# ── binary runner ─────────────────────────────────────────────────────────────

def run_prompt(binary: str, model: str, prompt: str,
               pruning_layers: list[int], keep_ratios: list[float],
               n_ctx: int, max_new_tokens: int,
               mode: str = "baseline") -> str:
    """Run llama-lazyllm-run for a single prompt, return generated text."""
    if mode == "baseline":
        kr = [1.0] * len(pruning_layers)
    else:
        kr = keep_ratios

    cmd = [
        binary,
        "--model", model,
        "--prompt", prompt,
        "--pruning-layers", *[str(l) for l in pruning_layers],
        "--keep-ratios", *[str(r) for r in kr],
        "--n-ctx", str(n_ctx),
        "--max-new-tokens", str(max_new_tokens),
        "--no-warmup",
        "--no-bench",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        # Extract generated text after "Generated:" marker
        for line in output.splitlines():
            if line.startswith("Generated:"):
                return line[len("Generated:"):].strip()
        # Fallback: return last non-empty line
        lines = [l for l in output.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return ""


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(args) -> tuple[float, float]:
    """Returns (baseline_acc, lazyllm_acc)."""
    print(f"Loading GSM8K ({args.n_examples} examples)...")
    examples = load_gsm8k(args.n_examples, seed=args.seed)

    bl_correct = 0
    lz_correct = 0
    n = len(examples)

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex["question"])
        gold   = ex["answer"]

        print(f"  [{i+1}/{n}] Q: {ex['question'][:50]}...", end=" ", flush=True)

        bl_out = run_prompt(args.binary, args.model, prompt,
                            args.pruning_layers, args.keep_ratios,
                            args.n_ctx, args.max_new_tokens, mode="baseline")
        lz_out = run_prompt(args.binary, args.model, prompt,
                            args.pruning_layers, args.keep_ratios,
                            args.n_ctx, args.max_new_tokens, mode="lazyllm")

        bl_ans = extract_answer(bl_out)
        lz_ans = extract_answer(lz_out)
        bl_ok  = answers_match(bl_ans, gold)
        lz_ok  = answers_match(lz_ans, gold)

        if bl_ok:
            bl_correct += 1
        if lz_ok:
            lz_correct += 1

        print(f"gold={gold} bl={bl_ans}({'✓' if bl_ok else '✗'}) "
              f"lz={lz_ans}({'✓' if lz_ok else '✗'})")

    return bl_correct / n, lz_correct / n


# ── report ────────────────────────────────────────────────────────────────────

def save_report(path: str, args, bl_acc: float, lz_acc: float):
    delta = lz_acc - bl_acc
    gate  = abs(delta) <= 0.03  # ≤3 pp absolute difference
    lines = [
        "# GSM8K Evaluation — LazyLLM Accuracy Verification",
        "",
        f"Date: {datetime.now().isoformat()}",
        f"Model: {Path(args.model).name}",
        f"Pruning: layers={args.pruning_layers} ratios={args.keep_ratios}",
        f"Examples: {args.n_examples}",
        "",
        "## Results",
        "",
        "| Metric | Baseline | LazyLLM | Delta |",
        "|--------|----------|---------|-------|",
        f"| Accuracy | {bl_acc:.1%} | {lz_acc:.1%} | {delta:+.1%} |",
        "",
        f"**Gate (|delta| ≤ 3 pp)**: {'PASS ✅' if gate else 'FAIL ❌'}",
        "",
        "> Short-context task: LazyLLM should have near-zero effect.",
        "> Any gap > 3 pp indicates a correctness bug, not a quality trade-off.",
    ]
    Path(path).write_text("\n".join(lines))
    print("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GSM8K accuracy check for LazyLLM")
    p.add_argument("--binary", default="./build/bin/llama-lazyllm-run")
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--n-examples", type=int, default=250)
    p.add_argument("--pruning-layers", nargs="+", type=int, default=[8, 16, 24])
    p.add_argument("--keep-ratios", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", default="gsm8k-report.md")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    bl_acc, lz_acc = evaluate(args)
    save_report(args.out, args, bl_acc, lz_acc)
    print(f"\nReport saved to: {args.out}")


if __name__ == "__main__":
    main()
