#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) prompt builder for LazyLLM evaluation.

Inserts a distinctive fact ("The magic number is XXXX.") at various depths
inside a filler corpus (Paul Graham essays, or synthetic repeated phrases
when --synthetic is given), then asks a retrieval question.

Output is a JSONL file with one object per prompt:
  {
    "context": "<full prompt text>",
    "question": "What is the magic number?",
    "answers": ["XXXX"],
    "needle": "XXXX",
    "depth": 0.25,
    "length": 4096,
    "trial": 0
  }

Usage:
  python3 scripts/niah-build.py \\
    --lengths 2048 4096 8192 \\
    --depths 0 0.25 0.5 0.75 1.0 \\
    --trials 5 \\
    --out niah_prompts.jsonl

  # Synthetic mode (no corpus download needed):
  python3 scripts/niah-build.py --synthetic --lengths 4096 --depths 0 0.5 1.0 --trials 3
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# ── filler corpus ─────────────────────────────────────────────────────────────

SYNTHETIC_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A stitch in time saves nine.",
    "All that glitters is not gold.",
    "Better late than never.",
    "Every cloud has a silver lining.",
    "Fortune favours the bold.",
    "Great minds think alike.",
    "Haste makes waste.",
    "Actions speak louder than words.",
    "The early bird catches the worm.",
    "Practice makes perfect.",
    "When in Rome, do as the Romans do.",
    "The pen is mightier than the sword.",
    "Time flies when you are having fun.",
    "Two wrongs do not make a right.",
    "You cannot judge a book by its cover.",
    "An apple a day keeps the doctor away.",
    "Beggars cannot be choosers.",
    "Birds of a feather flock together.",
    "Curiosity killed the cat.",
]


def build_synthetic_corpus(target_chars: int) -> str:
    """Build a filler corpus of approximately target_chars characters."""
    import itertools
    sentences = []
    total = 0
    for sent in itertools.cycle(SYNTHETIC_SENTENCES):
        sentences.append(sent)
        total += len(sent) + 1
        if total >= target_chars:
            break
    return " ".join(sentences)


def load_pg_essays(cache_dir: Path) -> str:
    """Try to load Paul Graham essays from cache, download if absent."""
    cache_file = cache_dir / "pg_essays.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    urls = [
        "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/needlehaystack/PaulGrahamEssays.txt",
    ]
    try:
        import urllib.request
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
                print(f"Downloaded Paul Graham essays → {cache_file}", file=sys.stderr)
                return text
            except Exception:
                continue
    except Exception:
        pass
    return ""


# ── needle generation ─────────────────────────────────────────────────────────

def make_needle(seed: int) -> tuple[str, str]:
    """Return (needle_sentence, answer) for the given seed."""
    rng = random.Random(seed)
    number = str(rng.randint(10000, 99999))
    sentence = f"The magic number is {number}."
    return sentence, number


QUESTION_TEMPLATE = (
    "Based only on the text above, what is the magic number mentioned? "
    "Only output the number, nothing else."
)


# ── prompt assembly ───────────────────────────────────────────────────────────

def build_prompt(corpus: str, needle: str, depth: float, target_chars: int) -> str:
    """
    Truncate corpus to target_chars, insert needle at fractional depth position,
    then append the question.
    """
    truncated = corpus[:target_chars]
    insert_pos = int(len(truncated) * depth)

    # Snap to a sentence boundary near the insert position.
    snap = truncated.find(". ", insert_pos)
    if snap == -1 or snap - insert_pos > 200:
        snap = insert_pos
    else:
        snap += 2  # past the period+space

    text = truncated[:snap] + " " + needle + " " + truncated[snap:]
    return text + "\n\n" + QUESTION_TEMPLATE


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build NIAH prompts for LazyLLM evaluation")
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 4096, 8192],
                   help="Target context lengths in tokens (approximated as chars×4)")
    p.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Needle insertion depths (0=start, 1=end)")
    p.add_argument("--trials", type=int, default=5,
                   help="Number of random seeds (needle numbers) per (depth, length) cell")
    p.add_argument("--out", default="niah_prompts.jsonl",
                   help="Output JSONL file path")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic filler instead of Paul Graham essays (no download)")
    p.add_argument("--cache-dir", default="datasets/niah",
                   help="Directory for cached corpus")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed")
    return p.parse_args()


def main():
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    corpus = ""

    if not args.synthetic:
        corpus = load_pg_essays(cache_dir)
        if not corpus:
            print("Could not download Paul Graham essays; falling back to synthetic.",
                  file=sys.stderr)

    total = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for length in args.lengths:
            # Characters ≈ tokens × 4 (rough but consistent)
            target_chars = length * 4

            if not corpus or args.synthetic:
                filler = build_synthetic_corpus(target_chars * 2)
            else:
                filler = corpus

            for depth in args.depths:
                for trial in range(args.trials):
                    needle_seed = args.seed + trial * 1000 + int(depth * 100) + length
                    needle_text, answer = make_needle(needle_seed)

                    prompt = build_prompt(filler, needle_text, depth, target_chars)

                    record = {
                        "context": prompt,
                        "question": QUESTION_TEMPLATE,
                        "answers": [answer],
                        "needle": answer,
                        "depth": depth,
                        "length": length,
                        "trial": trial,
                        "seed": needle_seed,
                    }
                    f.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} prompts → {out_path}")
    print(f"  lengths: {args.lengths}")
    print(f"  depths:  {args.depths}")
    print(f"  trials:  {args.trials}")


if __name__ == "__main__":
    main()
