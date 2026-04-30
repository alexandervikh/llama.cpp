#!/usr/bin/env python3
"""
RULER synthetic task generator for LazyLLM evaluation.
Implements the four "quick" tasks from arXiv:2404.06654 (Hsieh et al., NVIDIA 2024):

  niah_single_1   — single needle, single key, single query
  niah_multikey_1 — multiple needles (different keys), single query
  vt              — variable tracking (chains of assignments)
  fwe             — frequent word extraction

Output: JSONL with one object per example:
  {
    "task": "niah_single_1",
    "context": "...",
    "question": "...",
    "answers": ["..."],
    "length": 4096
  }

Usage:
  python3 scripts/ruler-build.py \\
    --tasks niah_single_1 niah_multikey_1 vt fwe \\
    --length 4096 \\
    --n-examples 100 \\
    --out ruler_prompts.jsonl \\
    --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ── task implementations ───────────────────────────────────────────────────────

FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a stitch in time saves nine all that glitters is not gold "
    "better late than never every cloud has a silver lining "
    "fortune favours the bold great minds think alike haste makes waste "
    "actions speak louder than words the early bird catches the worm "
    "practice makes perfect when in rome do as the romans do "
    "the pen is mightier than the sword time flies when you are having fun "
    "two wrongs do not make a right you cannot judge a book by its cover "
    "an apple a day keeps the doctor away beggars cannot be choosers "
    "birds of a feather flock together curiosity killed the cat "
).split()


def filler(rng: random.Random, target_chars: int) -> str:
    """Build a filler passage of approximately target_chars characters."""
    import itertools
    words = []
    total = 0
    for word in itertools.cycle(FILLER_WORDS):
        words.append(word)
        total += len(word) + 1
        if total >= target_chars:
            break
    return " ".join(words)


def insert_at_depth(background: str, needle: str, depth: float) -> str:
    """Insert needle into background at fractional depth."""
    pos = int(len(background) * depth)
    snap = background.find(". ", pos)
    if snap == -1 or abs(snap - pos) > 200:
        snap = pos
    else:
        snap += 2
    return background[:snap] + " " + needle + " " + background[snap:]


# ── niah_single_1 ──────────────────────────────────────────────────────────────

def build_niah_single_1(rng: random.Random, target_chars: int) -> dict:
    number = str(rng.randint(10000, 99999))
    needle = f"The special value is {number}."
    depth  = rng.uniform(0.1, 0.9)
    bg     = filler(rng, target_chars)
    context = insert_at_depth(bg, needle, depth)
    question = "What is the special value mentioned in the text? Output only the number."
    return {
        "task": "niah_single_1",
        "context": context + "\n\nQuestion: " + question,
        "question": question,
        "answers": [number],
        "length": target_chars // 4,
    }


# ── niah_multikey_1 ──────────────────────────────────────────────────────────

def build_niah_multikey_1(rng: random.Random, target_chars: int) -> dict:
    n_keys = 3
    keys   = [f"KEY_{rng.randint(1000, 9999)}" for _ in range(n_keys)]
    values = [str(rng.randint(10000, 99999)) for _ in range(n_keys)]
    query_idx = rng.randrange(n_keys)

    bg = filler(rng, target_chars)

    # Insert needles at distinct depths
    depths = sorted(rng.sample([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_keys))
    for key, val, depth in zip(keys, values, depths):
        needle = f"The value for {key} is {val}."
        bg = insert_at_depth(bg, needle, depth)

    query_key = keys[query_idx]
    answer    = values[query_idx]
    question  = f"What is the value for {query_key}? Output only the number."
    return {
        "task": "niah_multikey_1",
        "context": bg + "\n\nQuestion: " + question,
        "question": question,
        "answers": [answer],
        "length": target_chars // 4,
    }


# ── vt (variable tracking) ────────────────────────────────────────────────────

def build_vt(rng: random.Random, target_chars: int) -> dict:
    """Variable tracking: a chain A=B, B=C, ... ; ask what the final variable resolves to."""
    chain_len = rng.randint(3, 6)
    var_names = [f"var_{chr(65 + i)}" for i in range(chain_len)]
    final_val = str(rng.randint(1000, 9999))

    assignments = []
    for i in range(chain_len - 1):
        assignments.append(f"Let {var_names[i]} = {var_names[i+1]}.")
    assignments.append(f"Let {var_names[-1]} = {final_val}.")

    rng.shuffle(assignments)

    bg = filler(rng, target_chars - sum(len(a) for a in assignments) * 2)
    context = bg
    for assign in assignments:
        depth = rng.uniform(0.05, 0.95)
        context = insert_at_depth(context, assign, depth)

    question = f"What is the value of {var_names[0]}? Output only the number."
    return {
        "task": "vt",
        "context": context + "\n\nQuestion: " + question,
        "question": question,
        "answers": [final_val],
        "length": target_chars // 4,
    }


# ── fwe (frequent word extraction) ───────────────────────────────────────────

COMMON_WORDS = [
    "apple", "banana", "cherry", "dragon", "elephant",
    "falcon", "garden", "harbor", "island", "jungle",
]


def build_fwe(rng: random.Random, target_chars: int) -> dict:
    """
    Frequent word extraction: inject one word many times and several others
    rarely; ask which word appears most often.
    """
    target_word  = rng.choice(COMMON_WORDS)
    other_words  = [w for w in COMMON_WORDS if w != target_word]
    rng.shuffle(other_words)
    decoy_words  = other_words[:4]

    n_target = rng.randint(15, 25)
    n_decoy_each = rng.randint(2, 5)

    # Build passage with injected words
    passage_words = FILLER_WORDS[:]
    total_chars_budget = target_chars
    passage = filler(rng, total_chars_budget)
    words = passage.split()

    # Inject target word
    for _ in range(n_target):
        pos = rng.randrange(len(words))
        words.insert(pos, target_word)

    # Inject decoys
    for dw in decoy_words:
        for _ in range(n_decoy_each):
            pos = rng.randrange(len(words))
            words.insert(pos, dw)

    context = " ".join(words)
    question = "Which single word appears most frequently in the text? Output only the word."
    return {
        "task": "fwe",
        "context": context + "\n\nQuestion: " + question,
        "question": question,
        "answers": [target_word],
        "length": target_chars // 4,
    }


# ── dispatch ──────────────────────────────────────────────────────────────────

BUILDERS = {
    "niah_single_1":  build_niah_single_1,
    "niah_multikey_1": build_niah_multikey_1,
    "vt":             build_vt,
    "fwe":            build_fwe,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build RULER prompts for LazyLLM evaluation")
    p.add_argument("--tasks", nargs="+", default=list(BUILDERS.keys()),
                   choices=list(BUILDERS.keys()),
                   help="Task(s) to generate")
    p.add_argument("--length", type=int, default=4096,
                   help="Approximate context length in tokens")
    p.add_argument("--n-examples", type=int, default=100,
                   help="Examples per task")
    p.add_argument("--out", default="ruler_prompts.jsonl",
                   help="Output JSONL path")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    target_chars = args.length * 4  # rough tokens→chars

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w") as f:
        for task in args.tasks:
            builder = BUILDERS[task]
            for i in range(args.n_examples):
                rng = random.Random(args.seed + i * 100 + hash(task) % 10000)
                record = builder(rng, target_chars)
                record["example_id"] = i
                f.write(json.dumps(record) + "\n")
                total += 1

    print(f"Wrote {total} prompts → {out_path}")
    print(f"  tasks:      {args.tasks}")
    print(f"  length:     {args.length} tokens (target_chars={target_chars})")
    print(f"  n_examples: {args.n_examples} per task")


if __name__ == "__main__":
    main()
