#!/usr/bin/env python3
"""
Print the correct max_new_tokens value for a given LongBench subset.

Usage:
  python3 scripts/longbench_max_new.py <subset_name>

Exits with code 0 and prints the integer to stdout.
Unknown subsets print the default value (64).
"""

import sys

MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

DEFAULT = 64


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <subset_name>", file=sys.stderr)
        sys.exit(1)

    subset = sys.argv[1]
    print(MAX_NEW_TOKENS.get(subset, DEFAULT))


if __name__ == "__main__":
    main()
