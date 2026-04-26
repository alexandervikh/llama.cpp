#!/usr/bin/env python3
"""
Download LongBench multi_doc_qa subset and convert to the JSONL format
consumed by llama-lazyllm-run --prompts-file.

Output per line:
  {"id":"...", "prompt":"<full prompt text>", "answers":["ans1",...], "length":<n>}

Usage:
  python3 scripts/get-longbench.py --dataset multi_doc_qa --n 50 \
      --out datasets/longbench/multi_doc_qa.jsonl \
      --model-path /path/to/model.gguf   # optional: used for token-count column only

The prompt template matches LongBench's dataset2prompt.json for each task.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# LongBench prompt templates (from https://github.com/THUDM/LongBench)
PROMPT_TEMPLATES = {
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news passages.\n\n"
        "{context}\n\nNow, write a one-page summary of all the news passages above.\nSummary:"
    ),
    "qasper": (
        "You are given a scientific article and a question. "
        "Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\".\n\n"
        "Article: {context}\n\nQuestion: {input}\nAnswer:"
    ),
}

# Sub-datasets that together form LongBench multi_doc_qa
MULTI_DOC_QA_DATASETS = ["hotpotqa", "2wikimqa", "musique"]


def download_dataset(dataset_name: str, cache_dir: str, n: int) -> list:
    """Download a LongBench sub-dataset. Tries HuggingFace Hub parquet first,
    then falls back to direct JSONL download from the LongBench v2 repo."""
    import urllib.request

    # LongBench v1 stores test data as JSONL on GitHub
    url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{dataset_name}_test.jsonl"
    local = os.path.join(cache_dir, f"{dataset_name}_test.jsonl")

    if not os.path.exists(local):
        print(f"  Downloading {url} …")
        try:
            urllib.request.urlretrieve(url, local)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"  Using cached {local}")

    items = []
    with open(local, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if n > 0:
        items = items[:n]
    print(f"  Got {len(items)} examples")
    return items


def format_item(item: dict, dataset_name: str) -> dict:
    """Convert a LongBench item to our flat JSONL format."""
    template = PROMPT_TEMPLATES.get(
        dataset_name,
        "{context}\n\nQuestion: {input}\nAnswer:"
    )
    prompt = template.format(
        context=item.get("context", ""),
        input=item.get("input", item.get("question", "")),
    )

    answers = item.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    # LongBench sometimes uses "answer" (singular)
    if not answers:
        ans = item.get("answer", "")
        if ans:
            answers = [ans]

    return {
        "id": f"{dataset_name}_{item.get('_id', item.get('id', ''))}",
        "dataset": dataset_name,
        "prompt": prompt,
        "answers": answers,
        "length": len(prompt),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="multi_doc_qa",
                   help="hotpotqa | 2wikimqa | musique | multi_doc_qa (= all three combined)")
    p.add_argument("--n", type=int, default=50,
                   help="Max examples per sub-dataset (0=all, default=50)")
    p.add_argument("--out", default="datasets/longbench/multi_doc_qa.jsonl",
                   help="Output JSONL path")
    p.add_argument("--cache-dir", default="datasets/longbench/.cache")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.dataset == "multi_doc_qa":
        datasets = MULTI_DOC_QA_DATASETS
    else:
        datasets = [args.dataset]

    total = 0
    with open(args.out, "w") as f:
        for ds_name in datasets:
            items = download_dataset(ds_name, args.cache_dir, args.n)
            for item in items:
                formatted = format_item(item, ds_name)
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} prompts → {args.out}")
    print(f"\nNow run the C++ benchmark:")
    print(f"  ./build/bin/llama-lazyllm-run \\")
    print(f"    --model <model.gguf> \\")
    print(f"    --prompts-file {args.out} \\")
    print(f"    --pruning-layers 8 16 24 \\")
    print(f"    --keep-ratios 0.7 0.5 0.3 \\")
    print(f"    --n-ctx 4096 --max-tokens 64 --repeat 3 \\")
    print(f"    --out-csv results.csv")
    print(f"\nThen score F1:")
    print(f"  python3 scripts/score-f1.py results.csv")


if __name__ == "__main__":
    main()
