#!/usr/bin/env python3
"""
Download LongBench subsets and convert to the JSONL format
consumed by llama-lazyllm-run --prompts-file.

Output per line:
  {"id":"...", "dataset":"...", "prompt":"<full prompt text>", "answers":["ans1",...], "length":<n>}

Usage:
  python3 scripts/get-longbench.py --dataset multi_doc_qa --n 50 \
      --out datasets/longbench/multi_doc_qa.jsonl

  python3 scripts/get-longbench.py --all-tasks --n 50 \
      --out datasets/longbench/all.jsonl

Individual subset names are also valid for --dataset:
  python3 scripts/get-longbench.py --dataset hotpotqa --n 50 \
      --out datasets/longbench/hotpotqa.jsonl

The prompt templates match LongBench's dataset2prompt.json from
https://github.com/THUDM/LongBench.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# LongBench prompt templates (from https://github.com/THUDM/LongBench)
PROMPT_TEMPLATES = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. "
        "Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, "
        "only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
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
    "gov_report": (
        "You are given a report by a government agency. "
        "Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Meeting Transcript:\n{context}\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. "
        "Write a one-page summary of all news passages.\n\n"
        "{context}\n\n"
        "Now, write a one-page summary of all the news passages above.\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. "
        "Only give me the answer and do not output any other words. "
        "The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. "
        "Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. "
        "In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number.\nAnswer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 3\", \"Paragraph 7\", etc.\nAnswer:"
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}"
    ),
}

# All 16 subsets
ALL_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]

MULTI_DOC_QA_DATASETS = ["hotpotqa", "2wikimqa", "musique"]


def download_dataset(dataset_name: str, cache_dir: str, n: int) -> list:
    """Download a LongBench sub-dataset from HuggingFace as JSONL."""
    import urllib.request

    url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{dataset_name}_test.jsonl"
    local = os.path.join(cache_dir, f"{dataset_name}_test.jsonl")

    if not os.path.exists(local):
        print(f"  Downloading {url} ...")
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

    if dataset_name == "lcc":
        # lcc has no input field; template only uses {context}
        prompt = template.format(context=item.get("context", ""))
    elif dataset_name == "repobench-p":
        # repobench-p stores the code context in "code" field
        prompt = template.format(
            context=item.get("code", item.get("context", "")),
            input=item.get("input", ""),
        )
    else:
        prompt = template.format(
            context=item.get("context", ""),
            input=item.get("input", item.get("question", "")),
        )

    # Extract answers — LongBench uses several field conventions
    answers = item.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not answers:
        ans = item.get("answer", "")
        if ans:
            answers = [ans] if isinstance(ans, str) else list(ans)
    # trec uses all_classes as the reference labels
    if not answers:
        cls = item.get("all_classes", [])
        if cls:
            answers = [str(c) for c in cls]

    return {
        "id": f"{dataset_name}_{item.get('_id', item.get('id', ''))}",
        "dataset": dataset_name,
        "prompt": prompt,
        "answers": answers,
        "length": len(prompt),
    }


def main():
    p = argparse.ArgumentParser(
        description="Download LongBench subsets and emit JSONL for llama-lazyllm-run."
    )
    p.add_argument(
        "--dataset", default="multi_doc_qa",
        help=(
            "Subset name, or 'multi_doc_qa' for the three multi-doc QA subsets combined. "
            "Valid single subsets: " + ", ".join(ALL_SUBSETS)
        ),
    )
    p.add_argument(
        "--all-tasks", action="store_true",
        help="Download all 16 LongBench subsets (overrides --dataset).",
    )
    p.add_argument("--n", type=int, default=50,
                   help="Max examples per sub-dataset (0=all, default=50)")
    p.add_argument("--out", default="datasets/longbench/multi_doc_qa.jsonl",
                   help="Output JSONL path")
    p.add_argument("--cache-dir", default="datasets/longbench/.cache")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.all_tasks:
        datasets = ALL_SUBSETS
    elif args.dataset == "multi_doc_qa":
        datasets = MULTI_DOC_QA_DATASETS
    else:
        datasets = [args.dataset]

    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ds_name in datasets:
            print(f"\n[{ds_name}]")
            items = download_dataset(ds_name, args.cache_dir, args.n)
            for item in items:
                formatted = format_item(item, ds_name)
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} prompts -> {args.out}")
    print(f"\nNow run the C++ benchmark:")
    print(f"  ./build/bin/llama-lazyllm-run \\")
    print(f"    --model <model.gguf> \\")
    print(f"    --prompts-file {args.out} \\")
    print(f"    --pruning-layers 8 16 24 \\")
    print(f"    --keep-ratios 0.7 0.5 0.3 \\")
    print(f"    --n-ctx 4096 --max-tokens 64 --repeat 3 \\")
    print(f"    --out-csv results.csv")
    print(f"\nThen score with:")
    print(f"  python3 scripts/score-longbench.py --csv-dir results/ --out results/longbench_summary.md")


if __name__ == "__main__":
    main()
