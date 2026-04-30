#!/usr/bin/env python3
"""Section 4: Performance Evaluation — TTFT benchmarks at different context lengths.

Measures Time-To-First-Token (TTFT) for baseline llama-cli vs spec-prefill-run
at 2k, 8k, and 32k prompt context lengths.

Uses llama-cli --perf for internal timings and llama-spec-prefill-run for
spec-prefill TTFT (captured in JSONL output).
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

# Defaults (override via CLI; paper-style runs use Llama 8B + 1B GGUF, see SPEC_PREFILL_TEST_PLAN.md)
DEFAULT_MODEL = os.environ.get("SPEC_PREFILL_MODEL", "")
DEFAULT_LLAMA_CLI = os.environ.get("LLAMA_CLI", "llama-cli")
DEFAULT_SPEC_RUN = os.environ.get("LLAMA_SPEC_PREFILL_RUN", "llama-spec-prefill-run")
DEFAULT_OUTPUT_DIR = os.environ.get("SPEC_PREFILL_TTFT_OUT", "/tmp/spec_perf_ttft")

# Token counts for named context buckets (used with --contexts).
CONTEXT_PRESETS = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}
# Default sweep: all paper-relevant lengths (2k/8k/32k from the test plan, plus 4k/16k for denser curves).
DEFAULT_CONTEXT_LIST = "2k,4k,8k,16k,32k"


def parse_context_list(s: str):
    """Return [(n_tokens, label), ...] e.g. '2k,8k' or '2048:2k,8192:8k'."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            n_str, label = part.split(":", 1)
            n_str, label = n_str.strip(), label.strip()
            out.append((int(n_str), label or n_str))
        elif part.lower() in CONTEXT_PRESETS:
            k = part.lower()
            out.append((CONTEXT_PRESETS[k], k))
        else:
            # Plain integer tokens
            out.append((int(part), f"{int(part)}tok"))
    return out


def generate_prompt(target_ctx_tokens: int, seed=42):
    """Build a long filler prompt intended to stay near ``target_ctx_tokens`` BPE tokens.

    Subword tokenization yields **more** tokens than English words; using ``target_ctx_tokens``
    as a *word* count (the old bug) produced ~3k+ tokens for a ``2k`` label and broke ``-c``.
    Heuristic: ~0.65 words per target token upper-bounds length for diverse filler text.
    """
    rng = random.Random(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "in", "a", "field", "of", "green", "grass", "under", "the", "sun",
        "and", "the", "sky", "is", "blue", "birds", "are", "singing",
        "the", "river", "flows", "through", "the", "valley", "mountains",
        "stand", "tall", "trees", "sway", "in", "the", "wind", "flowers",
        "bloom", "in", "spring", "leaves", "fall", "in", "autumn",
        "winter", "brings", "snow", "summer", "brings", "warmth",
        "the", "ocean", "waves", "crash", "against", "the", "shore",
        "fish", "swim", "in", "the", "deep", "sea", "whales", "sing",
        "dolphins", "play", "in", "the", "surf", "seagulls", "fly",
        "above", "the", "water", "the", "sun", "sets", "horizon",
        "painting", "sky", "orange", "pink", "and", "purple", "stars",
        "appear", "night", "moon", "rises", "silver", "light",
    ]
    n_words = max(64, int(target_ctx_tokens * 0.65))
    chosen = [rng.choice(words) for _ in range(n_words)]
    return " ".join(chosen)


def run_llama_cli_ttft(
    llama_cli,
    model_path,
    prompt,
    ctx_size,
    label,
    n_gpu_layers: int,
    tensor_split: str | None = None,
    split_mode: str | None = None,
    main_gpu: int | None = None,
):
    # Avoid ARG_MAX overflow on long prompts (32k words); llama-cli logs perf on stdout.
    use_file = len(prompt) > 65536
    tmp_path = None
    if use_file:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write(prompt)
        tmp.close()
        tmp_path = tmp.name
        inp_args = ["-f", tmp_path]
    else:
        inp_args = ["-p", prompt]

    cmd = [
        llama_cli,
        "-m", model_path,
        "-c", str(ctx_size),
        "-n", "1",
        "--perf",
        *inp_args,
        "--threads", "4",
        "-no-cnv",
        "-st",
    ]
    if n_gpu_layers and n_gpu_layers > 0:
        cmd += ["-ngl", str(n_gpu_layers)]
        if main_gpu is not None:
            cmd += ["-mg", str(main_gpu)]
        else:
            cmd += ["-mg", "0"]
    if tensor_split:
        cmd += ["-ts", tensor_split]
    if split_mode:
        cmd += ["-sm", split_mode]
    try:
        to = max(300, min(7200, ctx_size // 2))
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=to,
        )
    except subprocess.TimeoutExpired:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return {"label": label, "status": "timeout", "error": "timeout (scaled with ctx)"}

    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    blob = res.stdout or ""
    result = {"label": label, "stderr": blob[-8000:]}

    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", blob)
    if m:
        result["prompt_eval_ms"] = float(m.group(1))
    else:
        m = re.search(r"prompt.*eval time\s*=\s*([\d.]+)\s*ms", blob)
        if m:
            result["prompt_eval_ms"] = float(m.group(1))

    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms", blob)
    if m:
        result["eval_ms"] = float(m.group(1))

    # Newer llama-cli batch summary: "[ Prompt: 880.4 t/s | Generation: ... ]" (no "prompt eval time = ... ms" line).
    if "prompt_eval_ms" not in result:
        m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", blob)
        if m:
            tps = float(m.group(1))
            est_tokens = max(1, int(len(prompt.split()) * 1.28))
            result["prompt_eval_ms"] = round(1000.0 * est_tokens / tps, 2)
            result["prompt_tps_inferred"] = tps
            result["prompt_est_tokens"] = est_tokens

    # Prefer parsed perf lines over exit code (llama-cli may return non-zero while printing timings).
    result["status"] = "ok" if "prompt_eval_ms" in result else ("error" if res.returncode != 0 else "no_perf_line")

    return result


def run_spec_prefill_ttft(
    spec_run,
    base_model,
    spec_model,
    output_dir,
    prompt,
    ctx_size,
    ctx_label: str,
    kr,
    lah,
    pool,
    chunk,
    n_gpu_layers: int,
    tensor_split: str | None = None,
    split_mode: str | None = None,
    main_gpu: int | None = None,
    spec_n_batch_cap: int = 4096,
):
    safe = ctx_label.replace("/", "_").replace(" ", "_")
    out_path = f"{output_dir}/ttft_{safe}_kr{kr}.jsonl"
    prompt_path = f"{output_dir}/prompt_{safe}.jsonl"

    with open(prompt_path, 'w') as f:
        f.write(json.dumps({"id": 0, "prompt": prompt}) + "\n")

    cmd = [
        spec_run,
        "--model", base_model,
        "--spec-model", spec_model,
        "--prompt-file", prompt_path,
        "--out", out_path,
        "--keep-ratio", str(kr),
        "--lookahead", str(lah),
        "--pool", str(pool),
        "--chunk-size", str(chunk),
        "--n-ctx", str(max(ctx_size + 512, 8192)),
        # n_batch must be >= largest decode batch (full prompt); capping below ctx caused GGML_ASSERT in llama_decode.
        "--n-batch", str(min(max(512, ctx_size + 512), spec_n_batch_cap)),
    ]
    if n_gpu_layers and n_gpu_layers > 0:
        cmd += ["-ngl", str(n_gpu_layers)]
    if tensor_split:
        cmd += ["-ts", tensor_split]
    if split_mode:
        cmd += ["-sm", split_mode]
    if main_gpu is not None:
        cmd += ["-mg", str(main_gpu)]
    try:
        to = max(300, min(7200, ctx_size // 2))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=to)
    except subprocess.TimeoutExpired:
        return {"label": f"spec-{ctx_label}-kr{kr}", "status": "timeout"}

    if res.returncode != 0:
        return {"label": f"spec-{ctx_label}-kr{kr}", "status": "error", "stderr": res.stderr[:500]}

    result = {"label": f"spec-{ctx_label}-kr{kr}", "status": "ok"}
    with open(out_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result["ttft_ms"] = float(obj.get("ttft_ms", 0))
                result["n_kept"] = int(obj.get("n_kept", 0))
                result["n_total"] = int(obj.get("n_total", 0))
                result["output_len"] = len(obj.get("output", "").split())
            except json.JSONDecodeError:
                continue

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="TTFT: baseline llama-cli vs llama-spec-prefill-run (SPEC_PREFILL_TEST_PLAN §4)"
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL or None,
        help="GGUF path for baseline llama-cli and (unless --spec-model) spec-prefill base+spec",
    )
    p.add_argument(
        "--spec-model",
        default=None,
        help="Draft model GGUF for spec-prefill; defaults to --model (self-spec)",
    )
    p.add_argument("--llama-cli", default=DEFAULT_LLAMA_CLI, help="Path to llama-cli")
    p.add_argument(
        "--spec-run",
        default=DEFAULT_SPEC_RUN,
        help="Path to llama-spec-prefill-run",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON prompts, JSONL outputs, and ttft_results.json",
    )
    p.add_argument(
        "--min-speedup",
        type=float,
        default=1.5,
        help="Report pass if spec TTFT <= baseline/min-speedup at 8k and kr=0.25 (plan §4 bar)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when §4 gate fails; default exits 0 so runners collect JSON anyway.",
    )
    p.add_argument(
        "--gpu-layers",
        "-ngl",
        type=int,
        default=int(os.environ.get("SPEC_PREFILL_GPU_LAYERS", "99")),
        help="Offload N layers to GPU for llama-cli baseline and llama-spec-prefill-run (0=CPU). "
        "Default 99 or SPEC_PREFILL_GPU_LAYERS env — required for paper-like TTFT on CUDA builds.",
    )
    p.add_argument(
        "--skip-32k",
        action="store_true",
        help="Omit 32k-token prompts (faster; avoids long baseline loads).",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="Paper-style keep ratios: 1.0, 0.10 (~10%% tokens, Fig. 3 caption), 0.25; lah=8, pool=13, chunk=32",
    )
    p.add_argument(
        "--spec-kr",
        nargs="+",
        type=float,
        default=None,
        help="Override spec-prefill keep ratios (default: 1.0 0.5 0.25, or with --paper: 1.0 0.10 0.25)",
    )
    p.add_argument(
        "--only-2k",
        action="store_true",
        help="Only run the 2k context bucket (faster CPU smoke).",
    )
    p.add_argument(
        "--contexts",
        default=os.environ.get("SPEC_PREFILL_CONTEXTS", DEFAULT_CONTEXT_LIST),
        help=f"Comma-separated context buckets ({', '.join(CONTEXT_PRESETS)}) or explicit N:label pairs. Default: {DEFAULT_CONTEXT_LIST}",
    )
    p.add_argument(
        "--tensor-split",
        "-ts",
        default=os.environ.get("LLAMA_ARG_TENSOR_SPLIT") or None,
        help="Multi-GPU tensor split for llama-cli and llama-spec-prefill-run, e.g. 0.25,0.25,0.25,0.25 (4 GPUs).",
    )
    p.add_argument(
        "--split-mode",
        "-sm",
        choices=("none", "layer", "row"),
        default=os.environ.get("LLAMA_ARG_SPLIT_MODE") or None,
        help="GPU split mode (default: omit; use 'layer' with --tensor-split for multi-GPU).",
    )
    p.add_argument(
        "--main-gpu",
        "-mg",
        type=int,
        default=None,
        help="Main GPU index (passed to -mg when using GPU offload).",
    )
    p.add_argument(
        "--spec-n-batch-cap",
        type=int,
        default=262144,
        help="Upper bound for spec-prefill-run --n-batch (actual = min(ctx_tokens+512, this)). "
        "Must stay >= prompt length; default 262144 avoids incorrect low caps.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.model:
        print(
            "Error: set --model <path/to/model.gguf> (or env SPEC_PREFILL_MODEL).",
            file=sys.stderr,
        )
        return 2

    model_path = args.model
    spec_model_path = args.spec_model or model_path
    output_dir = args.output_dir
    llama_cli = args.llama_cli
    spec_run = args.spec_run

    os.makedirs(output_dir, exist_ok=True)

    try:
        contexts = parse_context_list(args.contexts)
    except ValueError as e:
        print(f"Error: bad --contexts: {e}", file=sys.stderr)
        return 2
    if not contexts:
        print("Error: --contexts produced no entries.", file=sys.stderr)
        return 2
    if args.skip_32k:
        contexts = [c for c in contexts if c[1] != "32k"]
    if args.only_2k:
        contexts = [c for c in contexts if c[1] == "2k"]

    ts = args.tensor_split
    sm = args.split_mode
    if ts and sm is None:
        # Match llama-spec-prefill-run: tensor split without an explicit -sm uses layer split.
        sm = "layer"
    mg = args.main_gpu

    if args.spec_kr:
        spec_params = [{"kr": kr, "lah": 8, "pool": 13, "chunk": 32} for kr in args.spec_kr]
    elif args.paper:
        spec_params = [
            {"kr": 1.0, "lah": 8, "pool": 13, "chunk": 32},
            {"kr": 0.10, "lah": 8, "pool": 13, "chunk": 32},
            {"kr": 0.25, "lah": 8, "pool": 13, "chunk": 32},
        ]
    else:
        spec_params = [
            {"kr": 1.0, "lah": 8, "pool": 13, "chunk": 32},
            {"kr": 0.5, "lah": 8, "pool": 13, "chunk": 32},
            {"kr": 0.25, "lah": 8, "pool": 13, "chunk": 32},
        ]

    results = {"ttft": [], "overhead": []}

    print("=" * 70)
    print("Section 4: Performance Evaluation — TTFT Benchmarks")
    print("=" * 70)

    for ctx_size, ctx_label in contexts:
        print(f"\n--- Context: {ctx_label} ({ctx_size} tokens) ---")

        prompt = generate_prompt(ctx_size)

        print(f"  Baseline llama-cli TTFT ({ctx_label})...", end=" ")
        base = run_llama_cli_ttft(
            llama_cli,
            model_path,
            prompt,
            ctx_size,
            f"baseline-{ctx_label}",
            args.gpu_layers,
            tensor_split=ts,
            split_mode=sm,
            main_gpu=mg,
        )
        if base["status"] == "ok" and "prompt_eval_ms" in base:
            print(f"{base['prompt_eval_ms']:.1f} ms")
            base["prompt_eval_ms"] = round(base["prompt_eval_ms"], 2)
            results["ttft"].append({"context": ctx_label, "baseline_ms": base["prompt_eval_ms"]})
        else:
            print(f"FAILED ({base['status']})")
            results["ttft"].append({"context": ctx_label, "baseline_ms": None, "error": base.get("status")})

        for sp in spec_params:
            spec = run_spec_prefill_ttft(
                spec_run,
                model_path,
                spec_model_path,
                output_dir,
                prompt,
                ctx_size,
                ctx_label,
                sp["kr"],
                sp["lah"],
                sp["pool"],
                sp["chunk"],
                args.gpu_layers,
                tensor_split=ts,
                split_mode=sm,
                main_gpu=mg,
                spec_n_batch_cap=args.spec_n_batch_cap,
            )
            label = f"spec-{ctx_label}-kr{sp['kr']}"
            if spec["status"] == "ok" and "ttft_ms" in spec:
                print(f"  {label}: {spec['ttft_ms']:.1f} ms (kept={spec.get('n_kept', '?')}/{spec.get('n_total', '?')}, output={spec.get('output_len', '?')})")
                results["ttft"].append({
                    "context": ctx_label,
                    "keep_ratio": sp["kr"],
                    "ttft_ms": round(spec["ttft_ms"], 2),
                    "n_kept": spec.get("n_kept"),
                    "n_total": spec.get("n_total"),
                })
                if "prompt_eval_ms" in base:
                    overhead_pct = ((spec["ttft_ms"] - base["prompt_eval_ms"]) / base["prompt_eval_ms"] * 100) if base["prompt_eval_ms"] > 0 else 0
                    results["overhead"].append({
                        "context": ctx_label,
                        "keep_ratio": sp["kr"],
                        "baseline_ms": base["prompt_eval_ms"],
                        "spec_ttft_ms": spec["ttft_ms"],
                        "overhead_pct": round(overhead_pct, 2),
                    })
            else:
                print(f"  {label}: FAILED ({spec['status']})")
                results["ttft"].append({"context": ctx_label, "keep_ratio": sp["kr"], "status": spec["status"]})

    out_path = os.path.join(output_dir, "ttft_results.json")

    # Speedups (baseline_ms / spec_ttft_ms) where lower spec TTFT is better
    speedups = []
    for o in results["overhead"]:
        b, s = o.get("baseline_ms"), o.get("spec_ttft_ms")
        if b and s and s > 0:
            speedups.append(
                {
                    "context": o["context"],
                    "keep_ratio": o["keep_ratio"],
                    "speedup": round(b / s, 3),
                }
            )
    results["speedups"] = speedups

    gate = {"plan_section": "§4", "min_speedup_at_8k_kr025": args.min_speedup, "pass": None}
    for sp in speedups:
        if sp["context"] == "8k" and abs(sp["keep_ratio"] - 0.25) < 1e-6:
            gate["pass"] = sp["speedup"] >= args.min_speedup
            gate["observed_speedup"] = sp["speedup"]
            break
    if gate["pass"] is None:
        gate["pass"] = False
        gate["note"] = "no 8k kr=0.25 speedup computed (missing baseline or spec TTFT)"
    results["acceptance_gate"] = gate

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 70)
    print("Overhead Summary (spec-prefill TTFT vs baseline TTFT)")
    print("=" * 70)
    for o in results["overhead"]:
        ctx = o["context"]
        kr = o["keep_ratio"]
        overhead = o["overhead_pct"]
        print(f"  {ctx} | kr={kr:.2f} | baseline={o['baseline_ms']:.0f}ms | spec={o['spec_ttft_ms']:.0f}ms | overhead={overhead:+.1f}%")

    print("\n" + "=" * 70)
    print("Speedups (baseline / spec TTFT)")
    print("=" * 70)
    for sp in speedups:
        print(f"  {sp['context']} | kr={sp['keep_ratio']:.2f} | speedup={sp['speedup']:.2f}x")

    print("\n§4 acceptance (GPU, canonical pair): TTFT speedup ≥ {:.1f}x at 8k ctx, kr=0.25".format(args.min_speedup))
    if gate.get("observed_speedup") is not None:
        print(
            f"  Observed: {gate['observed_speedup']:.2f}x -> {'PASS' if gate['pass'] else 'FAIL'}"
        )
    else:
        print(f"  {gate.get('note', 'FAIL')}")

    print(f"\nSummary saved to: {out_path}")
    if args.strict and not gate.get("pass"):
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
