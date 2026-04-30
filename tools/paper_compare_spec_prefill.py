#!/usr/bin/env python3
"""
Paper-aligned validation runner for speculative prefill (arXiv:2502.02789).

Implements the execution flow described in SPEC_PREFILL_TEST_PLAN.md and
SPEC_PREFILL_PAPER_COMPARE.md: unit tests, determinism, parity mock, LongBench
driver + scoring, TTFT bench, and optional vLLM environment probe.

Examples:
  python3 tools/paper_compare_spec_prefill.py --all \\
    --build-dir build --base models/Llama-3.1-8B-Instruct.Q4_K_M.gguf \\
    --draft models/Llama-3.2-1B-Instruct.Q4_K_M.gguf -o out/paper_compare

  python3 tools/paper_compare_spec_prefill.py --phase ctest determinism \\
    --build-dir build --base /path/to/model.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a command; return dict with returncode, stdout, stderr (truncated)."""
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": f"timeout: {e}",
            "status": "timeout",
        }
    except FileNotFoundError as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "status": "missing_binary",
        }
    tail = 4000
    return {
        "cmd": cmd,
        "returncode": r.returncode,
        "stdout": (r.stdout or "")[-tail:],
        "stderr": (r.stderr or "")[-tail:],
        "status": "ok" if r.returncode == 0 else "error",
    }


def phase_vllm(_: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    py = (
        "import importlib.util;"
        "s=importlib.util.find_spec('vllm');"
        "print('yes' if s else 'no')"
    )
    r = run_cmd([sys.executable, "-c", py])
    ok = r.get("returncode") == 0 and "yes" in (r.get("stdout") or "")
    return {
        "phase": "vllm_probe",
        "pass": ok,
        "note": "§1b IoU vs official vLLM stack requires working vLLM+PyTorch env",
        "result": r,
    }


def phase_ctest(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    ctest = shutil.which("ctest")
    if not ctest:
        return {"phase": "ctest", "pass": False, "skipped": True, "reason": "ctest not in PATH"}
    if not ns.build_dir or not ns.build_dir.is_dir():
        return {"phase": "ctest", "pass": False, "skipped": True, "reason": "missing --build-dir"}
    r = run_cmd(
        [ctest, "-R", "spec-prefill", "--output-on-failure"],
        cwd=ns.build_dir,
        timeout=600,
    )
    return {"phase": "ctest", "pass": r["returncode"] == 0, "result": r}


def phase_determinism(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    spec_run = ns.spec_run
    if not spec_run.is_file() and not shutil.which(str(spec_run)):
        return {
            "phase": "determinism",
            "pass": False,
            "skipped": True,
            "reason": "llama-spec-prefill-run not found (set --spec-run or build examples/spec-prefill-run)",
        }
    bin_path = str(spec_run) if spec_run.is_file() else shutil.which(str(spec_run))
    if not ns.base or not Path(ns.base).is_file():
        return {
            "phase": "determinism",
            "pass": False,
            "skipped": True,
            "reason": "missing --base GGUF",
        }
    draft = ns.draft if ns.draft and Path(ns.draft).is_file() else ns.base
    out_dir = Path(ns.output) / "determinism"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_line = json.dumps(
        {
            "id": 0,
            "prompt": "Determinism probe. The capital of France is Paris. " * 40,
        }
    )
    pfile = out_dir / "prompt.jsonl"
    pfile.write_text(prompt_line + "\n", encoding="utf-8")
    outs = []
    for i in range(2):
        op = out_dir / f"run{i}.jsonl"
        cmd = [
            bin_path,
            "--model",
            str(ns.base),
            "--spec-model",
            str(draft),
            "--prompt-file",
            str(pfile),
            "--out",
            str(op),
            "--keep-ratio",
            "0.25",
            "--lookahead",
            "8",
            "--pool",
            "13",
            "--chunk-size",
            "32",
        ]
        if getattr(ns, "gpu_layers", 0) and ns.gpu_layers > 0:
            cmd += ["-ngl", str(ns.gpu_layers)]
        r = run_cmd(cmd, timeout=300)
        outs.append({"run": i, "result": r, "out": str(op)})
    def read_first(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        line = path.read_text(encoding="utf-8").strip().split("\n", 1)[0]
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    o0 = read_first(out_dir / "run0.jsonl")
    o1 = read_first(out_dir / "run1.jsonl")
    match = (
        o0 is not None
        and o1 is not None
        and o0.get("n_kept") == o1.get("n_kept")
        and o0.get("n_total") == o1.get("n_total")
    )
    return {
        "phase": "determinism",
        "pass": match,
        "n_kept_a": o0.get("n_kept") if o0 else None,
        "n_kept_b": o1.get("n_kept") if o1 else None,
        "runs": outs,
        "plan_ref": "SPEC_PREFILL_TEST_PLAN §1a",
    }


def phase_parity_mock(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    script = rr / "tools" / "spec-prefill-parity-mock.sh"
    if not script.is_file():
        return {"phase": "parity_mock", "pass": False, "skipped": True, "reason": "script missing"}
    if not ns.base or not Path(ns.base).is_file():
        return {"phase": "parity_mock", "pass": False, "skipped": True, "reason": "missing --base"}
    spec_run = ns.spec_run
    bin_path = str(spec_run) if spec_run.is_file() else shutil.which(str(spec_run))
    if not bin_path:
        return {"phase": "parity_mock", "pass": False, "skipped": True, "reason": "no --spec-run"}
    r = run_cmd(
        ["bash", str(script), "--cpp-bin", bin_path, "--model", str(ns.base)],
        cwd=rr,
        timeout=120,
    )
    return {"phase": "parity_mock", "pass": r["returncode"] == 0, "result": r}


def phase_longbench(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    driver = ns.spec_run
    bin_path = str(driver) if driver.is_file() else shutil.which(str(driver))
    if not bin_path:
        return {"phase": "longbench", "pass": False, "skipped": True, "reason": "no spec-run binary"}
    if not ns.base or not ns.draft:
        return {
            "phase": "longbench",
            "pass": False,
            "skipped": True,
            "reason": "need both --base and --draft for LongBench driver",
        }
    runner = rr / "eval" / "run_longbench.py"
    if not runner.is_file():
        return {"phase": "longbench", "pass": False, "skipped": True, "reason": "eval/run_longbench.py missing"}
    out_lb = Path(ns.output) / "longbench"
    out_lb.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--driver",
        bin_path,
        "--base",
        str(ns.base),
        "--spec",
        str(ns.draft),
        "--out",
        str(out_lb),
        "--n-per-task",
        str(ns.n_per_task),
        "--keep-ratios",
        "0.1",
        "0.25",
        "1.0",
    ]
    if ns.longbench_synthetic:
        cmd.append("--synthetic")
    r = run_cmd(cmd, cwd=rr, timeout=7200)
    return {
        "phase": "longbench",
        "pass": r["returncode"] == 0,
        "result": r,
        "results_dir": str(out_lb),
    }


def phase_score_longbench(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    scorer = rr / "eval" / "score_longbench.py"
    out_lb = Path(ns.output) / "longbench"
    if not scorer.is_file():
        return {"phase": "score_longbench", "pass": False, "skipped": True, "reason": "score_longbench.py missing"}
    if not out_lb.is_dir():
        return {"phase": "score_longbench", "pass": False, "skipped": True, "reason": "run longbench first"}
    r = run_cmd(
        [sys.executable, str(scorer), "--results-dir", str(out_lb)],
        cwd=rr,
        timeout=120,
    )
    return {"phase": "score_longbench", "pass": r["returncode"] == 0, "result": r}


def phase_ttft(ns: argparse.Namespace, rr: Path) -> Dict[str, Any]:
    bench = rr / "tools" / "spec-prefill-ttft-bench.py"
    if not bench.is_file():
        return {"phase": "ttft", "pass": False, "skipped": True, "reason": "spec-prefill-ttft-bench.py missing"}
    if not ns.base or not Path(ns.base).is_file():
        return {"phase": "ttft", "pass": False, "skipped": True, "reason": "missing --base"}
    draft = ns.draft if ns.draft and Path(ns.draft).is_file() else ns.base
    out_t = Path(ns.output) / "ttft"
    out_t.mkdir(parents=True, exist_ok=True)
    spec_run = ns.spec_run
    bin_path = str(spec_run) if spec_run.is_file() else shutil.which(str(spec_run))
    llama_cli = shutil.which("llama-cli") or "llama-cli"
    cmd = [
        sys.executable,
        str(bench),
        "--model",
        str(ns.base),
        "--spec-model",
        str(draft),
        "--llama-cli",
        llama_cli,
        "--spec-run",
        bin_path or str(spec_run),
        "-o",
        str(out_t),
        "--gpu-layers",
        str(getattr(ns, "gpu_layers", 0) or 0),
    ]
    r = run_cmd(cmd, cwd=rr, timeout=3600)
    gate_pass = None
    json_path = out_t / "ttft_results.json"
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            gate_pass = (data.get("acceptance_gate") or {}).get("pass")
        except json.JSONDecodeError:
            pass
    return {
        "phase": "ttft",
        "pass": r["returncode"] == 0,
        "plan_gate_pass": gate_pass,
        "result": r,
        "ttft_json": str(json_path),
    }


PHASES = {
    "vllm": phase_vllm,
    "ctest": phase_ctest,
    "determinism": phase_determinism,
    "parity": phase_parity_mock,
    "longbench": phase_longbench,
    "score": phase_score_longbench,
    "ttft": phase_ttft,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-aligned spec-prefill validation (see SPEC_PREFILL_PAPER_COMPARE.md)")
    p.add_argument(
        "--phase",
        nargs="+",
        choices=list(PHASES.keys()) + ["all"],
        default=["all"],
        help="Phases to run (default: all)",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Output directory for artifacts + report JSON")
    p.add_argument("--build-dir", type=Path, default=None, help="CMake build dir for ctest")
    p.add_argument("--base", type=Path, help="Base model GGUF (Llama-3.1-8B-Instruct recommended)")
    p.add_argument("--draft", type=Path, help="Draft/speculator GGUF (Llama-3.2-1B-Instruct recommended)")
    p.add_argument(
        "--spec-run",
        type=Path,
        default=None,
        help="Path to llama-spec-prefill-run (default: build/bin or PATH)",
    )
    p.add_argument("--n-per-task", type=int, default=5, help="LongBench prompts per task (lower = faster smoke)")
    p.add_argument(
        "--longbench-synthetic",
        action="store_true",
        help="Use synthetic prompts if HF datasets unavailable",
    )
    p.add_argument(
        "--gpu-layers",
        "-ngl",
        type=int,
        default=int(os.environ.get("SPEC_PREFILL_GPU_LAYERS", "99")),
        help="Passed to llama-spec-prefill-run (-ngl); default 99 for CUDA paper-like runs",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failing phase",
    )
    return p.parse_args()


def resolve_defaults(ns: argparse.Namespace, rr: Path) -> None:
    if ns.output is None:
        ns.output = rr / "build" / "paper_compare"
    ns.output = Path(ns.output)
    ns.output.mkdir(parents=True, exist_ok=True)
    if ns.spec_run is None:
        cand = rr / "build" / "bin" / "llama-spec-prefill-run"
        ns.spec_run = cand if cand.is_file() else Path("llama-spec-prefill-run")


def main() -> int:
    ns = parse_args()
    rr = repo_root()
    resolve_defaults(ns, rr)

    phases: List[str]
    if "all" in ns.phase:
        phases = ["vllm", "ctest", "determinism", "parity", "longbench", "score", "ttft"]
    else:
        phases = list(ns.phase)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(rr),
        "paper": "arXiv:2502.02789 (SpecPrefill)",
        "phases": [],
        "overall_pass": True,
    }

    for name in phases:
        fn = PHASES[name]
        entry = fn(ns, rr)
        report["phases"].append(entry)
        if not entry.get("pass", False) and not entry.get("skipped", False):
            report["overall_pass"] = False
            if ns.fail_fast:
                break

    out_json = ns.output / "paper_compare_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}", file=sys.stderr)

    # Exit 1 if any non-skipped phase failed
    failed = any(
        not p.get("pass", False) and not p.get("skipped", False) for p in report["phases"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
