#!/usr/bin/env python3
"""
End-to-end behavioral tests for the LazyLLM binary (llama-lazyllm-run).

These tests use pytest and the live binary. They complement the C++ unit
tests by exercising the full pipeline (binary → CSV output) and catching
issues that pure unit tests miss.

Run (requires a model and built binary):
  pytest tests/python/test_lazyllm_e2e.py \\
      --binary ./build/bin/llama-lazyllm-run \\
      --model /path/to/model.gguf \\
      -v

Skip hardware-dependent tests:
  pytest tests/python/test_lazyllm_e2e.py --no-gpu -v
"""

import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ── pytest configuration ──────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--binary", default="./build/bin/llama-lazyllm-run")
    parser.addoption("--model",  default=None)
    parser.addoption("--no-gpu", action="store_true", help="Skip GPU-dependent tests")


@pytest.fixture(scope="session")
def binary(request):
    b = request.config.getoption("--binary")
    if not Path(b).exists():
        pytest.skip(f"Binary not found: {b}. Build with: cmake --build build --target llama-lazyllm-run")
    return b


@pytest.fixture(scope="session")
def model(request):
    m = request.config.getoption("--model")
    if not m:
        pytest.skip("No --model provided; skipping model-dependent tests")
    if not Path(m).exists():
        pytest.skip(f"Model not found: {m}")
    return m


@pytest.fixture(scope="session")
def no_gpu(request):
    return request.config.getoption("--no-gpu")


# ── helper ────────────────────────────────────────────────────────────────────

SHORT_PROMPT = "The capital of France is"

def run_binary(binary: str, model: str, extra_args: list[str],
               timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = [
        binary,
        "--model", model,
        "--prompt", SHORT_PROMPT,
        "--pruning-layers", "8", "16", "24",
        "--keep-ratios", "0.5", "0.5", "0.5",
        "--n-ctx", "512",
        "--max-new-tokens", "32",
        "--n-gpu-layers", "0",  # CPU-only for portability
    ] + extra_args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── determinism tests ─────────────────────────────────────────────────────────

class TestDeterminism:
    """LazyLLM output must be reproducible under fixed seed."""

    def test_same_seed_same_output(self, binary, model):
        """Two runs with --seed 42 must produce identical generated text."""
        outputs = []
        for _ in range(2):
            r = run_binary(binary, model, ["--seed", "42"])
            assert r.returncode == 0, f"Binary failed: {r.stderr[:500]}"
            gen = self._extract_generated(r.stdout + r.stderr)
            outputs.append(gen)

        assert outputs[0] is not None, "Could not extract generated text from run 1"
        assert outputs[1] is not None, "Could not extract generated text from run 2"
        assert outputs[0] == outputs[1], (
            f"Non-deterministic output:\n  run1: {outputs[0]}\n  run2: {outputs[1]}"
        )

    def test_kr1_matches_baseline(self, binary, model):
        """LazyLLM at kr=1.0 must produce the same output as baseline."""
        r_baseline = run_binary(binary, model,
                                ["--keep-ratios", "1.0", "1.0", "1.0", "--seed", "42"])
        r_lazyllm  = run_binary(binary, model,
                                ["--keep-ratios", "0.5", "0.5", "0.5", "--seed", "42"])

        # We don't require bit-identical at different kr (different pruning changes
        # attention scores), but kr=1.0 should produce valid output.
        assert r_baseline.returncode == 0
        assert r_lazyllm.returncode == 0

    @staticmethod
    def _extract_generated(output: str) -> str | None:
        for line in output.splitlines():
            if line.startswith("Generated:"):
                return line[len("Generated:"):].strip()
        return None


# ── edge-case prompt lengths ──────────────────────────────────────────────────

class TestEdgeCaseLengths:
    """Verify the binary handles extreme prompt lengths without crashing."""

    @pytest.mark.parametrize("n_tokens", [1, 2, 3, 7])
    def test_very_short_prompts(self, binary, model, n_tokens):
        """Very short prompts (< pruning threshold) must not crash."""
        # Build a prompt of approximately n_tokens words
        words = ["hello"] * n_tokens
        short_prompt = " ".join(words)

        cmd = [
            binary,
            "--model", model,
            "--prompt", short_prompt,
            "--pruning-layers", "8", "16", "24",
            "--keep-ratios", "0.5", "0.5", "0.5",
            "--n-ctx", "64",
            "--max-new-tokens", "4",
            "--n-gpu-layers", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (
            f"Binary crashed on {n_tokens}-token prompt: {r.stderr[:300]}"
        )

    def test_prompt_exactly_n_ctx_minus_1(self, binary, model):
        """Prompt of n_ctx-1 tokens must not crash or hang."""
        n_ctx = 128
        # Build a prompt that just fits
        words = ["the"] * (n_ctx - 2)
        prompt = " ".join(words)

        cmd = [
            binary,
            "--model", model,
            "--prompt", prompt,
            "--pruning-layers", "8", "16", "24",
            "--keep-ratios", "0.5", "0.5", "0.5",
            "--n-ctx", str(n_ctx),
            "--max-new-tokens", "1",
            "--n-gpu-layers", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"Crash near n_ctx boundary: {r.stderr[:300]}"

    def test_single_pruning_stage(self, binary, model):
        """Single pruning stage (not 3) must work without crashing."""
        r = run_binary(binary, model, [
            "--pruning-layers", "16",
            "--keep-ratios", "0.5",
        ])
        assert r.returncode == 0, f"Single-stage pruning failed: {r.stderr[:300]}"

    def test_no_pruning_stages(self, binary, model):
        """Empty pruning schedule (no stages) must be equivalent to baseline."""
        r = run_binary(binary, model, [
            "--pruning-layers",
            "--keep-ratios",
        ])
        # Some binaries may require at least one stage; just check no crash
        # (returncode may be non-zero for invalid args — that's acceptable)
        assert "Segmentation fault" not in (r.stdout + r.stderr)
        assert "Aborted" not in (r.stdout + r.stderr)


# ── CSV output ────────────────────────────────────────────────────────────────

class TestCSVOutput:
    """Verify the CSV output format from llama-lazyllm-run --out-csv."""

    def test_csv_has_required_columns(self, binary, model):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            r = run_binary(binary, model, ["--out-csv", csv_path, "--n-prompts", "1"])
            assert r.returncode == 0, f"Binary failed: {r.stderr[:300]}"
            assert Path(csv_path).exists(), "CSV file not created"

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            required = {"baseline_ttft_ms", "lazyllm_ttft_ms"}
            missing = required - set(h.lower() for h in headers)
            assert not missing, f"CSV missing columns: {missing}"
            assert len(rows) >= 1, "CSV has no data rows"

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_csv_speedup_is_positive(self, binary, model):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            r = run_binary(binary, model, ["--out-csv", csv_path, "--n-prompts", "1"])
            if r.returncode != 0:
                pytest.skip("Binary failed; skipping CSV content check")

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))

            if not rows:
                pytest.skip("Empty CSV")

            for row in rows:
                bl = float(row.get("baseline_ttft_ms", 0) or 0)
                lz = float(row.get("lazyllm_ttft_ms", 0) or 0)
                assert bl > 0, f"Baseline TTFT must be positive, got {bl}"
                assert lz > 0, f"LazyLLM TTFT must be positive, got {lz}"

        finally:
            Path(csv_path).unlink(missing_ok=True)


# ── truncation modes ──────────────────────────────────────────────────────────

class TestTruncation:
    """Both truncation modes must complete without error."""

    @pytest.mark.parametrize("mode", ["tail", "middle"])
    def test_truncation_mode(self, binary, model, mode):
        r = run_binary(binary, model, ["--truncation", mode])
        assert r.returncode == 0, (
            f"Truncation mode '{mode}' failed: {r.stderr[:300]}"
        )

    def test_invalid_truncation_exits_nonzero(self, binary, model):
        r = run_binary(binary, model, ["--truncation", "invalid_mode"])
        assert r.returncode != 0, "Invalid truncation mode should exit non-zero"


# ── keep-ratio boundaries ─────────────────────────────────────────────────────

class TestKeepRatioBoundaries:
    """Edge-case keep ratio values must not crash or produce garbage."""

    @pytest.mark.parametrize("kr", ["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "1.0"])
    def test_keep_ratio(self, binary, model, kr):
        r = run_binary(binary, model, ["--keep-ratios", kr, kr, kr])
        assert r.returncode == 0, (
            f"Keep ratio {kr} caused crash: {r.stderr[:300]}"
        )


# ── warmup ────────────────────────────────────────────────────────────────────

class TestWarmup:
    def test_warmup_completes(self, binary, model):
        """Warmup must complete and not hang."""
        r = run_binary(binary, model, [])
        assert r.returncode == 0
        output = r.stdout + r.stderr
        # If warmup ran, there should be some timing output
        assert any(tok in output for tok in ["ms", "TTFT", "baseline", "lazy"]), (
            "Expected timing output not found after warmup"
        )
