# Spec-Prefill Test Runbook — Step-by-Step Executor Guide

This is the **operational runbook** for `SPEC_PREFILL_TEST_PLAN.md` + `SPEC_PREFILL_TEST_GUIDE.md`. Every step has a copy-pasteable command, expected output, and a checkpoint. Designed so someone who has never seen this repo can execute it end-to-end.

Conventions:
- `$REPO` = `/Users/alexander.vikhorev/src/llama.cpp`
- `$REF` = clone path of reference repo (to be created in step 0).
- `$MODELS` = `$REPO/models/` (already exists).
- Every `bash` block starts with `cd $REPO` or `cd $REF` explicitly.
- ✅ = checkpoint you should verify before moving on.

---

## Step 0 — Prerequisites (do once)

### 0.1 Tooling versions
```bash
python3 --version     # expect >= 3.10
cmake --version       # expect >= 3.22
git --version
```
Install HF + scientific stack in a fresh venv:
```bash
cd $REPO
python3 -m venv .venv-sp
source .venv-sp/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy jsonlines tqdm \
            huggingface_hub datasets transformers sentencepiece \
            torch --index-url https://download.pytorch.org/whl/cpu
```
✅ `python -c "import torch, datasets, transformers"` returns no error.

### 0.2 HuggingFace auth
```bash
huggingface-cli login   # paste token with read access to Llama weights
```
✅ `huggingface-cli whoami` prints your handle.

### 0.3 Download models
```bash
# Draft (1B Llama)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir $MODELS/llama-3.2-1b-instruct

# Base (8B Llama) — used everywhere except parity step where paper uses 70B
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $MODELS/llama-3.1-8b-instruct

# Convert to GGUF F16 for C++ side
python $REPO/convert_hf_to_gguf.py $MODELS/llama-3.2-1b-instruct \
    --outfile $MODELS/llama-3.2-1b-instruct-f16.gguf --outtype f16
python $REPO/convert_hf_to_gguf.py $MODELS/llama-3.1-8b-instruct \
    --outfile $MODELS/llama-3.1-8b-instruct-f16.gguf --outtype f16
```
✅ Both `.gguf` files exist and `ls -la` shows expected sizes (~2.3GB and ~16GB).

### 0.4 Build llama.cpp
```bash
cd $REPO
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu) \
      --target test-spec-prefill test-spec-prefill-bench llama-cli llama-perplexity llama-bench
```
✅ `./build/bin/test-spec-prefill --help` runs without crashing.

### 0.5 Clone reference repo
```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/Jingyu6/speculative_prefill.git
export REF=~/src/speculative_prefill
cd $REF
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r <(cat <<EOF
vllm
torch
numpy
pandas
matplotlib
pyyaml
transformers
datasets
EOF
)
```
Note: vLLM requires CUDA for full function. If you're on Apple Silicon, **skip the reference run on this machine** — use a Linux+CUDA box for steps that require it (1.2, 1.4 comparison input). Everything else runs locally.

✅ `python -c "from speculative_prefill import enable_prefill_spec"` succeeds on the CUDA machine.

---

## Step 1 — Algorithmic Parity

### 1.1 Prepare prompt corpus (both machines)
Create `$REPO/tests/data/parity_prompts.jsonl`:
```bash
mkdir -p $REPO/tests/data
python3 - <<'EOF'
import json, random
from datasets import load_dataset
random.seed(42)
ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
out = []
for i, r in enumerate(ds.select(range(50))):
    out.append({"id": i, "input": r["input"], "context": r["context"][:8000]})
with open("$REPO/tests/data/parity_prompts.jsonl".replace("$REPO","/Users/alexander.vikhorev/src/llama.cpp"), "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
EOF
```
✅ File has 50 lines.

### 1.2 Instrument the reference repo (CUDA machine)

Open `$REF/speculative_prefill/vllm_patch/worker/look_ahead_spec_worker.py`. Find the function that produces lookahead tokens (likely named `_run_speculative_decoding_step` or similar). At its end, add:

```python
import os, json, numpy as np
_dump_dir = os.environ.get("SPEC_PREFILL_DUMP")
if _dump_dir:
    os.makedirs(_dump_dir, exist_ok=True)
    req_id = getattr(seq_group_metadata, "request_id", "unknown")
    with open(f"{_dump_dir}/{req_id}_lookahead.json", "w") as f:
        json.dump({
            "prompt_ids": prompt_token_ids,
            "lookahead_ids": [int(t) for t in lookahead_tokens],
        }, f)
```

Open `$REF/speculative_prefill/vllm_patch/worker/spec_prefill_worker.py`. Find where importance scores are computed (search for `attn`, `score`, or `importance`). Insert dumps at three points — **before pooling**, **after pooling**, and **after token selection**:

```python
if _dump_dir:
    np.save(f"{_dump_dir}/{req_id}_imp_raw.npy", importance_raw.cpu().numpy())
    np.save(f"{_dump_dir}/{req_id}_imp_pooled.npy", importance_pooled.cpu().numpy())
    with open(f"{_dump_dir}/{req_id}_kept.json", "w") as f:
        json.dump({"kept_indices": kept_idx.tolist(),
                   "strategy": self.keep_strategy,
                   "keep_ratio": self.percentage}, f)
```

### 1.3 Run the reference to produce traces
```bash
cd $REF
source .venv/bin/activate
export SPEC_PREFILL_DUMP=$REF/dumps/ref
mkdir -p $SPEC_PREFILL_DUMP
python - <<'EOF'
from speculative_prefill import enable_prefill_spec
enable_prefill_spec(spec_model='meta-llama/Llama-3.2-1B-Instruct',
                    spec_config_path='./configs/config_p1_full_lah8.yaml')
from vllm import LLM, SamplingParams
import json
llm = LLM('meta-llama/Llama-3.1-8B-Instruct',
          gpu_memory_utilization=0.8, enforce_eager=True,
          enable_chunked_prefill=False)
sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
with open('/path/to/parity_prompts.jsonl') as f:
    for line in f:
        r = json.loads(line)
        prompt = r["context"] + "\n\nQuestion: " + r["input"]
        llm.generate([prompt], sp, request_id=str(r["id"]))
EOF
```
✅ `ls $SPEC_PREFILL_DUMP | wc -l` returns 200 (50 prompts × 4 files each).

Copy the dumps back: `scp -r cuda-box:$REF/dumps/ref $REPO/tests/data/ref_dumps`.

### 1.4 Instrument llama.cpp side
Edit `src/llama-spec-prefill.cpp`. At the top add:
```cpp
#include <cstdlib>
#include <fstream>
#include <sstream>
static const char * dump_dir() { return std::getenv("LLAMA_SPEC_PREFILL_DUMP"); }
static void dump_json(const std::string & path, const std::string & body) {
    std::ofstream f(path); f << body;
}
static void dump_floats(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = v.size(); f.write((char*)&n, 4);
    f.write((char*)v.data(), v.size()*sizeof(float));
}
```

In `llama_spec_prefill_generate_lookahead` (line 52), after the lookahead loop, if `dump_dir()` is set, write `{dump_dir}/{req_id}_lookahead.json` with `prompt_ids` and `lookahead_ids`.

In `llama_spec_prefill_compute_importance` (line 241): dump the raw importance vector to `_imp_raw.bin`, then after `llama_spec_prefill_apply_pooling` dump `_imp_pooled.bin`.

In `llama_spec_prefill_filter_tokens` and `_chunked` (lines 342, 402): dump kept indices to `_kept.json`.

The request_id comes from a new optional parameter you'll thread through — simplest: use the prompt index passed by the harness.

Rebuild: `cmake --build build --target test-spec-prefill -j`.

### 1.5 Add a parity driver binary
Create `tests/test-spec-prefill-parity.cpp`. It loads prompts from `parity_prompts.jsonl`, tokenizes via `llama_tokenize`, and for each prompt calls the full pipeline with `LLAMA_SPEC_PREFILL_DUMP=$REPO/tests/data/cpp_dumps`. Wire into `tests/CMakeLists.txt` alongside the existing test targets.

```bash
cd $REPO
export LLAMA_SPEC_PREFILL_DUMP=$REPO/tests/data/cpp_dumps
mkdir -p $LLAMA_SPEC_PREFILL_DUMP
./build/bin/test-spec-prefill-parity \
    --base  $MODELS/llama-3.1-8b-instruct-f16.gguf \
    --spec  $MODELS/llama-3.2-1b-instruct-f16.gguf \
    --prompts tests/data/parity_prompts.jsonl \
    --keep-ratio 0.25 --lookahead 8 --pool 13
```
✅ `ls $LLAMA_SPEC_PREFILL_DUMP | wc -l` ≈ 200.

### 1.6 Compare
Write `scripts/compare_spec_prefill_traces.py`:
```python
import json, numpy as np, sys, os, struct
from pathlib import Path
ref = Path(sys.argv[1]); cpp = Path(sys.argv[2])

def load_bin(p):
    with open(p, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.float32, count=n)

def load_npy(p): return np.load(p)

iou_list, corr_raw, corr_pooled = [], [], []
for i in range(50):
    ref_kept = set(json.load(open(ref/f"{i}_kept.json"))["kept_indices"])
    cpp_kept = set(json.load(open(cpp/f"{i}_kept.json"))["kept_indices"])
    iou_list.append(len(ref_kept & cpp_kept) / len(ref_kept | cpp_kept))

    r_raw = load_npy(ref/f"{i}_imp_raw.npy").flatten()
    c_raw = load_bin(cpp/f"{i}_imp_raw.bin")
    n = min(len(r_raw), len(c_raw))
    corr_raw.append(np.corrcoef(r_raw[:n], c_raw[:n])[0,1])

    r_pool = load_npy(ref/f"{i}_imp_pooled.npy").flatten()
    c_pool = load_bin(cpp/f"{i}_imp_pooled.bin")
    n = min(len(r_pool), len(c_pool))
    corr_pooled.append(np.corrcoef(r_pool[:n], c_pool[:n])[0,1])

print(f"IoU  mean={np.mean(iou_list):.3f} min={min(iou_list):.3f}")
print(f"Raw  corr mean={np.mean(corr_raw):.3f}")
print(f"Pool corr mean={np.mean(corr_pooled):.3f}")
```
Run:
```bash
python scripts/compare_spec_prefill_traces.py \
    $REPO/tests/data/ref_dumps $REPO/tests/data/cpp_dumps | tee parity_report.txt
```
✅ `IoU mean ≥ 0.95`, `Pool corr mean ≥ 0.98`. If not, open `parity_report.txt` + per-prompt dumps and bisect which stage first diverges.

---

## Step 2 — Unit / Component Tests

### 2.1 Extend `tests/test-spec-prefill.cpp`
Open the file; each existing test is a `static void test_*(...)` function called from `main()`. Add these between `test_end_to_end_spec_prefill` and `main`:

```cpp
static void test_short_prompt(llama_model*, llama_model*, llama_context*, llama_context*);
static void test_single_token(...);
static void test_keep_ratio_one(...);
static void test_keep_ratio_tiny(...);
static void test_eos_in_lookahead(...);
static void test_determinism(...);
static void test_vocab_mismatch(...);
```
Each follows the pattern of existing tests (init context, call pipeline, `assert` on outputs). Register in `main()`.

### 2.2 Build & run
```bash
cd $REPO
cmake --build build --target test-spec-prefill -j
./build/bin/test-spec-prefill $MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf 2>&1 | tee test_spec_prefill.log
```
✅ Last line reads `[PASS] 15/15` (8 existing + 7 new).

### 2.3 Sanitizer run
```bash
cd $REPO
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
cmake --build build-asan --target test-spec-prefill -j
ASAN_OPTIONS=detect_leaks=1 ./build-asan/bin/test-spec-prefill \
    $MODELS/qwen2.5-0.5b-instruct-q4_k_m.gguf 2>&1 | tee asan.log
```
✅ No `ERROR:` lines in `asan.log`.

### 2.4 CTest registration
Edit `tests/CMakeLists.txt`, add:
```cmake
llama_build_and_test(test-spec-prefill.cpp LABEL "main"
                     ARGS ${CMAKE_SOURCE_DIR}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
```
Run:
```bash
cd $REPO/build && ctest -R spec-prefill --output-on-failure
```
✅ Exit code 0.

---

## Step 2.5 — Quality Sanity Gate

**Why this step exists:** before spending CUDA-machine time on parity (Step 1), confirm the filter produces sane output on any box. Uses the already-built binaries and the Qwen 0.5B that's already on the bench server.

### 2.5.1 Build a tiny shim (if bench binary doesn't decode)
The existing `test-spec-prefill-bench` only times the pipeline; it doesn't generate text. Create `tests/test-spec-prefill-quality.cpp` that:
- Takes `--base`, `--spec`, `--keep-ratio`, `--prompt-file` (JSONL), `--out` (JSONL).
- For each prompt: tokenize, call `llama_spec_prefill()`, run greedy decode for 64 tokens, write `{id, keep_ratio, output, n_kept, n_total}`.
- ~120 lines of C++ — mostly lifted from `test-spec-prefill-bench.cpp` + the decode loop from `examples/simple/simple.cpp`.

Register in `tests/CMakeLists.txt`, rebuild:
```bash
cmake --build build --target test-spec-prefill-quality -j
```

### 2.5.2 Fetch 20 LongBench prompts
```bash
python - <<'EOF'
import json
from datasets import load_dataset
ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test").select(range(20))
with open("tests/data/quality_gate_prompts.jsonl", "w") as f:
    for i, r in enumerate(ds):
        ctx = r["context"][:6000]
        prompt = ctx + "\n\nQuestion: " + r["input"] + "\nAnswer:"
        f.write(json.dumps({
            "id": i, "prompt": prompt, "answers": r["answers"]
        }) + "\n")
EOF
```
✅ File has 20 lines.

### 2.5.3 Run the sweep on the bench box
```bash
# On the remote:
cd /tmp/llama-cpp-bench
for kr in 1.0 0.5 0.25 0.1; do
  ./build/bin/test-spec-prefill-quality \
    --base models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --spec models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --keep-ratio $kr \
    --prompt-file tests/data/quality_gate_prompts.jsonl \
    --out results/quality/kr_${kr}.jsonl
done
# Copy results back:
# scp coder-bench:/tmp/llama-cpp-bench/results/quality/*.jsonl results/quality/
```
✅ 4 output files, each with 20 lines.

### 2.5.4 Score
```bash
pip install rouge-score
python - <<'EOF'
import json, glob
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
for path in sorted(glob.glob("results/quality/kr_*.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    scores = []
    for r in rows:
        best = max(scorer.score(ans, r["output"])["rougeL"].fmeasure
                   for ans in r["answers"])
        scores.append(best)
    print(f"{path}: mean Rouge-L = {sum(scores)/len(scores):.3f}")
EOF
```

### 2.5.5 Acceptance
- At `kr=0.25`: mean Rouge-L ≥ 0.70 × mean at `kr=1.0`.
- At `kr=0.10`: eyeball 5 outputs — must still be coherent English (not token salad).
- **Fail → stop**. Don't run Steps 1/3/4 until fixed.

---

## Step 3 — Quality Evaluation

### 3.1 Build the CLI driver
Create `examples/spec-prefill-run/`:
```
examples/spec-prefill-run/
    CMakeLists.txt
    spec-prefill-run.cpp
```
`spec-prefill-run.cpp` skeleton:
```cpp
// Parses --base, --spec, --keep-ratio, --lookahead, --pool, --chunk-size,
// --prompt-file (.jsonl with {id, prompt}), --out (.jsonl)
// For each prompt: tokenize, run llama_spec_prefill(), decode N tokens,
// emit {id, output, n_kept, n_total, ttft_ms, tok_per_sec}.
```
Add to `examples/CMakeLists.txt`: `add_subdirectory(spec-prefill-run)`.

Build:
```bash
cmake --build build --target llama-spec-prefill-run -j
```
✅ Binary exists at `./build/bin/llama-spec-prefill-run`.

### 3.2 LongBench driver script
Create `eval/run_longbench.py`:
```python
import argparse, subprocess, json, os
from datasets import load_dataset
TASKS = ["narrativeqa", "qasper", "multifieldqa_en",
         "hotpotqa", "gov_report", "passage_retrieval_en"]
ap = argparse.ArgumentParser()
ap.add_argument("--driver", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--keep-ratios", nargs="+", type=float, default=[0.1, 0.25, 1.0])
ap.add_argument("--out", required=True)
args = ap.parse_args()

for task in TASKS:
    ds = load_dataset("THUDM/LongBench", task, split="test").select(range(50))
    prompts_path = f"{args.out}/{task}_prompts.jsonl"
    os.makedirs(args.out, exist_ok=True)
    with open(prompts_path, "w") as f:
        for i, r in enumerate(ds):
            f.write(json.dumps({"id": i,
                "prompt": r["context"] + "\n\n" + r["input"],
                "answer": r["answers"]}) + "\n")
    for kr in args.keep_ratios:
        out_path = f"{args.out}/{task}_kr{kr}.jsonl"
        subprocess.check_call([args.driver,
            "--base", args.base, "--spec", args.spec,
            "--keep-ratio", str(kr), "--lookahead", "8", "--pool", "13",
            "--prompt-file", prompts_path, "--out", out_path])
```
Run:
```bash
python eval/run_longbench.py \
    --driver ./build/bin/llama-spec-prefill-run \
    --base   $MODELS/llama-3.1-8b-instruct-f16.gguf \
    --spec   $MODELS/llama-3.2-1b-instruct-f16.gguf \
    --keep-ratios 0.10 0.25 1.00 \
    --out    results/longbench
```
✅ `ls results/longbench/*.jsonl` shows 6 tasks × 3 ratios = 18 output files.

### 3.3 Score with LongBench official metrics
```bash
git clone https://github.com/THUDM/LongBench.git /tmp/longbench-ref
python eval/score_longbench.py \
    --results-dir results/longbench \
    --metrics-module /tmp/longbench-ref/metrics.py \
    --out results/longbench/scores.csv
```
You'll need to write `score_longbench.py` (~30 lines) that imports `/tmp/longbench-ref/metrics.py`, iterates result JSONLs, and applies the task-appropriate metric.

✅ `scores.csv` has columns `task, keep_ratio, score`.

### 3.4 Acceptance check
```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/longbench/scores.csv")
base = df[df.keep_ratio == 1.00].set_index("task")["score"]
for kr in [0.10, 0.25]:
    d = (base - df[df.keep_ratio == kr].set_index("task")["score"]).abs()
    print(f"keep_ratio={kr}: max drop={d.max():.2f}, mean drop={d.mean():.2f}")
EOF
```
✅ At `kr=0.25`: mean drop ≤ 2.0. At `kr=0.10`: mean drop ≤ 5.0 (on QA/summary tasks).

### 3.5 RULER
```bash
git clone https://github.com/NVIDIA/RULER.git /tmp/ruler
# Use its prompt-generation script to produce JSONL at lengths 4k/8k/16k/32k,
# then feed through the same driver, score with RULER's eval script.
```
Same structure as LongBench. ✅ Pass-rate within 5 points at 8k.

---

## Step 4 — Performance Evaluation

### 4.1 Extend the bench binary
Edit `tests/test-spec-prefill-bench.cpp`: add CLI args `--ctx-lens`, `--keep-ratios`, `--csv`, a warmup loop (3 discarded, 10 measured, median), and CSV output.

```bash
cmake --build build --target test-spec-prefill-bench -j
./build/bin/test-spec-prefill-bench \
    --base $MODELS/llama-3.1-8b-instruct-f16.gguf \
    --spec $MODELS/llama-3.2-1b-instruct-f16.gguf \
    --ctx-lens 2048,4096,8192,16384,32768 \
    --keep-ratios 0.1,0.25,0.5 \
    --csv results/perf/ttft_cpu.csv
```
✅ CSV has 5 × 3 = 15 rows. `speedup` column > 1.0 for large contexts.

### 4.2 CUDA build (on GPU box)
```bash
cd $REPO
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target test-spec-prefill-bench -j
./build-cuda/bin/test-spec-prefill-bench ... --csv results/perf/ttft_cuda.csv
```
✅ CUDA speedup at ctx=8192, kr=0.25 is ≥ 1.5×.

### 4.3 Metal build (on Mac)
```bash
cmake -B build-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --target test-spec-prefill-bench -j
./build-metal/bin/test-spec-prefill-bench ... --csv results/perf/ttft_metal.csv
```

### 4.4 Server QPS
```bash
cd $REPO
cmake --build build --target llama-server -j
# Baseline
./build/bin/llama-server --model $MODELS/llama-3.1-8b-instruct-f16.gguf \
    --port 8080 &
SERVER=$!
python tools/server/bench/server-bench.py --url http://localhost:8080 \
    --concurrency 1,2,4,8,16 --out results/perf/qps_base.csv
kill $SERVER
# With spec-prefill (requires server flag you'll add, or a separate binary)
```
✅ Max QPS with spec-prefill ≥ 1.3× of baseline at equal p99 TTFT.

### 4.5 Overhead breakdown
Add to `src/llama-spec-prefill.cpp`:
```cpp
#define TIME_STAGE(name, body) { \
    auto t0 = std::chrono::high_resolution_clock::now(); \
    body; \
    auto t1 = std::chrono::high_resolution_clock::now(); \
    if (std::getenv("LLAMA_SPEC_PREFILL_PROFILE")) \
        fprintf(stderr, "[prof] %s: %.2f ms\n", name, \
            std::chrono::duration<double,std::milli>(t1-t0).count()); \
}
```
Wrap the four stages in `llama_spec_prefill()`. Run once with the flag set; pipe stderr to a file. ✅ Stage totals sum to ~100% of the pipeline time.

---

## Step 5 — Ablations

### 5.1 Grid runner
`eval/run_ablations.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd $REPO
mkdir -p results/ablation
for lah in 1 4 8 16; do
  for pool in 1 7 13; do
    for chunk in 0 16 32 64; do
      for spec in $MODELS/qwen2.5-0.5b*.gguf $MODELS/llama-3.2-1b*.gguf; do
        tag="lah${lah}_pool${pool}_chunk${chunk}_$(basename $spec .gguf)"
        python eval/run_longbench.py \
            --driver ./build/bin/llama-spec-prefill-run \
            --base $MODELS/llama-3.1-8b-instruct-f16.gguf \
            --spec $spec --keep-ratios 0.25 \
            --extra "--lookahead $lah --pool $pool --chunk-size $chunk" \
            --out results/ablation/$tag
      done
    done
  done
done
```
Run overnight. ✅ `results/ablation/` has 96+ subdirs.

### 5.2 Aggregate & plot
```python
# eval/plot_pareto.py
import pandas as pd, matplotlib.pyplot as plt, glob, json
rows = []
for d in glob.glob("results/ablation/*"):
    cfg = d.split("/")[-1]
    # parse cfg string, load scores + ttft, append row
    ...
df = pd.DataFrame(rows)
plt.scatter(df.ttft_ms, df.score)
plt.xlabel("TTFT (ms)"); plt.ylabel("LongBench score")
plt.savefig("results/ablation/pareto.png")
```
✅ Inspect `pareto.png`. Find the config(s) on the frontier. If current defaults (`lah=8,pool=13,chunk=32`) are not within 2% of the frontier, update `llama_spec_prefill_params` ctor in `include/llama-spec-prefill.h`.

---

## Step 6 — Integration & Regression

### 6.1 Feature-off identity
```bash
cd $REPO
# Record baseline from pre-spec-prefill commit
git stash
git checkout ae83b1def^
cmake --build build --target llama-perplexity -j
./build/bin/llama-perplexity -m $MODELS/llama-3.2-1b-instruct-f16.gguf \
    -f wikitext-2-raw/wiki.test.raw > /tmp/ppl_before.txt
git checkout spec-prefill
git stash pop
cmake --build build --target llama-perplexity -j
./build/bin/llama-perplexity -m $MODELS/llama-3.2-1b-instruct-f16.gguf \
    -f wikitext-2-raw/wiki.test.raw > /tmp/ppl_after.txt
diff /tmp/ppl_before.txt /tmp/ppl_after.txt
```
✅ `diff` shows only timestamp differences (no numeric divergence).

### 6.2 CI smoke gate
Add to `.github/workflows/build.yml` a new job `spec-prefill-smoke`:
```yaml
  spec-prefill-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: build
        run: cmake -B build && cmake --build build --target test-spec-prefill llama-spec-prefill-run -j
      - name: download tiny models
        run: |
          mkdir -p models
          curl -L -o models/qwen0.5b.gguf https://huggingface.co/.../qwen2.5-0.5b.gguf
          curl -L -o models/qwen0.1b.gguf https://huggingface.co/.../qwen0.1b-draft.gguf
      - name: unit tests
        run: ./build/bin/test-spec-prefill models/qwen0.5b.gguf
      - name: smoke quality
        run: python eval/smoke_quality.py --driver ./build/bin/llama-spec-prefill-run \
                    --base models/qwen0.5b.gguf --spec models/qwen0.1b.gguf --n 10
        timeout-minutes: 5
```
✅ Job finishes green in < 5 minutes.

### 6.3 Sanity baselines
Add to `tests/test-spec-prefill.cpp`:
```cpp
static int filter_random(...);   // returns kept indices at random
static int filter_last_n(...);   // returns last N indices
```
Run the driver three ways (spec-prefill, random, last-N) on the LongBench subset; compare scores.
```bash
python eval/compare_baselines.py --results-dir results/baselines
```
✅ spec-prefill mean score ≥ random + 5 points AND ≥ last-N + 5 points.

---

## Final gate

Before declaring "done":
- [ ] Step 1: parity IoU ≥ 0.95 at kr=0.25.
- [ ] Step 2: 15/15 tests pass, zero ASan reports.
- [ ] Step 3: LongBench drop ≤ 2 pts at kr=0.25.
- [ ] Step 4: TTFT speedup ≥ 1.5× at 8k on GPU.
- [ ] Step 5: defaults within 2% of Pareto frontier.
- [ ] Step 6: CI smoke job green; baselines beaten by ≥5 pts.

All checked → the C++ spec-prefill port is validated against the reference paper.
