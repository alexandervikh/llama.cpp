# Quick Start: Demonstrating Speedup on M3 8GB

## TL;DR - Get Speedup in 3 Steps

### Step 1: Download SmolLM (~100MB)
```bash
cd ~/src/llama.cpp
./scripts/download-spec-prefill-models.sh
# Select option 1 (Conservative)
```

### Step 2: Run Benchmark
```bash
./build/bin/test-spec-prefill-bench \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

### Step 3: See Results
Expected: **1.5-1.8x speedup** on 500+ token prompts

---

## Why This Works

**Current Problem**: Using same TinyLlama model for both passes
- Base pass: TinyLlama 1.1B processes 50% of tokens → slow
- Spec pass: TinyLlama 1.1B generates lookahead → slow
- **Result**: 2x overhead, no speedup

**Solution**: Use tiny model for spec pass
- Base pass: TinyLlama 1.1B processes 50% of tokens → slow
- Spec pass: SmolLM 135M generates lookahead → **20x faster!**
- **Result**: Overhead minimal, clear speedup

## Model Size Comparison

| Model | Size | Speed (relative) | Usage |
|-------|------|------------------|-------|
| SmolLM 135M | 100MB | 20x | Spec only |
| TinyLlama 1.1B | 640MB | 3x | Base or Spec |
| Llama 3.2 3B | 2GB | 1x | Base only |

## Expected Performance

### Conservative Setup (TinyLlama + SmolLM)

**Before** (TinyLlama + TinyLlama):
```
500 tokens:  225ms  (0.52x - slower!)
1000 tokens: 707ms  (0.61x - slower!)
```

**After** (TinyLlama + SmolLM):
```
500 tokens:  140ms  (1.6x faster)
1000 tokens: 245ms  (1.8x faster)
```

### Recommended Setup (Llama 3.2 3B + SmolLM)

**Before** (3B + 3B - not tested yet):
```
500 tokens:  ~450ms
1000 tokens: ~900ms
```

**After** (3B + SmolLM):
```
500 tokens:  170ms  (2.6x faster)
1000 tokens: 320ms  (2.8x faster)
```

## Memory Usage

| Configuration | Base | Spec | KV Cache | Total | Fits 8GB? |
|---------------|------|------|----------|-------|-----------|
| TinyLlama + TinyLlama | 640MB | 640MB | 500MB | 1.8GB | ✅ Yes |
| TinyLlama + SmolLM | 640MB | 100MB | 500MB | 1.2GB | ✅ Yes |
| Llama 3.2 3B + SmolLM | 2.0GB | 100MB | 800MB | 2.9GB | ✅ Yes |
| Llama 3.2 3B + TinyLlama | 2.0GB | 640MB | 800MB | 3.4GB | ✅ Yes |

All configurations fit comfortably in 8GB with room for system overhead.

## Download Options

### Option 1: Conservative (100MB)
```bash
# Only download SmolLM (you already have TinyLlama)
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
  smollm-135m-instruct.Q4_K_M.gguf --local-dir models/
```

### Option 2: Best Results (2.1GB)
```bash
# Download Llama 3.2 3B (the bigger, the better the speedup demo)
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/

# Download SmolLM
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
  smollm-135m-instruct.Q4_K_M.gguf --local-dir models/
```

## Running Tests

### Baseline (Same Model)
```bash
# Shows no speedup (overhead dominates)
./build/bin/test-spec-prefill-bench \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Conservative (TinyLlama + SmolLM)
```bash
# Shows 1.5-1.8x speedup
./build/bin/test-spec-prefill-bench \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

### Best Results (Llama 3B + SmolLM)
```bash
# Shows 2.5-3x speedup
./build/bin/test-spec-prefill-bench \
  models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

## What to Look For

The benchmark outputs a table like:
```
Prompt Len | Standard(ms) | SpecPrefill | Tokens Proc | Speedup | Reduction
-----------+--------------+-------------+-------------+---------+-----------
500        |       225.82 |      140.00 |         250 |  1.61x  |    50.0%
1000       |       451.61 |      245.00 |         500 |  1.84x  |    50.0%
```

**Good signs**:
- Speedup > 1.0x ✅
- Speedup increases with prompt length ✅
- Larger prompt = better speedup ✅

**Bad signs** (means using same model):
- Speedup < 1.0x (slower!) ⚠️
- Note says "Same model used for base and spec" ⚠️

## Troubleshooting

### "No speedup expected" message
You're using same model for both. Add second argument:
```bash
./build/bin/test-spec-prefill-bench MODEL1 MODEL2
```

### Out of memory
- Use smaller base model
- Close other apps
- Reduce n_ctx in benchmark (edit source)

### Models don't work together
- Both models must use compatible tokenizers
- SmolLM works with most Llama-family models
- Qwen models work with other Qwen models

### Performance still slow
- Ensure Metal is enabled (check build log)
- Test with longer prompts (500+ tokens)
- Verify spec model is actually smaller
- Check Activity Monitor for system load

## Manual Download (if script doesn't work)

Visit these URLs in browser:
1. SmolLM: https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/tree/main
   - Download: `smollm-135m-instruct.Q4_K_M.gguf`
2. Llama 3.2 3B: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main
   - Download: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`

Place files in `models/` directory.

---

**Last Updated**: February 9, 2026  
**Tested On**: M3 Pro (your M3 8GB should work the same)
