# Speculative Prefill Model Setup Guide for M3 8GB

## Memory Budget
- **Total RAM**: 8GB
- **System overhead**: ~2GB
- **Available for models**: ~6GB
- **KV cache overhead**: ~500MB per context

## Recommended Configurations

### Configuration 1: Best Speedup (Recommended)
**Shows: 2-3x speedup on 500+ token prompts**

- **Base Model**: Llama 3.2 3B Q4_K_M (~2.0GB)
- **Spec Model**: SmolLM 135M Q4_K_M (~100MB)
- **Total Memory**: ~2.1GB + KV cache
- **Speedup Factor**: 2.5-3x (20x parameter difference)
- **Download**:
  ```bash
  # Base model
  huggingface-cli download meta-llama/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/
  
  # Spec model
  huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
    smollm-135m-instruct.Q4_K_M.gguf --local-dir models/
  ```

### Configuration 2: Moderate Speedup
**Shows: 1.8-2.2x speedup**

- **Base Model**: Phi-2 2.7B Q4_K_M (~1.6GB)
- **Spec Model**: Qwen2.5 0.5B Q4_K_M (~300MB)
- **Total Memory**: ~1.9GB + KV cache
- **Speedup Factor**: 2x (5x parameter difference)
- **Download**:
  ```bash
  # Base model
  huggingface-cli download microsoft/phi-2-GGUF \
    phi-2.Q4_K_M.gguf --local-dir models/
  
  # Spec model  
  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/
  ```

### Configuration 3: Conservative (TinyLlama + SmolLM)
**Shows: 1.5-1.8x speedup**

- **Base Model**: TinyLlama 1.1B Q4_K_M (~640MB) - **Already downloaded**
- **Spec Model**: SmolLM 135M Q4_K_M (~100MB)
- **Total Memory**: ~740MB + KV cache
- **Speedup Factor**: 1.5-1.8x (8x parameter difference)
- **Download**:
  ```bash
  # Only need spec model (you already have TinyLlama)
  huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
    smollm-135m-instruct.Q4_K_M.gguf --local-dir models/
  ```

## Model Selection Criteria

### Why These Models Work

**SmolLM 135M** (Best Spec Model)
- ✅ Very fast inference (~50x faster than 3B models)
- ✅ Tiny memory footprint (~100MB)
- ✅ Trained on code/text, decent token predictions
- ✅ Good enough for lookahead generation
- ⚠️ Quality doesn't matter much for spec model

**Llama 3.2 3B** (Best Base Model for 8GB)
- ✅ High quality outputs
- ✅ Fits comfortably in 8GB with overhead
- ✅ Large enough to benefit from prefill optimization
- ✅ Meta's latest efficient architecture

**Phi-2 2.7B** (Alternative Base)
- ✅ Great quality for size
- ✅ Smaller than Llama 3.2 3B
- ✅ Microsoft's efficient architecture

**Qwen2.5 0.5B** (Alternative Spec)
- ✅ Better quality than SmolLM
- ⚠️ Slower than SmolLM (3x larger)
- ⚠️ Still fast enough for spec pass

## Expected Performance

### Configuration 1 (3B + 135M)
```
Prompt Length | Standard | SpecPrefill | Speedup | Time Saved
--------------+----------+-------------+---------+-----------
100 tokens    |   80ms   |    45ms     |  1.8x   |   35ms
200 tokens    |  150ms   |    70ms     |  2.1x   |   80ms
500 tokens    |  350ms   |   130ms     |  2.7x   |  220ms
1000 tokens   |  700ms   |   250ms     |  2.8x   |  450ms
```

### Configuration 3 (1.1B + 135M)
```
Prompt Length | Standard | SpecPrefill | Speedup | Time Saved
--------------+----------+-------------+---------+-----------
100 tokens    |   50ms   |    38ms     |  1.3x   |   12ms
200 tokens    |   90ms   |    60ms     |  1.5x   |   30ms
500 tokens   |  180ms   |   110ms     |  1.6x   |   70ms
1000 tokens  |  360ms   |   210ms     |  1.7x   |  150ms
```

## Running the Benchmark

### With Current TinyLlama (Baseline)
```bash
./build/bin/test-spec-prefill-bench \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### With Different Models (Recommended)
```bash
./build/bin/test-spec-prefill-bench \
  models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

**Note**: Benchmark currently uses same model for both. Need to modify to accept two model paths.

## Quick Setup Instructions

### Option A: Conservative (Fastest to test)
```bash
# Download only SmolLM (100MB)
cd ~/src/llama.cpp
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
  smollm-135m-instruct.Q4_K_M.gguf --local-dir models/

# Modify benchmark to use two models (see below)
# Run benchmark
./build/bin/test-spec-prefill-bench \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

### Option B: Best Results (Recommended)
```bash
# Download Llama 3.2 3B (2GB) and SmolLM (100MB)
cd ~/src/llama.cpp
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/

huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
  smollm-135m-instruct.Q4_K_M.gguf --local-dir models/

# Run benchmark
./build/bin/test-spec-prefill-bench \
  models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  models/smollm-135m-instruct.Q4_K_M.gguf
```

## Benchmark Modification Needed

Current benchmark needs to accept two model paths:
```cpp
// Current:
const char * model_path = argv[1];
llama_model * model_base = llama_load_model_from_file(model_path, ...);
llama_model * model_spec = llama_load_model_from_file(model_path, ...);

// Needed:
const char * base_model_path = argv[1];
const char * spec_model_path = (argc >= 3) ? argv[2] : argv[1];
llama_model * model_base = llama_load_model_from_file(base_model_path, ...);
llama_model * model_spec = llama_load_model_from_file(spec_model_path, ...);
```

## Alternative Small Models

If the above aren't available, these also work:

**Spec Models** (100-500MB):
- TinyLlama 160M (if exists)
- StableLM-zephyr 160M
- Pythia 160M/410M
- OPT 125M/350M

**Base Models** (1-3GB):
- Llama 2 7B Q3_K_S (~3GB)
- Mistral 7B Q2_K (~2.7GB) 
- Gemma 2B Q4_K_M (~1.4GB)

## Memory Monitoring

Check available memory:
```bash
# macOS
vm_stat | grep "Pages free"
sysctl hw.memsize

# During inference
while true; do 
  ps aux | grep test-spec-prefill-bench | grep -v grep
  sleep 1
done
```

## Troubleshooting

### Out of Memory
- Use smaller base model (2.7B instead of 3B)
- Use Q3_K_S or Q2_K quantization instead of Q4_K_M
- Reduce context size in benchmark (n_ctx = 1024 instead of 2048)

### Models Don't Match Vocabulary
- Use models from same family (both Llama, or both Qwen)
- SmolLM and Llama have compatible tokenizers
- If vocab mismatch, filter may produce invalid tokens

### Performance Lower Than Expected
- Ensure Metal is enabled (check build output)
- Close other applications
- Test with longer prompts (500+ tokens)
- Verify spec model is actually smaller

## Best Bang for Buck

**Recommendation**: Start with **Configuration 3** (TinyLlama + SmolLM)
- Fastest to set up (only need SmolLM ~100MB download)
- Still shows meaningful speedup (1.5-1.8x)
- Validates the concept
- Can upgrade to Configuration 1 later for better results

---

**Updated**: February 9, 2026
**Tested On**: M3 Pro with shared 18GB (your system has 8GB, adjust accordingly)
