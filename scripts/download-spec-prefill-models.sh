#!/bin/bash
# Download models for speculative prefill testing on M3 8GB

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "========================================="
echo "Speculative Prefill Model Downloader"
echo "========================================="
echo ""
echo "This script downloads models optimized for M3 8GB systems"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found"
    echo ""
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

echo "Select configuration:"
echo ""
echo "1) Conservative (TinyLlama 1.1B + SmolLM 135M) - 100MB download"
echo "   Speedup: 1.5-1.8x | Already have TinyLlama"
echo ""
echo "2) Recommended (Llama 3.2 3B + SmolLM 135M) - 2.1GB download"
echo "   Speedup: 2.5-3x | Best results"
echo ""
echo "3) Alternative (Phi-2 2.7B + Qwen2.5 0.5B) - 1.9GB download"
echo "   Speedup: 2-2.2x | Different model families"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Downloading SmolLM 135M Q4_K_M (~100MB)..."
        huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
            smollm-135m-instruct.Q4_K_M.gguf --local-dir "$MODELS_DIR"
        
        BASE_MODEL="$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        SPEC_MODEL="$MODELS_DIR/smollm-135m-instruct.Q4_K_M.gguf"
        ;;
    
    2)
        echo ""
        echo "Downloading Llama 3.2 3B Q4_K_M (~2GB)..."
        huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
            Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir "$MODELS_DIR"
        
        echo ""
        echo "Downloading SmolLM 135M Q4_K_M (~100MB)..."
        huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
            smollm-135m-instruct.Q4_K_M.gguf --local-dir "$MODELS_DIR"
        
        BASE_MODEL="$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        SPEC_MODEL="$MODELS_DIR/smollm-135m-instruct.Q4_K_M.gguf"
        ;;
    
    3)
        echo ""
        echo "Downloading Phi-2 2.7B Q4_K_M (~1.6GB)..."
        huggingface-cli download TheBloke/phi-2-GGUF \
            phi-2.Q4_K_M.gguf --local-dir "$MODELS_DIR"
        
        echo ""
        echo "Downloading Qwen2.5 0.5B Q4_K_M (~300MB)..."
        huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
            qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir "$MODELS_DIR"
        
        BASE_MODEL="$MODELS_DIR/phi-2.Q4_K_M.gguf"
        SPEC_MODEL="$MODELS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Download Complete!"
echo "========================================="
echo ""
echo "Run benchmark with:"
echo ""
echo "./build/bin/test-spec-prefill-bench \\"
echo "  $BASE_MODEL \\"
echo "  $SPEC_MODEL"
echo ""
echo "Or run all tests:"
echo ""
echo "./build/bin/test-spec-prefill \\"
echo "  $BASE_MODEL"
echo ""
