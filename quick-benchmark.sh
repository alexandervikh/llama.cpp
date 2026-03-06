#!/bin/bash
# Quick benchmark for reasoning budgets on GPT-OSS

echo "Starting GPT-OSS reasoning budget benchmark..."
echo ""

# Function to test a budget
test_budget() {
    local BUDGET=$1
    echo "Testing budget=$BUDGET..."
    
    ssh main.bench.alexandervikhorev.coder "pkill llama-server; sleep 2; \
        cd llama.cpp && build-test/bin/llama-server \
        --model models/gpt-oss-20b-q4_k_m.gguf \
        --reasoning-budget $BUDGET \
        --port 8080 -ngl 999 --jinja --log-disable > /tmp/srv.log 2>&1 &" 2>&1 | grep -v "workspace is outdated" &
    
    sleep 35
    
    # Test with timing
    local START=$(date +%s)
    ssh main.bench.alexandervikhorev.coder "curl -s http://localhost:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is (15*8)+(23*4)-(7*6)? Think step by step.\"}],\"max_tokens\":120}'" \
        > /tmp/test-b${BUDGET}.json 2>&1
    local END=$(date +%s)
    local TIME=$((END - START))
    
    local REASONING=$(jq -r '.usage.reasoning_tokens // 0' /tmp/test-b${BUDGET}.json)
    local COMPLETION=$(jq -r '.usage.completion_tokens // 0' /tmp/test-b${BUDGET}.json)
    local TOTAL=$(jq -r '.usage.total_tokens // 0' /tmp/test-b${BUDGET}.json)
    
    echo "$BUDGET,$REASONING,$COMPLETION,$TOTAL,$TIME"
}

echo "Budget,Reasoning_Tokens,Completion_Tokens,Total_Tokens,Time(s)"
echo "---------------------------------------------------------------"

test_budget -1
test_budget 0  
test_budget 10
test_budget 50

ssh main.bench.alexandervikhorev.coder 'pkill llama-server' 2>/dev/null

echo ""
echo "Benchmark complete!"
