#!/bin/bash
# Simpler benchmark script that runs on remote machine

QUESTION="Calculate: (15 * 8) + (23 * 4) - (7 * 6). Show your work."
MODEL="models/gpt-oss-20b-q4_k_m.gguf"
PORT=8080

echo "Budget,Reasoning_Tokens,Completion_Tokens,Total_Tokens,Time_Seconds"

for BUDGET in -1 0 10 50; do
    # Restart server
    pkill llama-server 2>/dev/null || true
    sleep 2
    
    cd ~/llama.cpp
    build-test/bin/llama-server \
        --model "$MODEL" \
        --reasoning-budget $BUDGET \
        --port $PORT \
        -ngl 999 \
        --jinja \
        --log-disable \
        > /tmp/server-b${BUDGET}.log 2>&1 &
    
    sleep 35
    
    # Run test
    START=$(date +%s.%N)
    curl -s http://localhost:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}],\"max_tokens\":150}" \
        > /tmp/result-b${BUDGET}.json
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    
    # Extract data
    REASONING=$(jq -r '.usage.reasoning_tokens // 0' /tmp/result-b${BUDGET}.json)
    COMPLETION=$(jq -r '.usage.completion_tokens' /tmp/result-b${BUDGET}.json)
    TOTAL=$(jq -r '.usage.total_tokens' /tmp/result-b${BUDGET}.json)
    
    echo "$BUDGET,$REASONING,$COMPLETION,$TOTAL,$TIME"
    
    # Save answer preview
    jq -r '.choices[0].message.content' /tmp/result-b${BUDGET}.json | head -5 > /tmp/answer-b${BUDGET}.txt
done

pkill llama-server 2>/dev/null || true

echo ""
echo "=== Answer Previews ==="
for BUDGET in -1 0 10 50; do
    echo ""
    echo "Budget=$BUDGET:"
    cat /tmp/answer-b${BUDGET}.txt
done
