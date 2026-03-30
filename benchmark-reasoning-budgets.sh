#!/bin/bash
# Benchmark different reasoning budgets on GPT-OSS model

set -e

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to your bench hostname}"
MODEL="models/gpt-oss-20b-q4_k_m.gguf"
PORT=8080

# Test question that benefits from reasoning
QUESTION="Calculate: (15 * 8) + (23 * 4) - (7 * 6). Show your work step by step."

echo "==================================================================="
echo "Reasoning Budget Benchmark - GPT-OSS Model"
echo "==================================================================="
echo "Question: $QUESTION"
echo ""

# Array of budgets to test
BUDGETS=(-1 0 10 50)

# Results arrays
declare -a REASONING_TOKENS
declare -a COMPLETION_TOKENS
declare -a PROMPT_TOKENS
declare -a TIMINGS

for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "--- Testing budget=$BUDGET ---"
    
    # Kill and restart server with new budget
    ssh "$REMOTE_HOST" 'pkill llama-server || true'
    sleep 2
    
    ssh "$REMOTE_HOST" "cd llama.cpp && build-test/bin/llama-server \
        --model $MODEL \
        --reasoning-budget $BUDGET \
        --port $PORT \
        -ngl 999 \
        --jinja \
        --log-disable \
        > /tmp/server-budget-${BUDGET}.log 2>&1 &"
    
    echo "Waiting for server to start..."
    sleep 35
    
    # Run test and measure time
    START=$(date +%s.%N)
    
    ssh "$REMOTE_HOST" "curl -s http://localhost:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
            \"messages\": [{\"role\": \"user\", \"content\": \"$QUESTION\"}],
            \"max_tokens\": 150
        }'" > /tmp/result-budget-${BUDGET}.json
    
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    
    # Extract metrics
    REASONING=$(jq -r '.usage.reasoning_tokens // 0' /tmp/result-budget-${BUDGET}.json)
    COMPLETION=$(jq -r '.usage.completion_tokens' /tmp/result-budget-${BUDGET}.json)
    PROMPT=$(jq -r '.usage.prompt_tokens' /tmp/result-budget-${BUDGET}.json)
    
    REASONING_TOKENS+=($REASONING)
    COMPLETION_TOKENS+=($COMPLETION)
    PROMPT_TOKENS+=($PROMPT)
    TIMINGS+=($ELAPSED)
    
    echo "  reasoning_tokens: $REASONING"
    echo "  completion_tokens: $COMPLETION"
    echo "  time: ${ELAPSED}s"
    
    # Save preview of answer
    jq -r '.choices[0].message.content' /tmp/result-budget-${BUDGET}.json | head -3 > /tmp/preview-${BUDGET}.txt
done

# Kill server
ssh "$REMOTE_HOST" 'pkill llama-server || true'

echo ""
echo "==================================================================="
echo "BENCHMARK RESULTS"
echo "==================================================================="
echo ""
printf "| %-8s | %-16s | %-18s | %-10s |\n" "Budget" "Reasoning Tokens" "Completion Tokens" "Time (s)"
printf "|----------|------------------|--------------------|-----------|\n"

for i in "${!BUDGETS[@]}"; do
    printf "| %-8s | %-16s | %-18s | %-10s |\n" \
        "${BUDGETS[$i]}" \
        "${REASONING_TOKENS[$i]}" \
        "${COMPLETION_TOKENS[$i]}" \
        "${TIMINGS[$i]}"
done

echo ""
echo "==================================================================="
echo "ANSWER PREVIEWS (first 3 lines)"
echo "==================================================================="

for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "--- Budget=$BUDGET ---"
    cat /tmp/preview-${BUDGET}.txt
done

echo ""
echo "Full results saved to /tmp/result-budget-*.json"
