#!/bin/bash
# Quick GSM8K benchmark - runs in single SSH session

cd ~/llama.cpp

# Simple GSM8K problems (easy to verify)
test_problem() {
    local BUDGET=$1
    local PROBLEM=$2
    local EXPECTED=$3
    
    RESULT=$(curl -s http://localhost:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROBLEM Answer with just the number.\"}],\"max_tokens\":150}" 2>/dev/null)
    
    REASONING=$(echo "$RESULT" | jq -r '.usage.reasoning_tokens // 0' 2>/dev/null)
    COMPLETION=$(echo "$RESULT" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
    CONTENT=$(echo "$RESULT" | jq -r '.choices[0].message.content' 2>/dev/null)
    
    # Default to 0 if empty
    REASONING=${REASONING:-0}
    COMPLETION=${COMPLETION:-0}
    CONTENT=${CONTENT:-""}
    
    # Check if answer contains expected number
    if echo "$CONTENT" | grep -qE "\b$EXPECTED\b"; then
        echo "1,$REASONING,$COMPLETION"
    else
        echo "0,$REASONING,$COMPLETION"
    fi
}

echo "Budget,Correct,TotalProblems,AvgReasoning,AvgCompletion,Accuracy"

for BUDGET in -1 0 10 50 100 200 300; do
    # Start server
    pkill llama-server 2>/dev/null
    sleep 2
    
    build-test/bin/llama-server \
        --model models/gpt-oss-20b-q4_k_m.gguf \
        --reasoning-budget $BUDGET \
        --port 8080 -ngl 999 --jinja --log-disable \
        > /dev/null 2>&1 &
    
    sleep 40
    
    CORRECT=0
    TOTAL_REASONING=0
    TOTAL_COMPLETION=0
    
    # Problem 1
    RESULT=$(test_problem $BUDGET "What is 15 times 8?" "120")
    CORRECT=$((CORRECT + $(echo $RESULT | cut -d',' -f1 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_REASONING=$((TOTAL_REASONING + $(echo $RESULT | cut -d',' -f2 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_COMPLETION=$((TOTAL_COMPLETION + $(echo $RESULT | cut -d',' -f3 | grep -E '^[0-9]+$' || echo 0)))
    
    # Problem 2
    RESULT=$(test_problem $BUDGET "What is 23 times 4?" "92")
    CORRECT=$((CORRECT + $(echo $RESULT | cut -d',' -f1 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_REASONING=$((TOTAL_REASONING + $(echo $RESULT | cut -d',' -f2 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_COMPLETION=$((TOTAL_COMPLETION + $(echo $RESULT | cut -d',' -f3 | grep -E '^[0-9]+$' || echo 0)))
    
    # Problem 3
    RESULT=$(test_problem $BUDGET "If Betty needs 100 dollars and has 50 dollars, then gets 15 from parents and 30 from grandparents, how much more does she need?" "5")
    CORRECT=$((CORRECT + $(echo $RESULT | cut -d',' -f1 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_REASONING=$((TOTAL_REASONING + $(echo $RESULT | cut -d',' -f2 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_COMPLETION=$((TOTAL_COMPLETION + $(echo $RESULT | cut -d',' -f3 | grep -E '^[0-9]+$' || echo 0)))
    
    # Problem 4
    RESULT=$(test_problem $BUDGET "James writes 3 pages to 2 friends, twice per week. How many pages per year?" "624")
    CORRECT=$((CORRECT + $(echo $RESULT | cut -d',' -f1 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_REASONING=$((TOTAL_REASONING + $(echo $RESULT | cut -d',' -f2 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_COMPLETION=$((TOTAL_COMPLETION + $(echo $RESULT | cut -d',' -f3 | grep -E '^[0-9]+$' || echo 0)))
    
    # Problem 5
    RESULT=$(test_problem $BUDGET "A large pizza has 16 slices, a small pizza has 8 slices. If you buy 2 large and 2 small pizzas, how many total slices?" "48")
    CORRECT=$((CORRECT + $(echo $RESULT | cut -d',' -f1 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_REASONING=$((TOTAL_REASONING + $(echo $RESULT | cut -d',' -f2 | grep -E '^[0-9]+$' || echo 0)))
    TOTAL_COMPLETION=$((TOTAL_COMPLETION + $(echo $RESULT | cut -d',' -f3 | grep -E '^[0-9]+$' || echo 0)))
    
    PROBLEMS=5
    AVG_REASONING=$((TOTAL_REASONING / PROBLEMS))
    AVG_COMPLETION=$((TOTAL_COMPLETION / PROBLEMS))
    ACCURACY=$((CORRECT * 100 / PROBLEMS))
    
    echo "$BUDGET,$CORRECT,$PROBLEMS,$AVG_REASONING,$AVG_COMPLETION,$ACCURACY%"
done

pkill llama-server 2>/dev/null
