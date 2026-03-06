#!/bin/bash
# GSM8K subset benchmark for reasoning budgets
# 10 problems from GSM8K test set

MODEL="models/gpt-oss-20b-q4_k_m.gguf"
PORT=8080

# GSM8K problems with answers
declare -a PROBLEMS=(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden in total?"
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
    "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?"
    "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?"
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by her hourly wage + 1/2 her hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"
)

declare -a ANSWERS=(
    "72"
    "10"
    "5"
    "42"
    "624"
    "35"
    "48"
    "16"
    "41"
    "990"
)

echo "GSM8K Subset Benchmark - GPT-OSS Model"
echo "Testing ${#PROBLEMS[@]} problems with budgets: -1, 0, 10, 50"
echo ""

# Results storage
declare -A CORRECT
declare -A TOTAL_TIME
declare -A AVG_REASONING
declare -A AVG_COMPLETION

for BUDGET in -1 0 10 50; do
    echo "=========================================="
    echo "Testing Budget: $BUDGET"
    echo "=========================================="
    
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
        > /tmp/server-gsm-b${BUDGET}.log 2>&1 &
    
    sleep 40
    
    CORRECT[$BUDGET]=0
    TOTAL_TIME[$BUDGET]=0
    TOTAL_REASONING=0
    TOTAL_COMPLETION=0
    
    for i in "${!PROBLEMS[@]}"; do
        PROBLEM="${PROBLEMS[$i]}"
        ANSWER="${ANSWERS[$i]}"
        
        echo "Problem $((i+1))/10..."
        
        START=$(date +%s)
        curl -s http://localhost:$PORT/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROBLEM\"}],\"max_tokens\":200}" \
            > /tmp/result-b${BUDGET}-p${i}.json
        END=$(date +%s)
        
        TIME=$((END - START))
        TOTAL_TIME[$BUDGET]=$((${TOTAL_TIME[$BUDGET]} + TIME))
        
        # Extract answer (look for final number)
        RESPONSE=$(jq -r '.choices[0].message.content' /tmp/result-b${BUDGET}-p${i}.json)
        
        # Simple answer extraction - look for the expected number in response
        if echo "$RESPONSE" | grep -qE "\b$ANSWER\b"; then
            CORRECT[$BUDGET]=$((${CORRECT[$BUDGET]} + 1))
            echo "  ✓ Correct (found $ANSWER)"
        else
            echo "  ✗ Wrong (expected $ANSWER)"
            echo "  Response: $(echo "$RESPONSE" | head -c 100)..."
        fi
        
        # Collect metrics
        REASONING=$(jq -r '.usage.reasoning_tokens // 0' /tmp/result-b${BUDGET}-p${i}.json)
        COMPLETION=$(jq -r '.usage.completion_tokens' /tmp/result-b${BUDGET}-p${i}.json)
        
        TOTAL_REASONING=$((TOTAL_REASONING + REASONING))
        TOTAL_COMPLETION=$((TOTAL_COMPLETION + COMPLETION))
    done
    
    AVG_REASONING[$BUDGET]=$((TOTAL_REASONING / ${#PROBLEMS[@]}))
    AVG_COMPLETION[$BUDGET]=$((TOTAL_COMPLETION / ${#PROBLEMS[@]}))
    
    echo ""
    echo "Budget $BUDGET Results:"
    echo "  Correct: ${CORRECT[$BUDGET]}/${#PROBLEMS[@]}"
    echo "  Avg Reasoning Tokens: ${AVG_REASONING[$BUDGET]}"
    echo "  Avg Completion Tokens: ${AVG_COMPLETION[$BUDGET]}"
    echo "  Total Time: ${TOTAL_TIME[$BUDGET]}s"
    echo ""
done

pkill llama-server 2>/dev/null || true

echo ""
echo "========================================"
echo "FINAL RESULTS TABLE"
echo "========================================"
echo ""
printf "| %-8s | %-10s | %-12s | %-18s | %-18s | %-12s |\n" \
    "Budget" "Accuracy" "Correct" "Avg Reasoning" "Avg Completion" "Total Time"
printf "|----------|------------|--------------|--------------------|--------------------|-------------|\n"

for BUDGET in -1 0 10 50; do
    ACCURACY=$(echo "scale=1; ${CORRECT[$BUDGET]} * 100 / ${#PROBLEMS[@]}" | bc)
    printf "| %-8s | %-9s%% | %-12s | %-18s | %-18s | %-11ss |\n" \
        "$BUDGET" \
        "$ACCURACY" \
        "${CORRECT[$BUDGET]}/${#PROBLEMS[@]}" \
        "${AVG_REASONING[$BUDGET]}" \
        "${AVG_COMPLETION[$BUDGET]}" \
        "${TOTAL_TIME[$BUDGET]}"
done

echo ""
echo "Notes:"
echo "- Budget -1 = unlimited reasoning"
echo "- Budget 0 = no reasoning tokens"
echo "- Budget 10/50 = enforced reasoning token limit"
echo "- Answer matching is simple string search (may have false positives/negatives)"
