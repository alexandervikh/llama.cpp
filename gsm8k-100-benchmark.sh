#!/bin/bash
# GSM8K 100-problem benchmark for reasoning budgets

cd ~/llama.cpp

if [ ! -f "gsm8k_100.json" ]; then
    echo "Error: gsm8k_100.json not found. Run prepare-gsm8k.sh first."
    exit 1
fi

MODEL="models/gpt-oss-20b-q4_k_m.gguf"
PORT=8080

echo "GSM8K 100-Problem Benchmark"
echo "==========================="
echo ""

# Test function
test_all_problems() {
    local BUDGET=$1
    echo "Testing budget=$BUDGET"
    
    # Start server
    pkill llama-server 2>/dev/null
    sleep 3
    
    build-test/bin/llama-server \
        --model "$MODEL" \
        --reasoning-budget $BUDGET \
        --port $PORT \
        -ngl 999 \
        --jinja \
        --log-disable \
        > /tmp/server-b${BUDGET}.log 2>&1 &
    
    sleep 45
    
    # Process problems with Python
    python3 << 'PYTHON_SCRIPT'
import json
import subprocess
import re
import time

# Load problems
with open('gsm8k_100.json', 'r') as f:
    problems = json.load(f)

correct = 0
total_reasoning = 0
total_completion = 0
total_time = 0

for i, problem in enumerate(problems):
    question = problem['question']
    expected = problem['answer']
    
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/100", flush=True)
    
    # Make request
    start = time.time()
    cmd = [
        'curl', '-s', 'http://localhost:8080/v1/chat/completions',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'messages': [{'role': 'user', 'content': question}],
            'max_tokens': 200
        })
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed = time.time() - start
        total_time += elapsed
        
        response = json.loads(result.stdout)
        content = response['choices'][0]['message']['content']
        usage = response['usage']
        
        reasoning = usage.get('reasoning_tokens', 0)
        completion = usage.get('completion_tokens', 0)
        
        total_reasoning += reasoning
        total_completion += completion
        
        # Extract answer - look for numbers in the response
        # Try to find the expected answer in various formats
        expected_clean = expected.replace(',', '').replace('$', '').strip()
        
        # Check if exact answer appears
        if re.search(r'\b' + re.escape(expected_clean) + r'\b', content):
            correct += 1
        # Also try with commas for large numbers
        elif ',' in expected and expected in content:
            correct += 1
            
    except Exception as e:
        print(f"Error on problem {i+1}: {e}", flush=True)
        continue

# Calculate averages
avg_reasoning = total_reasoning / len(problems) if problems else 0
avg_completion = total_completion / len(problems) if problems else 0
accuracy = (correct / len(problems) * 100) if problems else 0

# Output results
print(f"$BUDGET,{correct},{len(problems)},{avg_reasoning:.1f},{avg_completion:.1f},{accuracy:.1f}")
PYTHON_SCRIPT
}

# Run benchmarks
echo "Budget,Correct,Total,AvgReasoning,AvgCompletion,Accuracy(%)"

for BUDGET in 0 10 50 100 200 300 -1; do
    test_all_problems $BUDGET
done

pkill llama-server 2>/dev/null

echo ""
echo "Benchmark complete!"
