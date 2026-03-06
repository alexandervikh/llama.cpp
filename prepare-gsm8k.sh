#!/bin/bash
# Download and prepare GSM8K dataset (100 problems)

cd ~/llama.cpp

# Download GSM8K test set if not exists
if [ ! -f "gsm8k_test.jsonl" ]; then
    echo "Downloading GSM8K test set..."
    curl -sL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -o gsm8k_test.jsonl
fi

# Extract 100 problems (evenly spaced from the dataset)
echo "Extracting 100 problems..."
python3 << 'PYTHON_SCRIPT'
import json
import random

# Set seed for reproducibility
random.seed(42)

# Read full test set
problems = []
with open('gsm8k_test.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        # Extract final answer (usually in format #### number)
        answer_part = data['answer'].split('####')[-1].strip()
        # Clean answer - remove commas, dollar signs, etc
        answer = answer_part.replace(',', '').replace('$', '').strip()
        
        problems.append({
            'question': data['question'],
            'answer': answer
        })

# Sample 100 problems evenly
total = len(problems)
indices = sorted(random.sample(range(total), min(100, total)))
selected = [problems[i] for i in indices]

print(f"Total problems: {total}")
print(f"Selected: {len(selected)}")

# Save to file
with open('gsm8k_100.json', 'w') as f:
    json.dump(selected, f, indent=2)

print("Saved to gsm8k_100.json")
PYTHON_SCRIPT

if [ -f "gsm8k_100.json" ]; then
    echo "✓ Dataset prepared: gsm8k_100.json"
    echo "Sample size: $(cat gsm8k_100.json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
else
    echo "✗ Failed to prepare dataset"
    exit 1
fi
