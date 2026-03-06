#!/bin/bash

# Kill existing server
ssh main.bench.alexandervikhorev.coder 'pkill llama-server; sleep 2'

# Start server with reasoning budget
ssh main.bench.alexandervikhorev.coder 'cd llama.cpp && nohup build-test/bin/llama-server --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --reasoning-budget 20 --port 8080 -ngl 999 --jinja 2>&1 | tee /tmp/test-fixed.log &' &

# Wait for server to start
sleep 40

# Test with explicit thinking parameters
echo "Testing with thinking_forced_open=true..."
curl -s http://main.bench.alexandervikhorev.coder:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 7*8?"}],
    "max_tokens": 100,
    "thinking_forced_open": true,
    "thinking_open_tag": "<think>",
    "thinking_close_tag": "</think>"
  }' > /tmp/result-fixed.json

echo ""
echo "=== Usage Stats ==="
jq .usage /tmp/result-fixed.json

echo ""
echo "=== Debug Log ==="
ssh main.bench.alexandervikhorev.coder 'grep "REASONING-DEBUG" /tmp/test-fixed.log | head -3'

echo ""
echo "=== Initialization ==="
ssh main.bench.alexandervikhorev.coder 'grep "REASONING-INIT" /tmp/test-fixed.log | head -3'

echo ""
echo "=== Counting ==="
ssh main.bench.alexandervikhorev.coder 'grep "REASONING-COUNT" /tmp/test-fixed.log | tail -5'

echo ""
echo "=== Budget Enforcement ==="
ssh main.bench.alexandervikhorev.coder 'grep "REASONING-INJECT\\|REASONING-CLOSE" /tmp/test-fixed.log | head -3'

echo ""
echo "=== Generated Response ==="
jq -r '.choices[0].message.content' /tmp/result-fixed.json | head -20
