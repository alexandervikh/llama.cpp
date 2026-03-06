#!/bin/bash

echo "Waiting for server to start..."
sleep 35

echo "Testing with thinking_forced_open=true..."
curl -s http://main.bench.alexandervikhorev.coder:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 15*12?"}],
    "max_tokens": 100,
    "thinking_forced_open": true,
    "thinking_open_tag": "<think>",
    "thinking_close_tag": "</think>"
  }' > /tmp/test-result.json

echo "=== USAGE ==="
jq .usage /tmp/test-result.json

echo ""
echo "=== REASONING DEBUG LOG (should show thinking_forced_open=1) ==="
ssh main.bench.alexandervikhorev.coder 'tail -100 /tmp/server-fix.log | grep "REASONING-DEBUG"'

echo ""
echo "=== INITIALIZATION LOG (should appear if inline tracking enabled) ==="
ssh main.bench.alexandervikhorev.coder 'tail -100 /tmp/server-fix.log | grep "REASONING-INIT"'

echo ""
echo "=== COUNTING LOGS ==="
ssh main.bench.alexandervikhorev.coder 'tail -100 /tmp/server-fix.log | grep "REASONING-COUNT" | tail -5'

echo ""
echo "=== INJECTION LOG ==="
ssh main.bench.alexandervikhorev.coder 'tail -100 /tmp/server-fix.log | grep "REASONING-INJECT"'
