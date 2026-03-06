#!/bin/bash

pkill -9 llama-server 2>/dev/null || true
sleep 2

cd ~/llama.cpp

echo "Starting server..."
build-test/bin/llama-server \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  --reasoning-budget 20 \
  --port 8080 \
  --ctx-size 4096 \
  -ngl 999 \
  --jinja \
  2>&1 | tee /tmp/server-debug.log &

echo "Waiting 40 seconds..."
sleep 40

echo ""
echo "=== Test: WITH thinking_forced_open ==="
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":50,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/result.json

echo ""
echo "=== USAGE ==="
jq '.usage' /tmp/result.json

echo ""
echo "=== DEBUG LOGS ==="
grep '\[REASONING-DEBUG\]' /tmp/server-debug.log | tail -5

echo ""
echo "=== INIT LOGS ==="
grep '\[REASONING-INIT\]' /tmp/server-debug.log || echo "(none)"

echo ""
echo "=== COUNT LOGS (first 5) ==="
grep '\[REASONING-COUNT\]' /tmp/server-debug.log | head -5 || echo "(none)"
