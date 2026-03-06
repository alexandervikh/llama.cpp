#!/bin/bash
set -e

echo "Killing existing server..."
pkill -9 llama-server 2>/dev/null || true
sleep 2

echo "Starting server with reasoning budget 20..."
cd ~/llama.cpp
build-test/bin/llama-server \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  --reasoning-budget 20 \
  --port 8080 \
  --ctx-size 4096 \
  -ngl 999 \
  --jinja \
  > /tmp/server-test.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

echo "Waiting 35 seconds for server to load..."
sleep 35

if ! ps -p $SERVER_PID > /dev/null; then
  echo "ERROR: Server failed to start!"
  echo "Last 30 lines of log:"
  tail -30 /tmp/server-test.log
  exit 1
fi

echo ""
echo "Testing with thinking_forced_open=true..."
curl -s --max-time 25 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 7*8"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/test-result.json

echo ""
echo "=== USAGE STATS ==="
jq .usage /tmp/test-result.json

echo ""
echo "=== REASONING INIT LOGS ==="
grep "\[REASONING-INIT\]" /tmp/server-test.log || echo "No init logs found"

echo ""
echo "=== REASONING COUNT LOGS (first 10) ==="
grep "\[REASONING-COUNT\]" /tmp/server-test.log | head -10 || echo "No count logs found"

echo ""
echo "=== REASONING CLOSE/INJECT LOGS ==="
grep -E "\[REASONING-(CLOSE|INJECT)\]" /tmp/server-test.log || echo "No close/inject logs found"

echo ""
echo "Test complete!"
