#!/bin/bash

echo "Killing existing server..."
pkill -9 llama-server 2>/dev/null || true

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

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
  if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "Server is ready after $i seconds"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: Server did not become ready in 60 seconds"
    tail -30 /tmp/server-test.log
    exit 1
  fi
  sleep 1
done

echo ""
echo "Testing with thinking_forced_open=true..."
curl -s --max-time 25 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"7*8=?"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/test-result.json

echo ""
echo "=== USAGE STATS ==="
jq .usage /tmp/test-result.json

echo ""
echo "=== REASONING LOGS ==="
echo "INIT:"
grep "\[REASONING-INIT\]" /tmp/server-test.log || echo "  (none)"
echo "COUNT (first 5):"
grep "\[REASONING-COUNT\]" /tmp/server-test.log | head -5 || echo "  (none)"
echo "CLOSE/INJECT:"
grep -E "\[REASONING-(CLOSE|INJECT)\]" /tmp/server-test.log | head -3 || echo "  (none)"

echo ""
echo "Done!"
