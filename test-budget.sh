#!/bin/bash

pkill -9 llama-server 2>/dev/null || true
sleep 2

cd ~/llama.cpp
build-test/bin/llama-server \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  --reasoning-budget 20 \
  --port 8080 \
  --ctx-size 4096 \
  -ngl 999 \
  --jinja \
  > /tmp/srv.log 2>&1 &

echo "Waiting 40 seconds for server..."
sleep 40

# Test 1
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 7*8"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/r1.json

# Test 2
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 15*12"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/r2.json

echo "Test 1 (7*8):"
jq -c '.usage' /tmp/r1.json

echo "Test 2 (15*12):"
jq -c '.usage' /tmp/r2.json

echo ""
echo "Log counts:"
echo "INIT: $(grep -c REASONING-INIT /tmp/srv.log 2>/dev/null || echo 0)"
echo "COUNT: $(grep -c REASONING-COUNT /tmp/srv.log 2>/dev/null || echo 0)"
echo "INJECT: $(grep -c REASONING-INJECT /tmp/srv.log 2>/dev/null || echo 0)"
