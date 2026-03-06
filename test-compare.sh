#!/bin/bash

pkill -9 llama-server 2>/dev/null || true
sleep 2

cd ~/llama.cpp

# Test WITHOUT thinking params
echo "=== Starting server with budget 20 ==="
build-test/bin/llama-server \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  --reasoning-budget 20 \
  --port 8080 \
  --ctx-size 4096 \
  -ngl 999 \
  --jinja \
  > /tmp/srv2.log 2>&1 &

echo "Waiting 40 seconds..."
sleep 40

echo ""
echo "=== Test A: WITHOUT thinking_forced_open ==="
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 7*8"}],"max_tokens":100}' \
  > /tmp/ta.json
jq -c '.usage' /tmp/ta.json

echo ""
echo "=== Test B: WITH thinking_forced_open ==="
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 7*8"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/tb.json
jq -c '.usage' /tmp/tb.json

echo ""
echo "=== Test C: WITH chat_template_kwargs ==="
curl -s --max-time 20 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Calculate 7*8"}],"max_tokens":100,"chat_template_kwargs":{"enable_thinking":true}}' \
  > /tmp/tc.json
jq -c '.usage' /tmp/tc.json

echo ""
echo "=== Checking logs ==="
echo "Has 'thinking_forced_open' in params: $(grep -c 'thinking_forced_open.*true' /tmp/srv2.log 2>/dev/null || echo 0)"
echo "REASONING-INIT count: $(grep -c '\[REASONING-INIT\]' /tmp/srv2.log 2>/dev/null || echo 0)"
echo "REASONING-COUNT count: $(grep -c '\[REASONING-COUNT\]' /tmp/srv2.log 2>/dev/null || echo 0)"

echo ""
echo "=== First REASONING log (if any) ==="
grep '\[REASONING' /tmp/srv2.log 2>/dev/null | head -1 || echo "(none)"
