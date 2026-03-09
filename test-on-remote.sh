#!/bin/bash
# Run this script ON the remote machine to test the parameter override fix
# Usage: ./test-on-remote.sh

set -e

MODEL="models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
PORT=8080
BUDGET=20

echo "Killing any existing servers..."
pkill llama-server || true
sleep 2

echo "Starting server with budget=$BUDGET..."
cd ~/llama.cpp
build-test/bin/llama-server \
  --model $MODEL \
  --reasoning-budget $BUDGET \
  --port $PORT \
  -ngl 999 \
  --jinja \
  > /tmp/server-test.log 2>&1 &

echo "Waiting for server to start..."
sleep 35

echo ""
echo "=== TEST 1: Default (no explicit parameters) ==="
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"15*12?"}],"max_tokens":100}' \
  | jq '{usage: .usage, first_20_chars: (.choices[0].message.content[:20])}'

echo ""
echo "=== TEST 2: Explicit thinking_forced_open=true ==="
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"15*12?"}],"max_tokens":100,"thinking_forced_open":true,"thinking_open_tag":"<think>","thinking_close_tag":"</think>"}' \
  > /tmp/result-explicit.json

jq '{usage: .usage, first_20_chars: (.choices[0].message.content[:20])}' /tmp/result-explicit.json

REASONING_TOKENS=$(jq -r '.usage.reasoning_tokens' /tmp/result-explicit.json)

echo ""
echo "=== DEBUG LOGS ==="
echo "REASONING-DEBUG (last 2):"
tail -200 /tmp/server-test.log | grep "REASONING-DEBUG" | tail -2
echo ""
echo "REASONING-INIT:"
tail -200 /tmp/server-test.log | grep "REASONING-INIT" || echo "(none)"
echo ""
echo "REASONING-COUNT (last 5):"
tail -200 /tmp/server-test.log | grep "REASONING-COUNT" | tail -5 || echo "(none)"
echo ""
echo "REASONING-INJECT:"
tail -200 /tmp/server-test.log | grep "REASONING-INJECT" || echo "(none)"

echo ""
echo "=== RESULT ==="
if [ "$REASONING_TOKENS" = "$BUDGET" ]; then
    echo "✅ SUCCESS: reasoning_tokens=$REASONING_TOKENS matches budget=$BUDGET"
else
    echo "⚠️  Got reasoning_tokens=$REASONING_TOKENS, expected $BUDGET"
fi
