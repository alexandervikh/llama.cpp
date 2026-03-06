#!/bin/bash
# Test script to verify parameter override fix for reasoning budget

set -e

REMOTE="main.bench.alexandervikhorev.coder"
MODEL="models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
PORT=8080

echo "==================================================================="
echo "Testing Parameter Override Fix for Reasoning Budget"
echo "==================================================================="

echo ""
echo "Step 1: Kill any existing servers..."
ssh $REMOTE 'pkill llama-server || true'
sleep 2

echo ""
echo "Step 2: Start server with reasoning-budget=20..."
ssh $REMOTE "cd llama.cpp && build-test/bin/llama-server \
  --model $MODEL \
  --reasoning-budget 20 \
  --port $PORT \
  -ngl 999 \
  --jinja \
  > /tmp/server-final-test.log 2>&1 &"

echo "Waiting 35 seconds for server to load model..."
sleep 35

echo ""
echo "Step 3: Test WITHOUT explicit parameters (should use post-processing)..."
curl -s http://$REMOTE:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 15*12?"}],
    "max_tokens": 100
  }' > /tmp/test-default.json

echo "Results (default):"
jq .usage /tmp/test-default.json
DEFAULT_REASONING=$(jq -r '.usage.reasoning_tokens' /tmp/test-default.json)

echo ""
echo "Step 4: Test WITH explicit parameters (should enforce budget with inline tracking)..."
curl -s http://$REMOTE:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 15*12?"}],
    "max_tokens": 100,
    "thinking_forced_open": true,
    "thinking_open_tag": "<think>",
    "thinking_close_tag": "</think>"
  }' > /tmp/test-explicit.json

echo "Results (explicit):"
jq .usage /tmp/test-explicit.json
EXPLICIT_REASONING=$(jq -r '.usage.reasoning_tokens' /tmp/test-explicit.json)

echo ""
echo "Step 5: Check debug logs..."
echo "--- REASONING-DEBUG (should show thinking_forced_open=1) ---"
ssh $REMOTE 'tail -200 /tmp/server-final-test.log | grep "REASONING-DEBUG" | tail -2'

echo ""
echo "--- REASONING-INIT (should appear for explicit test) ---"
ssh $REMOTE 'tail -200 /tmp/server-final-test.log | grep "REASONING-INIT" | tail -1'

echo ""
echo "--- REASONING-COUNT (should show token counting) ---"
ssh $REMOTE 'tail -200 /tmp/server-final-test.log | grep "REASONING-COUNT" | tail -5'

echo ""
echo "--- REASONING-INJECT (should show budget enforcement at 20 tokens) ---"
ssh $REMOTE 'tail -200 /tmp/server-final-test.log | grep "REASONING-INJECT" | tail -1'

echo ""
echo "==================================================================="
echo "SUMMARY"
echo "==================================================================="
echo "Test 1 (default):  reasoning_tokens = $DEFAULT_REASONING (no enforcement)"
echo "Test 2 (explicit): reasoning_tokens = $EXPLICIT_REASONING (should be ~20)"
echo ""

if [ "$EXPLICIT_REASONING" = "20" ]; then
    echo "✅ SUCCESS: Budget enforcement working correctly!"
else
    echo "⚠️  ISSUE: Expected reasoning_tokens=20, got $EXPLICIT_REASONING"
fi

echo ""
echo "Full responses:"
echo "--- Default ---"
jq -r '.choices[0].message.content' /tmp/test-default.json | head -5
echo ""
echo "--- Explicit ---"
jq -r '.choices[0].message.content' /tmp/test-explicit.json | head -5
