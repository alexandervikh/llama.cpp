#include "tools/server/server-context.h"

#include <cassert>
#include <optional>
#include <string>

static void test_budget_zero_closes_detected_open_tag() {
    bool in_thinking_block = false;
    int32_t n_thinking_tokens = 0;

    auto forced_close = server_reasoning_budget_handle_thinking_transition(
        in_thinking_block,
        n_thinking_tokens,
        /* reasoning_budget = */ 0,
        /* open_tag = */ "<|channel|>analysis<|message|>",
        /* close_tag = */ "<|end|>",
        /* candidate = */ "<|channel|>analysis<|message|>");

    assert(forced_close.has_value());
    assert(*forced_close == "<|end|>");
    assert(!in_thinking_block);
    assert(n_thinking_tokens == 0);
}

static void test_positive_budget_enters_thinking_block() {
    bool in_thinking_block = false;
    int32_t n_thinking_tokens = 0;

    auto forced_close = server_reasoning_budget_handle_thinking_transition(
        in_thinking_block,
        n_thinking_tokens,
        /* reasoning_budget = */ 20,
        /* open_tag = */ "<|channel|>analysis<|message|>",
        /* close_tag = */ "<|end|>",
        /* candidate = */ "<|channel|>analysis<|message|>");

    assert(!forced_close.has_value());
    assert(in_thinking_block);
    assert(n_thinking_tokens == 0);
}

static void test_budget_zero_closes_existing_thinking_block() {
    bool in_thinking_block = true;
    int32_t n_thinking_tokens = 0;

    auto forced_close = server_reasoning_budget_handle_thinking_transition(
        in_thinking_block,
        n_thinking_tokens,
        /* reasoning_budget = */ 0,
        /* open_tag = */ "<|channel|>analysis<|message|>",
        /* close_tag = */ "<|end|>",
        /* candidate = */ "<|channel|>analysis<|message|>internal step");

    assert(forced_close.has_value());
    assert(*forced_close == "<|end|>");
    assert(!in_thinking_block);
    assert(n_thinking_tokens == 0);
}

int main() {
    test_budget_zero_closes_detected_open_tag();
    test_positive_budget_enters_thinking_block();
    test_budget_zero_closes_existing_thinking_block();
    return 0;
}
