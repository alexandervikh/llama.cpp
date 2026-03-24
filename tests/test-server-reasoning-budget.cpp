#include "tools/server/server-context.h"

#include <cstdlib>
#include <optional>
#include <string>

static void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}

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

    check(forced_close.has_value());
    check(*forced_close == "<|end|>");
    check(!in_thinking_block);
    check(n_thinking_tokens == 0);
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

    check(!forced_close.has_value());
    check(in_thinking_block);
    check(n_thinking_tokens == 0);
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

    check(forced_close.has_value());
    check(*forced_close == "<|end|>");
    check(!in_thinking_block);
    check(n_thinking_tokens == 0);
}

static void test_reparsed_reasoning_count_wins_when_inline_is_partial() {
    check(server_resolve_reasoning_token_count(
        /* inline_count = */ 8,
        /* reparsed_count = */ 125) == 125);
}

static void test_inline_reasoning_count_kept_when_reparsed_is_smaller() {
    check(server_resolve_reasoning_token_count(
        /* inline_count = */ 20,
        /* reparsed_count = */ 13) == 20);
}

int main() {
    test_budget_zero_closes_detected_open_tag();
    test_positive_budget_enters_thinking_block();
    test_budget_zero_closes_existing_thinking_block();
    test_reparsed_reasoning_count_wins_when_inline_is_partial();
    test_inline_reasoning_count_kept_when_reparsed_is_smaller();
    return 0;
}
