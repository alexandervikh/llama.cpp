// tests/test-lazyllm-topk.cpp
// Unit tests for llama_lazyllm_top_k_indices — no model required.
#include "llama-lazyllm.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

static void test_topk_last_always_kept() {
    // Last token must always be in result regardless of its score.
    std::vector<float> scores = {10.f, 9.f, 8.f, 7.f, 0.f}; // last has lowest score
    auto kept = llama_lazyllm_top_k_indices(scores, 0.4f, 5); // keep 2 = ceil(5*0.4)
    bool has_last = false;
    for (int32_t k : kept) { if (k == 4) { has_last = true; } }
    (void)has_last;
    assert(has_last);
    printf("PASS: test_topk_last_always_kept\n");
}

static void test_topk_ascending_order() {
    // Returned indices must be in ascending order.
    std::vector<float> scores = {1.f, 5.f, 3.f, 4.f, 2.f};
    auto kept = llama_lazyllm_top_k_indices(scores, 0.6f, 5);
    for (size_t i = 1; i < kept.size(); i++) {
        assert(kept[i] > kept[i-1]);
    }
    printf("PASS: test_topk_ascending_order\n");
}

static void test_topk_ratio_1() {
    // keep_ratio=1.0 → all tokens kept.
    std::vector<float> scores = {3.f, 1.f, 4.f, 1.f, 5.f};
    const int n = 5;
    auto kept = llama_lazyllm_top_k_indices(scores, 1.0f, n);
    assert((int)kept.size() == n);
    for (int i = 0; i < n; i++) assert(kept[(size_t)i] == i);
    printf("PASS: test_topk_ratio_1\n");
}

static void test_topk_selects_best() {
    // Verify the highest-scoring tokens are selected.
    // scores = {0, 1, 10, 0, 5}  — top-2: indices 2 (10) and 4 (5), plus last=4 (already in)
    std::vector<float> scores = {0.f, 1.f, 10.f, 0.f, 5.f};
    auto kept = llama_lazyllm_top_k_indices(scores, 0.4f, 5); // ceil(5*0.4)=2
    // Must contain 2 (score=10) and 4 (score=5, and also last).
    bool has2 = false, has4 = false;
    for (int32_t k : kept) {
        if (k == 2) { has2 = true; }
        if (k == 4) { has4 = true; }
    }
    (void)has2; (void)has4;
    assert(has2 && has4);
    printf("PASS: test_topk_selects_best\n");
}

static void test_topk_min_one() {
    // keep_ratio approaching 0 must still keep at least 1 token (the last).
    std::vector<float> scores = {9.f, 8.f, 7.f, 6.f, 5.f};
    auto kept = llama_lazyllm_top_k_indices(scores, 0.0001f, 5);
    assert(!kept.empty());
    printf("PASS: test_topk_min_one\n");
}

static void test_topk_empty() {
    std::vector<float> scores;
    auto kept = llama_lazyllm_top_k_indices(scores, 0.5f, 0);
    assert(kept.empty());
    printf("PASS: test_topk_empty\n");
}

int main() {
    test_topk_last_always_kept();
    test_topk_ascending_order();
    test_topk_ratio_1();
    test_topk_selects_best();
    test_topk_min_one();
    test_topk_empty();
    printf("ALL PASS: test-lazyllm-topk\n");
    return 0;
}
