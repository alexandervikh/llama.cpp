/*
 * Unit tests for LazyLLM Phase 1 — attention-based token pruning.
 * arXiv:2407.14057
 *
 * Tests covered:
 *   1.  params default constructor values
 *   2.  top_k_indices basic selection
 *   3.  top_k_indices last token always kept
 *   4.  top_k_indices keep all (ratio = 1.0)
 *   5.  apply_pooling pool_size=1 is identity
 *   6.  apply_pooling pool_size=3 smooths a spike
 *   7.  aux_cache store and retrieve
 *   8.  aux_cache pruning_point tracking
 *   9.  aux_cache reset clears everything
 *  10.  llama_lazyllm_init with null ctx returns nullptr
 *
 * Build: compiled as part of tests/CMakeLists.txt via llama_build().
 * Run:   ./test-lazyllm-extract   (exits 0 on all pass, 1 on any fail)
 */

#include "llama-lazyllm.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

/* ── test harness ─────────────────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  assertion failed: %s\n", msg); \
            return false; \
        } \
    } while (0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)

static void report(const char * name, bool ok) {
    if (ok) {
        std::printf("PASS: %s\n", name);
        g_pass++;
    } else {
        std::printf("FAIL: %s\n", name);
        g_fail++;
    }
}

/* ── individual tests ─────────────────────────────────────────────────────── */

static bool test_params_default_constructor() {
    llama_lazyllm_params p{};

    ASSERT_EQ(p.pruning_layers.size(), (size_t)3,
              "pruning_layers default size == 3");
    ASSERT_TRUE(p.pruning_layers.size() >= 3 &&
                p.pruning_layers[0] == 8 &&
                p.pruning_layers[1] == 16 &&
                p.pruning_layers[2] == 24,
                "pruning_layers == {8, 16, 24}");

    ASSERT_EQ(p.keep_ratios.size(), (size_t)3,
              "keep_ratios default size == 3");
    ASSERT_TRUE(p.keep_ratios.size() >= 3 &&
                std::fabs(p.keep_ratios[0] - 0.7f) < 1e-5f &&
                std::fabs(p.keep_ratios[1] - 0.5f) < 1e-5f &&
                std::fabs(p.keep_ratios[2] - 0.3f) < 1e-5f,
                "keep_ratios == {0.7, 0.5, 0.3}");

    ASSERT_EQ(p.pool_kernel_size, 13, "pool_kernel_size == 13");
    return true;
}

static bool test_top_k_basic() {
    // scores = {0.1, 0.9, 0.2, 0.8, 0.3}, n_keep = ceil(5 * 0.6) = 3
    std::vector<float> scores = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f};
    int n_tokens = (int)scores.size();
    float keep_ratio = 0.6f;

    std::vector<int32_t> idx = llama_lazyllm_top_k_indices(scores, keep_ratio, n_tokens);

    int n_keep = (int)std::ceil(n_tokens * keep_ratio);
    ASSERT_EQ((int)idx.size(), n_keep, "result size == n_keep");

    // Result must be sorted ascending (by position)
    for (int i = 1; i < (int)idx.size(); i++) {
        ASSERT_TRUE(idx[i] > idx[i-1], "indices in ascending order");
    }

    // Last token (index 4) must always be kept
    ASSERT_TRUE(std::find(idx.begin(), idx.end(), 4) != idx.end(),
                "last token (index 4) always in result");

    // High-score tokens (indices 1 and 3) must also be included (they + last = 3 = n_keep)
    ASSERT_TRUE(std::find(idx.begin(), idx.end(), 1) != idx.end(),
                "index 1 (score 0.9) included");
    ASSERT_TRUE(std::find(idx.begin(), idx.end(), 3) != idx.end(),
                "index 3 (score 0.8) included");

    return true;
}

static bool test_top_k_last_always_kept() {
    // scores descending: last token has lowest score
    std::vector<float> scores = {0.9f, 0.8f, 0.7f, 0.6f, 0.1f};
    int n_tokens = (int)scores.size();
    float keep_ratio = 0.2f;  // n_keep = ceil(5*0.2) = 1, but last must also be kept

    std::vector<int32_t> idx = llama_lazyllm_top_k_indices(scores, keep_ratio, n_tokens);

    // At minimum last token must be present
    ASSERT_TRUE(std::find(idx.begin(), idx.end(), 4) != idx.end(),
                "last token (index 4) kept even at low ratio");

    // All returned indices must be valid
    for (int32_t i : idx) {
        ASSERT_TRUE(i >= 0 && i < n_tokens, "index in valid range");
    }

    // Result must be sorted ascending
    for (int i = 1; i < (int)idx.size(); i++) {
        ASSERT_TRUE(idx[i] > idx[i-1], "indices in ascending order");
    }

    return true;
}

static bool test_top_k_keep_all() {
    std::vector<float> scores = {0.3f, 0.1f, 0.9f, 0.5f, 0.7f};
    int n_tokens = (int)scores.size();

    std::vector<int32_t> idx = llama_lazyllm_top_k_indices(scores, 1.0f, n_tokens);

    ASSERT_EQ((int)idx.size(), n_tokens, "all tokens returned at ratio 1.0");
    for (int i = 0; i < n_tokens; i++) {
        ASSERT_EQ(idx[i], i, "indices in order 0..n_tokens-1");
    }

    return true;
}

static bool test_pooling_identity() {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> orig   = scores;

    llama_lazyllm_apply_pooling(scores, 1);

    ASSERT_EQ(scores.size(), orig.size(), "length preserved");
    for (int i = 0; i < (int)orig.size(); i++) {
        ASSERT_TRUE(std::fabs(scores[i] - orig[i]) < 1e-5f,
                    "pool_size=1 leaves values unchanged");
    }

    return true;
}

static bool test_pooling_smooths() {
    // A spike in the middle; pool_size=3 should spread it to neighbours
    std::vector<float> scores = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    llama_lazyllm_apply_pooling(scores, 3);

    ASSERT_EQ((int)scores.size(), 5, "output length == input length");

    // Middle value should be lower than 1.0 (energy was redistributed)
    ASSERT_TRUE(scores[2] < 1.0f,
                "middle value decreases after pooling");

    // At least one neighbour should have picked up some energy
    ASSERT_TRUE(scores[1] > 0.0f || scores[3] > 0.0f,
                "neighbour(s) increased after pooling");

    return true;
}

static bool test_aux_cache_store_retrieve() {
    llama_lazyllm_aux_cache cache;

    float data[4] = {1.1f, 2.2f, 3.3f, 4.4f};
    cache.store(42, 1, 4, data);

    const float * got = cache.get(42);
    ASSERT_TRUE(got != nullptr, "get(42) returns non-null after store");
    ASSERT_TRUE(std::fabs(got[0] - data[0]) < 1e-5f,
                "first stored float matches retrieved value");

    ASSERT_TRUE(cache.get(99) == nullptr, "get(99) returns nullptr (not stored)");

    ASSERT_EQ(cache.total_bytes(), (size_t)(4 * sizeof(float)),
              "total_bytes() == 4 * sizeof(float)");

    return true;
}

static bool test_aux_cache_pruning_point() {
    llama_lazyllm_aux_cache cache;

    float dummy[1] = {0.0f};
    cache.store(10, 0, 1, dummy);
    cache.store(20, 2, 1, dummy);

    ASSERT_EQ(cache.get_pruning_point(10), 0,
              "get_pruning_point(10) == 0");
    ASSERT_EQ(cache.get_pruning_point(20), 2,
              "get_pruning_point(20) == 2");
    ASSERT_EQ(cache.get_pruning_point(99), -1,
              "get_pruning_point(99) == -1 (not stored)");

    return true;
}

static bool test_aux_cache_reset() {
    llama_lazyllm_aux_cache cache;

    float data[2] = {7.0f, 8.0f};
    cache.store(5, 0, 2, data);

    ASSERT_TRUE(cache.get(5) != nullptr, "get(5) non-null before reset");

    cache.reset();

    ASSERT_TRUE(cache.get(5) == nullptr, "get(5) nullptr after reset");

    return true;
}

static bool test_init_null_ctx() {
    llama_lazyllm_context * lctx = llama_lazyllm_init(nullptr);
    ASSERT_TRUE(lctx == nullptr, "llama_lazyllm_init(nullptr) == nullptr");
    // Calling free on nullptr should also be safe
    llama_lazyllm_free(lctx);
    return true;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main() {
    report("params_default_constructor",    test_params_default_constructor());
    report("top_k_indices_basic_selection", test_top_k_basic());
    report("top_k_indices_last_always_kept",test_top_k_last_always_kept());
    report("top_k_indices_keep_all",        test_top_k_keep_all());
    report("apply_pooling_identity",        test_pooling_identity());
    report("apply_pooling_smooths",         test_pooling_smooths());
    report("aux_cache_store_retrieve",      test_aux_cache_store_retrieve());
    report("aux_cache_pruning_point",       test_aux_cache_pruning_point());
    report("aux_cache_reset",               test_aux_cache_reset());
    report("init_null_ctx_returns_nullptr", test_init_null_ctx());

    std::printf("\n%d/%d tests passed\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
