// tests/test-lazyllm-identity-kr1.cpp
//
// Verifies the identity invariant: when keep_ratio = 1.0 at every pruning
// stage, the full set of token indices {0 .. n_tokens-1} is kept in original
// order after each stage.
//
// This is a pure unit test — no model or GPU required.
//
// Failure modes caught:
//   - top_k_indices returns wrong indices at kr=1.0
//   - Multi-stage simulation loses or reorders tokens at kr=1.0
//   - Edge cases: single token, two tokens, last-token-worst-score

#include "llama-lazyllm.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

static void report(const char * name, bool ok) {
    if (ok) {
        std::printf("PASS: %s\n", name);
        g_pass++;
    } else {
        std::printf("FAIL: %s\n", name);
        g_fail++;
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Returns true when indices == {0, 1, ..., n-1} in order.
static bool is_identity(const std::vector<int32_t> & indices, int n) {
    if ((int)indices.size() != n) return false;
    for (int i = 0; i < n; i++) {
        if (indices[i] != i) return false;
    }
    return true;
}

// Simulate one LazyLLM stage: given the current set of live positions and a
// score per live-position, return the positions that survive top-K at kr.
// In our implementation the last live position is always kept.
static std::vector<int32_t> simulate_stage(
        const std::vector<int32_t> & live_positions,
        const std::vector<float>   & scores,   // one score per live position
        float                        kr)
{
    assert(live_positions.size() == scores.size());
    int n = (int)live_positions.size();

    // Use the public API: top_k_indices works on scores indexed 0..n-1.
    std::vector<int32_t> local_kept = llama_lazyllm_top_k_indices(scores, kr, n);

    // Map local indices back to original positions.
    std::vector<int32_t> kept_positions;
    kept_positions.reserve(local_kept.size());
    for (int32_t li : local_kept) {
        kept_positions.push_back(live_positions[(size_t)li]);
    }
    return kept_positions;
}

// ── individual tests ──────────────────────────────────────────────────────────

static bool test_single_stage_kr1() {
    // Any score distribution at kr=1.0 must keep all tokens.
    std::vector<float> scores = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f, 0.7f, 0.4f};
    int n = (int)scores.size();

    std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, 1.0f, n);

    if (!is_identity(kept, n)) {
        std::fprintf(stderr, "  kr=1.0 single stage: expected identity, got size=%d\n",
                     (int)kept.size());
        return false;
    }
    return true;
}

static bool test_three_stage_kr1() {
    // Simulate a 3-stage pruning with kr=1.0 at every stage.
    // No matter what scores are, all tokens must survive every stage.
    const int n = 20;
    std::vector<int32_t> live(n);
    std::iota(live.begin(), live.end(), 0);  // {0,1,...,19}

    // Deliberately adversarial scores: assign lowest to last token each stage.
    for (int stage = 0; stage < 3; stage++) {
        int m = (int)live.size();
        std::vector<float> scores(m);
        for (int i = 0; i < m; i++) scores[i] = (float)(m - 1 - i);  // descending
        scores[m-1] = -1.0f;  // last has worst score — must still survive

        live = simulate_stage(live, scores, 1.0f);

        if ((int)live.size() != n) {
            std::fprintf(stderr, "  stage %d: expected %d tokens, got %d\n",
                         stage, n, (int)live.size());
            return false;
        }
        for (int i = 0; i < n; i++) {
            if (live[i] != i) {
                std::fprintf(stderr, "  stage %d: live[%d] = %d, expected %d\n",
                             stage, i, live[i], i);
                return false;
            }
        }
    }
    return true;
}

static bool test_single_token_kr1() {
    // n=1 at any kr must keep the single token (it is also the last token).
    std::vector<float> scores = {42.0f};
    std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, 1.0f, 1);
    if (kept.size() != 1 || kept[0] != 0) {
        std::fprintf(stderr, "  single token kr=1.0 failed\n");
        return false;
    }

    kept = llama_lazyllm_top_k_indices(scores, 0.1f, 1);
    if (kept.size() != 1 || kept[0] != 0) {
        std::fprintf(stderr, "  single token kr=0.1 failed\n");
        return false;
    }
    return true;
}

static bool test_two_tokens_kr1() {
    std::vector<float> scores = {0.9f, 0.1f};
    std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, 1.0f, 2);
    if ((int)kept.size() != 2 || kept[0] != 0 || kept[1] != 1) {
        std::fprintf(stderr, "  two tokens kr=1.0 failed: size=%d\n", (int)kept.size());
        return false;
    }
    return true;
}

static bool test_positions_preserved_at_kr1() {
    // After simulate_stage at kr=1.0 starting from non-zero positions,
    // all original positions must be returned unchanged.
    std::vector<int32_t> live = {5, 10, 15, 20, 25};  // sparse original positions
    std::vector<float>   scores = {0.1f, 0.5f, 0.9f, 0.2f, 0.3f};

    std::vector<int32_t> result = simulate_stage(live, scores, 1.0f);

    if (result != live) {
        std::fprintf(stderr, "  positions not preserved at kr=1.0\n");
        return false;
    }
    return true;
}

static bool test_uniform_scores_kr1() {
    // All scores equal, kr=1.0 — must still keep all tokens in order.
    const int n = 16;
    std::vector<float> scores(n, 1.0f);
    std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, 1.0f, n);

    if (!is_identity(kept, n)) {
        std::fprintf(stderr, "  uniform scores kr=1.0: not identity, size=%d\n",
                     (int)kept.size());
        return false;
    }
    return true;
}

static bool test_paper_schedule_kr1() {
    // Paper default schedule: 3 stages at layers {8,16,24} with kr={0.7,0.5,0.3}.
    // When overridden to kr={1.0,1.0,1.0}, all 100 tokens must survive.
    const int n = 100;
    std::vector<int32_t> live(n);
    std::iota(live.begin(), live.end(), 0);

    std::vector<float> keep_ratios_identity = {1.0f, 1.0f, 1.0f};

    for (float kr : keep_ratios_identity) {
        int m = (int)live.size();
        std::vector<float> scores(m);
        // Random-ish scores: sine wave
        for (int i = 0; i < m; i++) scores[i] = std::sin((float)i * 0.3f);

        live = simulate_stage(live, scores, kr);

        if ((int)live.size() != n) {
            std::fprintf(stderr, "  paper schedule identity: lost tokens, have %d of %d\n",
                         (int)live.size(), n);
            return false;
        }
    }
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    report("single_stage_kr1_identity",          test_single_stage_kr1());
    report("three_stage_kr1_identity",           test_three_stage_kr1());
    report("single_token_kr1",                   test_single_token_kr1());
    report("two_tokens_kr1",                     test_two_tokens_kr1());
    report("positions_preserved_at_kr1",         test_positions_preserved_at_kr1());
    report("uniform_scores_kr1_identity",        test_uniform_scores_kr1());
    report("paper_schedule_all_kr1_identity",    test_paper_schedule_kr1());

    std::printf("\n%d/%d tests passed\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
