// tests/test-lazyllm-rope-positions.cpp
//
// Verifies that LazyLLM's token selection preserves original RoPE positions:
//   - Kept token indices are always a subset of the original position set
//   - Positions remain in strictly ascending order after selection
//   - The last original position is always present
//   - Across multi-stage pruning, positions accumulate only drops, never shifts
//
// No model or GPU required — pure algorithmic unit test.

#include "llama-lazyllm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
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

// Simulate a pruning stage: map local kept indices back to original positions.
static std::vector<int32_t> keep_positions(
        const std::vector<int32_t> & live_positions,
        const std::vector<float>   & scores,
        float                        kr)
{
    int n = (int)live_positions.size();
    std::vector<int32_t> local_kept = llama_lazyllm_top_k_indices(scores, kr, n);
    std::vector<int32_t> result;
    result.reserve(local_kept.size());
    for (int32_t li : local_kept) result.push_back(live_positions[(size_t)li]);
    return result;
}

// ── tests ──────────────────────────────────────────────────────────────────────

static bool test_kept_are_subset_of_original() {
    // After one pruning stage, every kept position must appear in the original set.
    std::vector<int32_t> original(32);
    std::iota(original.begin(), original.end(), 0);  // positions 0..31

    std::vector<float> scores(32);
    for (int i = 0; i < 32; i++) scores[i] = std::cos((float)i);

    std::vector<int32_t> kept = keep_positions(original, scores, 0.5f);

    for (int32_t pos : kept) {
        if (std::find(original.begin(), original.end(), pos) == original.end()) {
            std::fprintf(stderr, "  position %d not in original set\n", pos);
            return false;
        }
    }
    return true;
}

static bool test_positions_ascending_after_pruning() {
    // Kept positions must be in strictly ascending order (original document order).
    std::vector<int32_t> original(50);
    std::iota(original.begin(), original.end(), 100);  // positions 100..149

    std::vector<float> scores(50);
    for (int i = 0; i < 50; i++) scores[i] = std::sin((float)i * 0.7f);

    std::vector<int32_t> kept = keep_positions(original, scores, 0.4f);

    for (int i = 1; i < (int)kept.size(); i++) {
        if (kept[i] <= kept[i-1]) {
            std::fprintf(stderr, "  positions not ascending: kept[%d]=%d kept[%d]=%d\n",
                         i-1, kept[i-1], i, kept[i]);
            return false;
        }
    }
    return true;
}

static bool test_last_position_always_kept() {
    // The last position in the live set must survive regardless of its score.
    std::vector<int32_t> original = {0, 5, 10, 15, 20, 25, 30};
    const int32_t last_pos = original.back();

    // Give last position the worst possible score.
    std::vector<float> scores = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, -99.0f};

    for (float kr : {0.1f, 0.2f, 0.3f, 0.5f, 0.7f}) {
        std::vector<int32_t> kept = keep_positions(original, scores, kr);

        bool found = std::find(kept.begin(), kept.end(), last_pos) != kept.end();
        if (!found) {
            std::fprintf(stderr, "  last position %d not kept at kr=%.1f\n", last_pos, kr);
            return false;
        }
    }
    return true;
}

static bool test_multistage_no_position_shift() {
    // Across 3 pruning stages, positions must only be dropped, never replaced
    // with a different value (no sequential re-indexing).
    const int n = 64;
    std::vector<int32_t> live(n);
    std::iota(live.begin(), live.end(), 0);
    const std::vector<int32_t> original = live;

    std::vector<float> krs = {0.7f, 0.5f, 0.3f};  // paper schedule

    for (float kr : krs) {
        int m = (int)live.size();
        std::vector<float> scores(m);
        for (int i = 0; i < m; i++) scores[i] = (float)i / m;  // ascending

        std::vector<int32_t> next = keep_positions(live, scores, kr);

        // Every surviving position must come from the original set.
        for (int32_t pos : next) {
            if (std::find(original.begin(), original.end(), pos) == original.end()) {
                std::fprintf(stderr, "  ghost position %d not in original set\n", pos);
                return false;
            }
        }

        // Positions must still be ascending.
        for (int i = 1; i < (int)next.size(); i++) {
            if (next[i] <= next[i-1]) {
                std::fprintf(stderr, "  positions out of order after pruning stage\n");
                return false;
            }
        }

        live = next;
    }
    return true;
}

static bool test_sparse_original_positions() {
    // Realistic scenario: original positions are non-contiguous (e.g., after
    // middle-truncation, positions jump from 512 to 3584).
    std::vector<int32_t> original;
    for (int i = 0; i < 512; i++) original.push_back(i);       // head 0..511
    for (int i = 3584; i < 4096; i++) original.push_back(i);   // tail 3584..4095
    const int32_t last_pos = original.back();

    std::vector<float> scores(original.size());
    for (int i = 0; i < (int)scores.size(); i++) scores[i] = std::sin((float)i);

    std::vector<int32_t> kept = keep_positions(original, scores, 0.5f);

    // All kept positions must be from the original (no interpolation).
    for (int32_t pos : kept) {
        if (std::find(original.begin(), original.end(), pos) == original.end()) {
            std::fprintf(stderr, "  kept position %d not in sparse original set\n", pos);
            return false;
        }
    }

    // Ascending order.
    for (int i = 1; i < (int)kept.size(); i++) {
        if (kept[i] <= kept[i-1]) {
            std::fprintf(stderr, "  sparse positions not ascending\n");
            return false;
        }
    }

    // Last position kept.
    if (std::find(kept.begin(), kept.end(), last_pos) == kept.end()) {
        std::fprintf(stderr, "  last position %d not kept in sparse scenario\n", last_pos);
        return false;
    }

    return true;
}

static bool test_no_duplicates_in_kept() {
    // top_k_indices must never return the same index twice.
    const int n = 32;
    std::vector<float> scores(n, 1.0f);  // uniform — all scores equal
    for (float kr : {0.25f, 0.5f, 0.75f, 1.0f}) {
        std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, kr, n);
        std::vector<int32_t> sorted_kept = kept;
        std::sort(sorted_kept.begin(), sorted_kept.end());
        for (int i = 1; i < (int)sorted_kept.size(); i++) {
            if (sorted_kept[i] == sorted_kept[i-1]) {
                std::fprintf(stderr, "  duplicate index %d at kr=%.2f\n",
                             sorted_kept[i], kr);
                return false;
            }
        }
    }
    return true;
}

static bool test_minimum_one_token_kept() {
    // Even at kr approaching 0, at least the last token must be kept.
    for (int n : {1, 2, 5, 10, 100}) {
        std::vector<float> scores(n);
        for (int i = 0; i < n; i++) scores[i] = (float)i;

        // Very small ratio (rounds to 0 before ceiling → still 1)
        std::vector<int32_t> kept = llama_lazyllm_top_k_indices(scores, 0.001f, n);
        if (kept.empty()) {
            std::fprintf(stderr, "  n=%d kr=0.001 returned empty set\n", n);
            return false;
        }
        // Last token must be in the result
        if (std::find(kept.begin(), kept.end(), n-1) == kept.end()) {
            std::fprintf(stderr, "  n=%d kr=0.001: last token %d not kept\n", n, n-1);
            return false;
        }
    }
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    report("kept_are_subset_of_original",    test_kept_are_subset_of_original());
    report("positions_ascending_after_prune",test_positions_ascending_after_pruning());
    report("last_position_always_kept",       test_last_position_always_kept());
    report("multistage_no_position_shift",    test_multistage_no_position_shift());
    report("sparse_original_positions",       test_sparse_original_positions());
    report("no_duplicates_in_kept",           test_no_duplicates_in_kept());
    report("minimum_one_token_kept",          test_minimum_one_token_kept());

    std::printf("\n%d/%d tests passed\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
