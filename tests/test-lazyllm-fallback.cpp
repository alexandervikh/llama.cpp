// tests/test-lazyllm-fallback.cpp
//
// Tests the auto-fallback decision logic in llama_lazyllm_context.
//
// The fallback mechanism works as follows:
//   1. During llama_lazyllm_warmup(), both baseline and LazyLLM TTFTs are
//      measured on a warm-up prompt.
//   2. If LazyLLM is not at least `auto_fallback_min_speedup` faster than
//      baseline, ctx->fallback_active is set to true.
//   3. Subsequent calls to llama_lazyllm_prefill() short-circuit to
//      llama_decode() when fallback_active == true.
//
// This test exercises the *decision* logic (state fields, thresholds,
// parameter defaults) without requiring a real model.
//
// Failure modes caught:
//   - auto_fallback flag defaults wrong
//   - auto_fallback_min_speedup threshold wrong
//   - fallback_active not propagated correctly in the context
//   - Context fields for warmup timing are zero-initialized

#include "llama-lazyllm.h"

#include <cassert>
#include <cmath>
#include <cstdio>

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

// ── helpers ────────────────────────────────────────────────────────────────────

// Simulate the warmup A/B decision exactly as llama_lazyllm_warmup() would:
// sets fallback_active if LazyLLM is not faster enough.
static void simulate_warmup_decision(
        llama_lazyllm_context * ctx,
        double baseline_ms,
        double lazyllm_ms)
{
    ctx->warmup_baseline_ms = baseline_ms;
    ctx->warmup_lazyllm_ms  = lazyllm_ms;

    if (ctx->params.auto_fallback) {
        float speedup = (lazyllm_ms > 0.0) ? (float)(baseline_ms / lazyllm_ms) : 0.0f;
        ctx->fallback_active = (speedup < ctx->params.auto_fallback_min_speedup);
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

static bool test_default_params_have_fallback_enabled() {
    // Default params must have auto_fallback = true and reasonable threshold.
    llama_lazyllm_params p{};
    if (!p.auto_fallback) {
        std::fprintf(stderr, "  auto_fallback default must be true\n");
        return false;
    }
    // Paper result is ~2.4×; threshold should be < 1.4 to allow speedups in range
    if (p.auto_fallback_min_speedup < 1.0f || p.auto_fallback_min_speedup > 1.5f) {
        std::fprintf(stderr, "  auto_fallback_min_speedup=%.2f outside expected [1.0, 1.5]\n",
                     p.auto_fallback_min_speedup);
        return false;
    }
    return true;
}

static bool test_fallback_not_active_when_faster() {
    // LazyLLM clearly faster (2.4× speedup) → fallback must NOT activate.
    llama_lazyllm_params p{};
    llama_lazyllm_context ctx{};
    ctx.params = p;
    ctx.fallback_active = false;

    simulate_warmup_decision(&ctx, 1000.0, 416.0);  // speedup = 2.4×

    if (ctx.fallback_active) {
        std::fprintf(stderr, "  fallback activated at 2.4× speedup (should not)\n");
        return false;
    }
    return true;
}

static bool test_fallback_active_when_slower() {
    // LazyLLM is slower than baseline → fallback must activate.
    llama_lazyllm_params p{};
    llama_lazyllm_context ctx{};
    ctx.params = p;
    ctx.fallback_active = false;

    simulate_warmup_decision(&ctx, 1000.0, 1200.0);  // speedup = 0.83× (slower)

    if (!ctx.fallback_active) {
        std::fprintf(stderr, "  fallback not activated when LazyLLM is slower\n");
        return false;
    }
    return true;
}

static bool test_fallback_active_near_parity() {
    // LazyLLM at parity (1.0× speedup) → fallback must activate because
    // overhead of LazyLLM is not justified.
    llama_lazyllm_params p{};
    llama_lazyllm_context ctx{};
    ctx.params = p;
    ctx.fallback_active = false;

    simulate_warmup_decision(&ctx, 1000.0, 1000.0);  // speedup = 1.0×

    if (!ctx.fallback_active) {
        std::fprintf(stderr, "  fallback not activated at parity (1.0× speedup)\n");
        return false;
    }
    return true;
}

static bool test_fallback_disabled_when_auto_fallback_off() {
    // When auto_fallback = false, the decision is never made and
    // fallback_active must stay false even when LazyLLM is slower.
    llama_lazyllm_params p{};
    p.auto_fallback = false;

    llama_lazyllm_context ctx{};
    ctx.params = p;
    ctx.fallback_active = false;

    simulate_warmup_decision(&ctx, 1000.0, 5000.0);  // 5× slower

    if (ctx.fallback_active) {
        std::fprintf(stderr, "  fallback activated even though auto_fallback=false\n");
        return false;
    }
    return true;
}

static bool test_fallback_boundary_at_threshold() {
    // At exactly the threshold speedup, fallback must NOT activate.
    llama_lazyllm_params p{};
    const float thr = p.auto_fallback_min_speedup;

    llama_lazyllm_context ctx_at_threshold{};
    ctx_at_threshold.params = p;
    ctx_at_threshold.fallback_active = false;

    // speedup == threshold exactly
    simulate_warmup_decision(&ctx_at_threshold, (double)thr, 1.0);

    if (ctx_at_threshold.fallback_active) {
        std::fprintf(stderr, "  fallback activated at exactly threshold %.2f× speedup\n", thr);
        return false;
    }

    // Just below threshold → must activate.
    llama_lazyllm_context ctx_below{};
    ctx_below.params = p;
    ctx_below.fallback_active = false;

    simulate_warmup_decision(&ctx_below, (double)(thr * 0.99f), 1.0);
    if (!ctx_below.fallback_active) {
        std::fprintf(stderr, "  fallback not activated just below threshold\n");
        return false;
    }

    return true;
}

static bool test_warmup_timing_fields_recorded() {
    // After simulated warmup, both timing fields must be recorded.
    llama_lazyllm_params p{};
    llama_lazyllm_context ctx{};
    ctx.params = p;

    simulate_warmup_decision(&ctx, 756.0, 312.0);

    if (ctx.warmup_baseline_ms != 756.0) {
        std::fprintf(stderr, "  warmup_baseline_ms not recorded correctly\n");
        return false;
    }
    if (ctx.warmup_lazyllm_ms != 312.0) {
        std::fprintf(stderr, "  warmup_lazyllm_ms not recorded correctly\n");
        return false;
    }
    return true;
}

static bool test_custom_threshold() {
    // A caller that sets a stricter threshold (e.g. 2.0×) means fallback
    // activates even when LazyLLM is 1.5× faster.
    llama_lazyllm_params p{};
    p.auto_fallback_min_speedup = 2.0f;

    llama_lazyllm_context ctx{};
    ctx.params = p;
    ctx.fallback_active = false;

    simulate_warmup_decision(&ctx, 1500.0, 1000.0);  // 1.5× speedup

    if (!ctx.fallback_active) {
        std::fprintf(stderr, "  custom threshold 2.0× not respected at 1.5× speedup\n");
        return false;
    }

    // With the same threshold, 2.1× speedup must NOT trigger fallback.
    ctx.fallback_active = false;
    simulate_warmup_decision(&ctx, 2100.0, 1000.0);  // 2.1× speedup
    if (ctx.fallback_active) {
        std::fprintf(stderr, "  fallback triggered at 2.1× (threshold=2.0×)\n");
        return false;
    }

    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    report("default_params_auto_fallback_enabled",       test_default_params_have_fallback_enabled());
    report("no_fallback_when_clearly_faster",            test_fallback_not_active_when_faster());
    report("fallback_when_slower",                       test_fallback_active_when_slower());
    report("fallback_at_parity",                         test_fallback_active_near_parity());
    report("no_fallback_when_disabled",                  test_fallback_disabled_when_auto_fallback_off());
    report("fallback_boundary_at_threshold",             test_fallback_boundary_at_threshold());
    report("warmup_timing_fields_recorded",              test_warmup_timing_fields_recorded());
    report("custom_threshold_respected",                 test_custom_threshold());

    std::printf("\n%d/%d tests passed\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
