#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Args {
    std::string model_path;
    std::string mode = "bench"; // bench | once
    int n_ctx = 8192;
    int n_batch = 8192;
    int n_gpu_layers = -1; // 0 = CPU only, -1 = all on GPU when CUDA build
    int n_early = 0; // 0 = L/4
    int n_warmup = 1;
    int n_repeat = 3;
    float keep_ratio = 0.25f;
    int pool = 13;
    bool chunking = false;
    int chunk_size = 32;
    int bench_min_prompt = 0;
    bool multi_gpu = false;       // use multiple GPUs via layer split
    double timeout_sec = 300.0;   // max seconds per operation before skip
    bool verbose = true;          // print progress to stderr
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static void log_progress(bool verbose, const char * fmt, ...) {
    if (!verbose) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

static void usage() {
    std::cerr
        << "llama-self-layer-prefill-run\n"
        << "  --model <gguf>           (required)\n"
        << "  --mode bench|once        (default: bench)\n"
        << "  --n-ctx <n>              (default 8192)\n"
        << "  --n-batch <n>            (default 8192)\n"
        << "  --n-gpu-layers <n>       (0=CPU-only, -1=all GPU; default -1)\n"
        << "  --n-early <n>            (shallow layer count; default L/4)\n"
        << "  --keep-ratio <f>         (for end-to-end prefill; bench sweeps several)\n"
        << "  --pool <n>               (pool kernel, default 13)\n"
        << "  --chunking               (chunk-based filter)\n"
        << "  --chunk-size <n>         (default 32)\n"
        << "  --n-warmup <n>           (default 1)\n"
        << "  --n-repeat <n>           (default 3)\n"
        << "  --bench-min-prompt <n>   (bench only: skip n_prompt < n; default 0)\n"
        << "  --multi-gpu              (use layer split across GPUs; default: single GPU)\n"
        << "  --timeout <sec>          (max seconds per op before skip; default 300)\n"
        << "  --quiet                  (suppress progress to stderr)\n";
}

// More realistic prompt: repeated paragraph of varying content (like doc retrieval)
static std::vector<llama_token> make_realistic_prompt(const llama_vocab * vocab, int n_tok) {
    (void) vocab;
    std::vector<llama_token> toks;
    toks.reserve((size_t) n_tok);
    // Simulate document with headers, paragraphs, varying token IDs
    // Token IDs 1000-9999 to avoid special tokens
    int para_len = 128;
    int header_len = 8;
    int para_idx = 0;
    for (int i = 0; i < n_tok; ) {
        // Header tokens (low IDs)
        for (int h = 0; h < header_len && i < n_tok; h++, i++) {
            toks.push_back((llama_token)(1000 + (para_idx * 17 + h) % 500));
        }
        // Paragraph tokens (higher IDs, more variation)
        for (int p = 0; p < para_len && i < n_tok; p++, i++) {
            int base = 2000 + (para_idx % 10) * 500;
            toks.push_back((llama_token)(base + (p * 31 + para_idx * 7) % 500));
        }
        para_idx++;
    }
    return toks;
}

int main(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (a == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (a == "--n-ctx" && i + 1 < argc) {
            args.n_ctx = std::stoi(argv[++i]);
        } else if (a == "--n-batch" && i + 1 < argc) {
            args.n_batch = std::stoi(argv[++i]);
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            args.n_gpu_layers = std::stoi(argv[++i]);
        } else if (a == "--n-early" && i + 1 < argc) {
            args.n_early = std::stoi(argv[++i]);
        } else if (a == "--keep-ratio" && i + 1 < argc) {
            args.keep_ratio = std::stof(argv[++i]);
        } else if (a == "--pool" && i + 1 < argc) {
            args.pool = std::stoi(argv[++i]);
        } else if (a == "--chunking") {
            args.chunking = true;
        } else if (a == "--chunk-size" && i + 1 < argc) {
            args.chunk_size = std::stoi(argv[++i]);
        } else if (a == "--n-warmup" && i + 1 < argc) {
            args.n_warmup = std::stoi(argv[++i]);
        } else if (a == "--n-repeat" && i + 1 < argc) {
            args.n_repeat = std::stoi(argv[++i]);
        } else if (a == "--bench-min-prompt" && i + 1 < argc) {
            args.bench_min_prompt = std::stoi(argv[++i]);
        } else if (a == "--multi-gpu") {
            args.multi_gpu = true;
        } else if (a == "--timeout" && i + 1 < argc) {
            args.timeout_sec = std::stod(argv[++i]);
        } else if (a == "--quiet") {
            args.verbose = false;
        } else {
            usage();
            return 1;
        }
    }
    if (args.model_path.empty()) {
        usage();
        return 1;
    }

    log_progress(args.verbose, "[init] Loading model: %s\n", args.model_path.c_str());
    log_progress(args.verbose, "[init] n_ctx=%d, n_batch=%d, n_gpu_layers=%d, multi_gpu=%s\n",
                 args.n_ctx, args.n_batch, args.n_gpu_layers, args.multi_gpu ? "yes" : "no");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;

    // GPU split mode: NONE for single GPU (required for Q/K tensor readback stability),
    // LAYER for multi-GPU (distributes layers across devices)
    if (args.n_gpu_layers != 0) {
        if (args.multi_gpu) {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
            log_progress(args.verbose, "[init] Using LLAMA_SPLIT_MODE_LAYER (multi-GPU)\n");
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            log_progress(args.verbose, "[init] Using LLAMA_SPLIT_MODE_NONE (single GPU)\n");
        }
    }

    double t_load_start = now_ms();
    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    double t_load_end = now_ms();
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    log_progress(args.verbose, "[init] Model loaded in %.1f ms\n", t_load_end - t_load_start);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = (int) llama_model_n_layer(model);
    const int n_early = args.n_early > 0 ? args.n_early : std::max(1, n_layer / 4);
    log_progress(args.verbose, "[init] n_layer=%d, n_early=%d\n", n_layer, n_early);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max(512, args.n_ctx);
    cparams.n_batch = (uint32_t) std::max(512, args.n_batch);
    cparams.n_ubatch = cparams.n_batch;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    log_progress(args.verbose, "[init] Creating context (n_ctx=%u, n_batch=%u, FA=disabled)...\n",
                 cparams.n_ctx, cparams.n_batch);
    double t_ctx_start = now_ms();
    llama_context * ctx = llama_init_from_model(model, cparams);
    double t_ctx_end = now_ms();
    if (!ctx) {
        std::cerr << "Failed to create context (likely OOM during graph_reserve)\n";
        llama_model_free(model);
        return 1;
    }
    log_progress(args.verbose, "[init] Context created in %.1f ms\n", t_ctx_end - t_ctx_start);

    // Initialize self-layer prefill caching for the fixed n_early value
    // This pre-reserves the partial graph to avoid rebuild overhead on every call
    log_progress(args.verbose, "[init] Initializing self-layer prefill cache (n_early=%d)...\n", n_early);
    double t_slp_start = now_ms();
    if (llama_self_layer_prefill_init(ctx, n_early) != 0) {
        log_progress(args.verbose, "[init] Warning: failed to init SLP cache (will still work, but slower)\n");
    } else {
        double t_slp_end = now_ms();
        log_progress(args.verbose, "[init] SLP cache initialized in %.1f ms\n", t_slp_end - t_slp_start);
    }

    const double timeout_ms = args.timeout_sec * 1000.0;

    // Helper: run with timeout check (returns {result, elapsed_ms, timed_out})
    struct TimedResult {
        int result;
        double elapsed_ms;
        bool timed_out;
    };

    auto run_baseline = [&](const std::vector<llama_token> & toks) -> TimedResult {
        llama_memory_clear(llama_get_memory(ctx), false);
        const int n = (int) toks.size();
        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = 0;
        for (int i = 0; i < n; i++) {
            b.token[b.n_tokens] = toks[(size_t) i];
            b.pos[b.n_tokens] = i;
            b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = 0;
            b.logits[b.n_tokens] = (i == n - 1);
            b.n_tokens++;
        }
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int r = llama_decode(ctx, b);
        llama_synchronize(ctx);
        double t1 = now_ms();
        llama_batch_free(b);
        return {r, t1 - t0, false};
    };

    auto run_partial = [&](const std::vector<llama_token> & toks, llama_self_layer_prefill_params & P) -> TimedResult {
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int n_kept = llama_self_layer_prefill_partial(ctx, toks.data(), (int) toks.size(), &P);
        llama_synchronize(ctx);
        double t1 = now_ms();
        return {n_kept, t1 - t0, false};
    };

    if (args.mode == "once") {
        const int n_prompt = std::min(512, (int) cparams.n_batch - 1);
        auto toks = make_realistic_prompt(vocab, n_prompt);
        llama_self_layer_prefill_params P;
        P.n_early_layers = n_early;
        P.keep_ratio = args.keep_ratio;
        P.pool_kernel_size = args.pool;
        P.use_chunking = args.chunking;
        P.chunk_size = args.chunk_size;

        auto bl = run_baseline(toks);
        auto part = run_partial(toks, P);

        std::cout << "{\"n_prompt\":" << n_prompt
                  << ",\"baseline_ms\":" << std::fixed << std::setprecision(2) << bl.elapsed_ms
                  << ",\"partial_ms\":" << part.elapsed_ms
                  << ",\"n_kept\":" << part.result
                  << ",\"speedup\":" << std::setprecision(2) << (bl.elapsed_ms / part.elapsed_ms)
                  << "}\n";
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    // bench mode: grid over context lengths and keep_ratio
    const int n_ub = (int) llama_n_ubatch(ctx);
    std::vector<int> ctx_lens;
    // Build ladder: 512, 1k, 2k, 4k, 8k, 16k, 32k
    if (n_ub >= 512)   ctx_lens.push_back(512);
    if (n_ub >= 1024)  ctx_lens.push_back(1024);
    if (n_ub >= 2048)  ctx_lens.push_back(2048);
    if (n_ub >= 4096)  ctx_lens.push_back(4096);
    if (n_ub >= 8192)  ctx_lens.push_back(8192);
    if (n_ub >= 16384) ctx_lens.push_back(16384);
    if (n_ub >= 32768) ctx_lens.push_back(32768);

    const std::vector<float> keep_ratios = {0.10f, 0.25f, 0.50f, 1.00f};

    // Count total benchmarks for progress
    int total_benches = 0;
    for (int L : ctx_lens) {
        if (L < args.bench_min_prompt || L > (int) cparams.n_batch) continue;
        total_benches += (int) keep_ratios.size();
    }

    log_progress(args.verbose, "\n[bench] Starting grid: %d context lengths x %d keep_ratios = %d points\n",
                 (int) ctx_lens.size(), (int) keep_ratios.size(), total_benches);
    log_progress(args.verbose, "[bench] warmup=%d, repeat=%d, timeout=%.0fs\n\n",
                 args.n_warmup, args.n_repeat, args.timeout_sec);

    // JSON output header
    std::cout << "{\n  \"model\": \"" << args.model_path << "\",\n";
    std::cout << "  \"n_layer\": " << n_layer << ",\n  \"n_early\": " << n_early << ",\n";
    std::cout << "  \"n_ctx\": " << cparams.n_ctx << ",\n  \"n_batch\": " << cparams.n_batch << ",\n";
    std::cout << "  \"n_gpu_layers\": " << mparams.n_gpu_layers << ",\n";
    std::cout << "  \"multi_gpu\": " << (args.multi_gpu ? "true" : "false") << ",\n";
    std::cout << "  \"runs\": " << args.n_repeat << ",\n";
    std::cout << "  \"bench_min_prompt\": " << args.bench_min_prompt << ",\n  \"results\": [\n";

    bool first = true;
    int bench_idx = 0;

    for (int L : ctx_lens) {
        if (L < args.bench_min_prompt) continue;
        if (L > (int) cparams.n_batch) continue;

        log_progress(args.verbose, "=== n_prompt = %d ===\n", L);
        auto toks = make_realistic_prompt(vocab, L);

        for (float kr : keep_ratios) {
            bench_idx++;
            log_progress(args.verbose, "  [%d/%d] n=%d, kr=%.2f: ", bench_idx, total_benches, L, kr);

            llama_self_layer_prefill_params P;
            P.n_early_layers = n_early;
            P.keep_ratio = kr;
            P.pool_kernel_size = args.pool;
            P.use_chunking = args.chunking;
            P.chunk_size = args.chunk_size;

            // Warmup with timeout check
            bool warmup_timeout = false;
            for (int w = 0; w < args.n_warmup && !warmup_timeout; w++) {
                log_progress(args.verbose, "W");
                auto bl = run_baseline(toks);
                if (bl.elapsed_ms > timeout_ms) {
                    warmup_timeout = true;
                    break;
                }
                auto part = run_partial(toks, P);
                if (part.elapsed_ms > timeout_ms) {
                    warmup_timeout = true;
                    break;
                }
            }

            if (warmup_timeout) {
                log_progress(args.verbose, " TIMEOUT (warmup took >%.0fs), skipping\n", args.timeout_sec);
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << kr
                          << ",\"error\":\"timeout\"}";
                continue;
            }

            // Timed runs
            double sum_base = 0, sum_part = 0;
            int ok = 0;
            int last_n_kept = 0;
            bool run_timeout = false;

            for (int r = 0; r < args.n_repeat && !run_timeout; r++) {
                log_progress(args.verbose, ".");

                auto bl = run_baseline(toks);
                if (bl.elapsed_ms > timeout_ms) {
                    run_timeout = true;
                    break;
                }

                auto part = run_partial(toks, P);
                if (part.elapsed_ms > timeout_ms) {
                    run_timeout = true;
                    break;
                }

                if (bl.result == 0 && part.result >= 0) {
                    sum_base += bl.elapsed_ms;
                    sum_part += part.elapsed_ms;
                    last_n_kept = part.result;
                    ok++;
                }
            }

            if (run_timeout) {
                log_progress(args.verbose, " TIMEOUT (run took >%.0fs)\n", args.timeout_sec);
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << kr
                          << ",\"error\":\"timeout\"}";
                continue;
            }

            if (!first) std::cout << ",\n";
            first = false;

            if (ok <= 0) {
                log_progress(args.verbose, " FAILED\n");
                std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << kr
                          << ",\"error\":\"failed\"}";
                continue;
            }

            double base_avg = sum_base / ok;
            double part_avg = sum_part / ok;
            double speedup = base_avg / part_avg;

            log_progress(args.verbose, " base=%.0fms, partial=%.0fms, speedup=%.2fx, kept=%d\n",
                         base_avg, part_avg, speedup, last_n_kept);

            std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << std::fixed << std::setprecision(2) << kr
                      << ",\"baseline_ms\":" << base_avg
                      << ",\"partial_ms\":" << part_avg
                      << ",\"speedup\":" << speedup
                      << ",\"n_kept\":" << last_n_kept << "}";
        }
    }

    std::cout << "\n  ]\n}\n";

    log_progress(args.verbose, "\n[done] Benchmark complete.\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
