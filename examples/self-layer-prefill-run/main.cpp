#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <chrono>
#include <cmath>
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
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static void usage() {
    std::cerr
        << "llama-self-layer-prefill-run\n"
        << "  --model <gguf>           (required)\n"
        << "  --mode bench|once        (default: bench)\n"
        << "  --n-ctx <n>              (default 8192; must be >= prompt len)\n"
        << "  --n-batch <n>           (default 8192; must be >= prompt len for Q/K extract)\n"
        << "  --n-gpu-layers <n>     (0=CPU-only, -1=all layers GPU; default -1)\n"
        << "  --n-early <n>           (shallow layer count; default L/4)\n"
        << "  --keep-ratio <f>        (for end-to-end prefill; bench sweeps several)\n"
        << "  --pool <n>              (pool kernel, default 13)\n"
        << "  --chunking              (chunk-based filter)\n"
        << "  --chunk-size <n>        (default 32)\n"
        << "  --n-warmup <n>          (default 1)\n"
        << "  --n-repeat <n>          (default 3)\n";
}

static std::vector<llama_token> make_prompt(const llama_vocab * vocab, int n_tok) {
    std::vector<llama_token> toks((size_t) n_tok);
    for (int i = 0; i < n_tok; i++) {
        toks[(size_t) i] = 1000 + (i % 256);
    }
    (void) vocab;
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
        } else {
            usage();
            return 1;
        }
    }
    if (args.model_path.empty()) {
        usage();
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    // Single-device graphs: required for stable ggml_backend_tensor_get of Qcur/Kcur.
    if (args.n_gpu_layers != 0) {
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    }
    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = (int) llama_model_n_layer(model);
    const int n_early = args.n_early > 0 ? args.n_early : std::max(1, n_layer / 4);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max(512, args.n_ctx);
    cparams.n_batch = (uint32_t) std::max(512, args.n_batch);
    cparams.n_ubatch = cparams.n_batch; // single-shot prefill: ubatch == batch so Q/K live in one graph
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create context\n";
        llama_model_free(model);
        return 1;
    }

    auto run_baseline = [&](const std::vector<llama_token> & toks) {
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
        return std::make_pair(r, t1 - t0);
    };

    auto run_profiles = [&](const std::vector<llama_token> & toks, llama_self_layer_prefill_params & P, float * p_sf, float * p_sd) {
        std::vector<float> shallow, deep, full;
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int r = llama_self_layer_prefill_attention_profiles(
            ctx, toks.data(), (int) toks.size(), &P, shallow, deep, full, p_sf, p_sd);
        llama_synchronize(ctx);
        double t1 = now_ms();
        return std::make_pair(r, t1 - t0);
    };

    auto run_e2e = [&](const std::vector<llama_token> & toks, llama_self_layer_prefill_params & P) {
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int n_kept = llama_self_layer_prefill(ctx, toks.data(), (int) toks.size(), &P);
        llama_synchronize(ctx);
        double t1 = now_ms();
        return std::make_pair(n_kept, t1 - t0);
    };

    auto run_kvprune = [&](const std::vector<llama_token> & toks, llama_self_layer_prefill_params & P) {
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int n_kept = llama_self_layer_prefill_with_kv_prune(ctx, toks.data(), (int) toks.size(), &P);
        llama_synchronize(ctx);
        double t1 = now_ms();
        return std::make_pair(n_kept, t1 - t0);
    };

    auto run_partial = [&](const std::vector<llama_token> & toks, llama_self_layer_prefill_params & P) {
        llama_synchronize(ctx);
        double t0 = now_ms();
        const int n_kept = llama_self_layer_prefill_partial(ctx, toks.data(), (int) toks.size(), &P);
        llama_synchronize(ctx);
        double t1 = now_ms();
        return std::make_pair(n_kept, t1 - t0);
    };

    if (args.mode == "once") {
        const int n_prompt = std::min(512, (int) cparams.n_batch - 1);
        auto toks = make_prompt(vocab, n_prompt);
        llama_self_layer_prefill_params P;
        P.n_early_layers = n_early;
        P.keep_ratio = args.keep_ratio;
        P.pool_kernel_size = args.pool;
        P.use_chunking = args.chunking;
        P.chunk_size = args.chunk_size;

        float p_sf = 0, p_sd = 0;
        auto pr = run_profiles(toks, P, &p_sf, &p_sd);
        if (pr.first != 0) {
            std::cerr << "attention_profiles failed: " << pr.first << "\n";
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::cout << "{\"n_prompt\":" << n_prompt << ",\"n_layer\":" << n_layer
                  << ",\"n_early\":" << n_early << ",\"profile_ms\":" << std::fixed << std::setprecision(2) << pr.second
                  << ",\"pearson_shallow_vs_full\":" << p_sf << ",\"pearson_shallow_vs_deep\":" << p_sd << "}\n";

        auto e2e = run_e2e(toks, P);
        if (e2e.first < 0) {
            std::cerr << "self_layer_prefill failed\n";
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::cout << "{\"n_kept\":" << e2e.first << ",\"e2e_ms\":" << e2e.second << "}\n";
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    // bench: grid over context lengths and keep_ratio (prompt must fit in one ubatch graph for Q/K extract)
    const int n_ub = (int) llama_n_ubatch(ctx);
    std::vector<int> ctx_lens = {512};
    if (n_ub >= 2048) {
        ctx_lens.push_back(2048);
    }
    if (n_ub >= 4096) {
        ctx_lens.push_back(4096);
    }
    if (n_ub >= 8192) {
        ctx_lens.push_back(8192);
    }
    const std::vector<float> keep_ratios = {0.10f, 0.25f, 0.50f, 1.00f};

    std::cout << "{\n  \"model\": \"" << args.model_path << "\",\n";
    std::cout << "  \"n_layer\": " << n_layer << ",\n  \"n_early_default\": " << n_early << ",\n";
    std::cout << "  \"n_ctx\": " << cparams.n_ctx << ",\n  \"n_batch\": " << cparams.n_batch
              << ",\n  \"n_gpu_layers\": " << mparams.n_gpu_layers << ",\n";
    std::cout << "  \"runs\": " << args.n_repeat << ",\n  \"results\": [\n";

    bool first = true;
    for (int L : ctx_lens) {
        if (L > (int) cparams.n_batch) {
            continue;
        }
        auto toks = make_prompt(vocab, L);

        for (float kr : keep_ratios) {
            llama_self_layer_prefill_params P;
            P.n_early_layers = n_early;
            P.keep_ratio = kr;
            P.pool_kernel_size = args.pool;
            P.use_chunking = args.chunking;
            P.chunk_size = args.chunk_size;

            // warmup (include partial so first sched-reserve cost is paid here)
            for (int w = 0; w < args.n_warmup; w++) {
                float a = 0, b = 0;
                run_profiles(toks, P, &a, &b);
                run_baseline(toks);
                run_e2e(toks, P);
                run_kvprune(toks, P);
                run_partial(toks, P);
            }

            double sum_base = 0, sum_prof = 0, sum_e2e = 0, sum_kvp = 0, sum_part = 0;
            float sum_psf = 0, sum_psd = 0;
            int ok = 0;
            int last_n_kept = 0;
            int last_n_kept_kvp = 0;
            int last_n_kept_part = 0;
            for (int r = 0; r < args.n_repeat; r++) {
                float p_sf = 0, p_sd = 0;
                auto pr   = run_profiles(toks, P, &p_sf, &p_sd);
                auto bl   = run_baseline(toks);
                auto e2   = run_e2e(toks, P);
                auto kvp  = run_kvprune(toks, P);
                auto part = run_partial(toks, P);
                if (pr.first == 0 && bl.first == 0 && e2.first >= 0 && kvp.first >= 0 && part.first >= 0) {
                    sum_prof += pr.second;
                    sum_base += bl.second;
                    sum_e2e  += e2.second;
                    sum_kvp  += kvp.second;
                    sum_part += part.second;
                    sum_psf  += p_sf;
                    sum_psd  += p_sd;
                    last_n_kept = e2.first;
                    last_n_kept_kvp = kvp.first;
                    last_n_kept_part = part.first;
                    ok++;
                }
            }
            if (!first) {
                std::cout << ",\n";
            }
            first = false;
            if (ok <= 0) {
                std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << kr << ",\"error\":\"failed\"}";
                continue;
            }
            std::cout << "    {\"n_prompt\":" << L << ",\"keep_ratio\":" << kr
                      << ",\"baseline_prefill_ms_avg\":" << std::fixed << std::setprecision(2) << (sum_base / ok)
                      << ",\"attention_profile_ms_avg\":" << (sum_prof / ok)
                      << ",\"e2e_self_layer_ms_avg\":" << (sum_e2e / ok)
                      << ",\"kv_prune_ms_avg\":" << (sum_kvp / ok)
                      << ",\"partial_self_layer_ms_avg\":" << (sum_part / ok)
                      << ",\"pearson_shallow_vs_full_avg\":" << std::setprecision(4) << (sum_psf / ok)
                      << ",\"pearson_shallow_vs_deep_avg\":" << (sum_psd / ok)
                      << ",\"n_kept_last_repeat\":" << last_n_kept
                      << ",\"n_kept_kvprune_last_repeat\":" << last_n_kept_kvp
                      << ",\"n_kept_partial_last_repeat\":" << last_n_kept_part << "}";
        }
    }
    std::cout << "\n  ]\n}\n";

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
