/*
 * llama-lazyllm-run — CLI harness for the LazyLLM token-pruning POC.
 * arXiv:2407.14057
 *
 * Two modes:
 *
 * 1. Single-prompt TTFT benchmark (original mode):
 *   llama-lazyllm-run --model <path.gguf> --prompt <text|@file> [opts]
 *
 * 2. Multi-prompt quality+TTFT benchmark (paper-reproduction mode):
 *   llama-lazyllm-run --model <path.gguf> --prompts-file <path.jsonl> \
 *     --out-csv results.csv --max-tokens 64 [opts]
 *
 *   Each JSONL line: {"prompt":"...", "answers":["ans1","ans2"], "id":"..."}
 *   (Compatible with LongBench export format from scripts/get-longbench.py)
 *
 * Options:
 *   --pruning-layers 8 16 24     space-separated ints  (default: 8 16 24)
 *   --keep-ratios 0.7 0.5 0.3   space-separated floats (default: 0.7 0.5 0.3)
 *   --pool-size 13               (default: 13)
 *   --repeat N                   timed repeats per prompt (default: 3)
 *   --n-ctx 4096                 KV cache size (default: 4096)
 *   --max-tokens N               tokens to generate per prompt (default: 64)
 *   --out-csv <path>             CSV output path
 *   --decode-pruning             enable Phase 5 decode KV pruning
 *   --decode-keep-ratio 0.7      (default: 0.7)
 */

#include "llama.h"
#include "llama-lazyllm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

/* ── minimal JSON helpers ─────────────────────────────────────────────────── */

// Extract the string value of a JSON key from a flat JSON object line.
// Handles simple string values and arrays of strings.
static std::string json_get_str(const std::string & line, const std::string & key) {
    const std::string pat = "\"" + key + "\":";
    auto pos = line.find(pat);
    if (pos == std::string::npos) return {};
    pos += pat.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return {};
    if (line[pos] != '"') return {};
    pos++;
    std::string val;
    while (pos < line.size() && line[pos] != '"') {
        if (line[pos] == '\\' && pos + 1 < line.size()) { pos++; }
        val += line[pos++];
    }
    return val;
}

static std::vector<std::string> json_get_str_array(const std::string & line, const std::string & key) {
    const std::string pat = "\"" + key + "\":";
    auto pos = line.find(pat);
    if (pos == std::string::npos) return {};
    pos += pat.size();
    while (pos < line.size() && line[pos] != '[') pos++;
    if (pos >= line.size()) return {};
    pos++;
    std::vector<std::string> result;
    while (pos < line.size() && line[pos] != ']') {
        while (pos < line.size() && line[pos] != '"' && line[pos] != ']') pos++;
        if (pos >= line.size() || line[pos] == ']') break;
        pos++;
        std::string val;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\' && pos + 1 < line.size()) { pos++; }
            val += line[pos++];
        }
        if (!val.empty()) result.push_back(val);
        pos++;
    }
    return result;
}

/* ── arg parsing ──────────────────────────────────────────────────────────── */

struct Args {
    std::string model_path;
    std::string prompt;           // single-prompt mode
    std::string prompts_file;     // multi-prompt mode (JSONL)
    std::vector<int>   pruning_layers = {8, 16, 24};
    std::vector<float> keep_ratios    = {0.7f, 0.5f, 0.3f};
    int      pool_size        = 13;
    int      repeat           = 3;
    int      max_tokens       = 64;
    uint32_t n_ctx            = 4096;
    int      n_gpu_layers     = 0;    // layers to offload to GPU (0 = CPU only)
    std::string out_csv;
    bool     decode_pruning   = false;
    float    decode_keep_ratio = 0.7f;
    int      n_prompts        = -1;   // max prompts to evaluate (-1 = all)
    bool     verbose          = false;
};

static void print_usage(const char * prog) {
    std::cout
        << "Usage (single prompt):\n"
        << "  " << prog << " --model <path.gguf> --prompt <text|@file> [opts]\n\n"
        << "Usage (multi-prompt LongBench):\n"
        << "  " << prog << " --model <path.gguf> --prompts-file <path.jsonl> [opts]\n\n"
        << "Options:\n"
        << "  --pruning-layers 8 16 24   (default: 8 16 24)\n"
        << "  --keep-ratios 0.7 0.5 0.3 (default: 0.7 0.5 0.3)\n"
        << "  --pool-size 13             (default: 13)\n"
        << "  --repeat N                 timed repeats per prompt (default: 3)\n"
        << "  --n-ctx 4096               KV cache size (default: 4096)\n"
        << "  --max-tokens N             generation tokens (default: 64)\n"
        << "  --n-prompts N              max prompts to process (default: all)\n"
        << "  --out-csv <path>           CSV output\n"
        << "  --n-gpu-layers N           layers to offload to GPU (default: 0)\n"
        << "  --decode-pruning           enable Phase 5 decode KV pruning\n"
        << "  --decode-keep-ratio 0.7    (default: 0.7)\n"
        << "  --verbose                  verbose logging\n";
}

static bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)            { a.model_path = argv[++i]; }
        else if (arg == "--prompt" && i + 1 < argc)      { a.prompt = argv[++i]; }
        else if (arg == "--prompts-file" && i + 1 < argc){ a.prompts_file = argv[++i]; }
        else if (arg == "--out-csv" && i + 1 < argc)     { a.out_csv = argv[++i]; }
        else if (arg == "--repeat" && i + 1 < argc)      { a.repeat = std::stoi(argv[++i]); }
        else if (arg == "--n-ctx" && i + 1 < argc)       { a.n_ctx = std::stoul(argv[++i]); }
        else if (arg == "--max-tokens" && i + 1 < argc)  { a.max_tokens = std::stoi(argv[++i]); }
        else if (arg == "--pool-size" && i + 1 < argc)   { a.pool_size = std::stoi(argv[++i]); }
        else if (arg == "--n-prompts" && i + 1 < argc)   { a.n_prompts = std::stoi(argv[++i]); }
        else if (arg == "--decode-keep-ratio" && i+1<argc){ a.decode_keep_ratio = std::stof(argv[++i]); }
        else if (arg == "--n-gpu-layers" && i+1<argc)    { a.n_gpu_layers = std::stoi(argv[++i]); }
        else if (arg == "--decode-pruning")               { a.decode_pruning = true; }
        else if (arg == "--verbose")                      { a.verbose = true; }
        else if (arg == "--pruning-layers") {
            a.pruning_layers.clear();
            while (i + 1 < argc && argv[i+1][0] != '-') a.pruning_layers.push_back(std::stoi(argv[++i]));
        } else if (arg == "--keep-ratios") {
            a.keep_ratios.clear();
            while (i + 1 < argc && argv[i+1][0] != '-') a.keep_ratios.push_back(std::stof(argv[++i]));
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return false;
        }
    }
    const bool has_prompt  = !a.prompt.empty();
    const bool has_prompts = !a.prompts_file.empty();
    if (!has_prompt && !has_prompts) return false;
    if (a.model_path.empty()) return false;
    return true;
}

static std::string resolve_prompt(const std::string & raw) {
    if (raw.size() > 1 && raw[0] == '@') {
        std::ifstream f(raw.substr(1));
        if (!f) { std::cerr << "Cannot open: " << raw.substr(1) << "\n"; std::exit(1); }
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
    return raw;
}

/* ── timing ───────────────────────────────────────────────────────────────── */

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static double elapsed_ms(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 == 0 ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2]);
}

static double iqr(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n < 4 ? 0.0 : v[3*n/4] - v[n/4];
}

/* ── greedy sampler ───────────────────────────────────────────────────────── */

static llama_token sample_greedy(llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits(ctx);
    const int n_vocab    = llama_vocab_n_tokens(vocab);
    if (!logits || n_vocab <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
    return (llama_token)best;
}

/* ── generate N tokens after prefill, return generated text ─────────────── */

static std::string generate_tokens(
        llama_context * ctx,
        llama_lazyllm_context * lz_ctx,  // null = baseline
        const llama_vocab * vocab,
        llama_token first_tok,
        llama_pos start_pos,
        int n_gen,
        llama_seq_id seq_id = 0) {
    std::string out;
    llama_token tok = first_tok;
    for (int t = 0; t < n_gen && tok >= 0; t++) {
        // decode tok to text
        char buf[256] = {};
        int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf)-1, 0, false);
        if (len > 0) { buf[len] = 0; out += buf; }

        // check for EOS
        if (llama_vocab_is_eog(vocab, tok)) break;

        // next decode step
        llama_batch dec = llama_batch_init(1, 0, 1);
        dec.n_tokens     = 1;
        dec.token[0]     = tok;
        dec.pos[0]       = start_pos + (llama_pos)t;
        dec.n_seq_id[0]  = 1;
        dec.seq_id[0][0] = seq_id;
        dec.logits[0]    = 1;

        int r;
        if (lz_ctx && lz_ctx->decode_pruning_enabled)
            r = llama_lazyllm_decode_step(lz_ctx, dec) >= 0 ? 0 : -1;
        else
            r = llama_decode(ctx, dec);

        llama_batch_free(dec);
        if (r != 0) break;
        tok = sample_greedy(ctx, vocab);
    }
    return out;
}

/* ── CSV escape ───────────────────────────────────────────────────────────── */

static std::string csv_escape(const std::string & s) {
    std::string r = "\"";
    for (char c : s) { if (c == '"') r += "\"\""; else r += c; }
    r += "\"";
    return r;
}

/* ── per-prompt benchmark (baseline + lazyllm) ────────────────────────────── */

struct PromptResult {
    std::string id;
    int         n_prompt = 0;
    double      baseline_ttft_ms   = 0;
    double      lazyllm_ttft_ms    = 0;
    double      speedup            = 0;
    int         n_kept             = 0;
    std::string baseline_output;
    std::string lazyllm_output;
    std::vector<std::string> answers;   // ground-truth answers
};

static PromptResult bench_prompt(
        llama_context * ctx,
        const llama_vocab * vocab,
        const llama_lazyllm_params & lz_params,
        const std::vector<llama_token> & tokens,
        int max_tokens,
        bool decode_pruning,
        float decode_keep_ratio,
        int repeat,
        const std::string & id,
        const std::vector<std::string> & answers,
        bool verbose) {
    PromptResult res;
    res.id      = id;
    res.n_prompt = (int)tokens.size();
    res.answers  = answers;

    /* ── baseline ── */
    std::vector<double> bl_times;
    for (int r = 0; r < repeat; r++) {
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch b = llama_batch_get_one(
            const_cast<llama_token*>(tokens.data()), (int32_t)tokens.size());
        auto t0 = Clock::now();
        if (llama_decode(ctx, b) != 0) continue;
        bl_times.push_back(elapsed_ms(t0, Clock::now()));
    }
    res.baseline_ttft_ms = median(bl_times);

    // generate baseline output (one pass, no timing pressure)
    if (max_tokens > 0) {
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch b = llama_batch_get_one(
            const_cast<llama_token*>(tokens.data()), (int32_t)tokens.size());
        if (llama_decode(ctx, b) == 0) {
            llama_token first = sample_greedy(ctx, vocab);
            res.baseline_output = generate_tokens(ctx, nullptr, vocab, first,
                                                  (llama_pos)tokens.size(), max_tokens);
        }
    }

    /* ── LazyLLM ── */
    std::vector<double> lz_times;
    for (int r = 0; r < repeat; r++) {
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_lazyllm_context * lz_ctx = llama_lazyllm_init_with_params(ctx, lz_params);
        if (!lz_ctx) continue;
        if (decode_pruning) llama_lazyllm_decode_pruning_enable(lz_ctx, decode_keep_ratio, 0);

        llama_batch b = llama_batch_get_one(
            const_cast<llama_token*>(tokens.data()), (int32_t)tokens.size());
        auto t0 = Clock::now();
        int n_kept = llama_lazyllm_prefill(lz_ctx, b);
        auto t1 = Clock::now();

        if (n_kept >= 0) {
            lz_times.push_back(elapsed_ms(t0, t1));
            res.n_kept = n_kept;
        }
        llama_lazyllm_free(lz_ctx);
    }
    res.lazyllm_ttft_ms = median(lz_times);
    res.speedup = (res.lazyllm_ttft_ms > 0) ? res.baseline_ttft_ms / res.lazyllm_ttft_ms : 0;

    // Generate lazyllm output (one pass).
    //
    // After llama_lazyllm_prefill the final decode_partial has valid logits
    // for the last surviving token — sample the first generated token directly.
    // Then build a fresh full-model KV on the kept tokens (at sequential
    // positions 0..n_kept-1) so subsequent generation steps work correctly.
    // This matches the paper's quality evaluation: drop pruned tokens entirely,
    // run all layers on the remaining tokens.
    if (max_tokens > 0) {
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_lazyllm_context * lz_ctx = llama_lazyllm_init_with_params(ctx, lz_params);
        if (lz_ctx) {
            if (decode_pruning) llama_lazyllm_decode_pruning_enable(lz_ctx, decode_keep_ratio, 0);
            llama_batch b = llama_batch_get_one(
                const_cast<llama_token*>(tokens.data()), (int32_t)tokens.size());
            int n_kept = llama_lazyllm_prefill(lz_ctx, b);

            if (n_kept >= 0) {
                // Sample first token from lazyllm logits (final decode_partial).
                llama_token first = sample_greedy(ctx, vocab);

                // Build full-model KV on kept tokens for correct generation.
                const auto & kept_idx = lz_ctx->kept_indices;
                std::vector<llama_token> kept_tokens;
                kept_tokens.reserve(kept_idx.size());
                for (int32_t ki : kept_idx) {
                    if (ki >= 0 && ki < (int32_t)tokens.size())
                        kept_tokens.push_back(tokens[(size_t)ki]);
                }
                llama_memory_clear(llama_get_memory(ctx), false);
                llama_batch kb = llama_batch_get_one(
                    kept_tokens.data(), (int32_t)kept_tokens.size());
                if (llama_decode(ctx, kb) == 0) {
                    res.lazyllm_output = generate_tokens(ctx, nullptr, vocab, first,
                                                         (llama_pos)kept_tokens.size(), max_tokens);
                }
            }
            llama_lazyllm_free(lz_ctx);
        }
    }

    if (verbose) {
        std::cerr << "  [" << id << "] n=" << res.n_prompt
                  << " baseline=" << std::fixed << std::setprecision(1) << res.baseline_ttft_ms
                  << "ms lazyllm=" << res.lazyllm_ttft_ms
                  << "ms speedup=" << std::setprecision(2) << res.speedup << "x"
                  << " kept=" << res.n_kept << "\n";
        if (!res.baseline_output.empty())
            std::cerr << "    baseline: " << res.baseline_output.substr(0,80) << "\n";
        if (!res.lazyllm_output.empty())
            std::cerr << "    lazyllm:  " << res.lazyllm_output.substr(0,80) << "\n";
    }

    return res;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return 1; }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mp);
    if (!model) { std::cerr << "Failed to load model: " << args.model_path << "\n"; return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx              = args.n_ctx;
    cp.n_batch            = args.n_ctx;
    cp.n_ubatch           = args.n_ctx;
    cp.flash_attn_type    = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // required for kq_soft_max extraction

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::cerr << "Failed to create context\n"; llama_model_free(model); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_lazyllm_params lz_params;
    lz_params.pruning_layers   = args.pruning_layers;
    lz_params.keep_ratios      = args.keep_ratios;
    lz_params.pool_kernel_size = args.pool_size;
    lz_params.verbose          = args.verbose;

    /* ─────────────────────────────── multi-prompt mode ─────────────────── */
    if (!args.prompts_file.empty()) {
        std::ifstream pf(args.prompts_file);
        if (!pf) { std::cerr << "Cannot open: " << args.prompts_file << "\n"; return 1; }

        std::ofstream csv;
        if (!args.out_csv.empty()) {
            csv.open(args.out_csv);
            if (csv) csv << "id,n_prompt,n_kept,baseline_ttft_ms,lazyllm_ttft_ms,speedup,"
                            "baseline_output,lazyllm_output,answers\n";
        }

        std::vector<double> all_baseline, all_lazyllm, all_speedup;
        int n_done = 0;

        // Print config
        std::cout << "Model: " << args.model_path << "\n";
        std::cout << "Config: layers=[";
        for (int i = 0; i < (int)args.pruning_layers.size(); i++) {
            if (i) std::cout << ","; std::cout << args.pruning_layers[i];
        }
        std::cout << "] ratios=[";
        for (int i = 0; i < (int)args.keep_ratios.size(); i++) {
            if (i) std::cout << ","; std::cout << args.keep_ratios[i];
        }
        std::cout << "] pool=" << args.pool_size
                  << " n_ctx=" << args.n_ctx
                  << " max_tokens=" << args.max_tokens << "\n\n";
        std::cout << std::left
                  << std::setw(6)  << "Idx"
                  << std::setw(7)  << "N_tok"
                  << std::setw(10) << "BL_ms"
                  << std::setw(10) << "LZ_ms"
                  << std::setw(8)  << "Spdup"
                  << std::setw(7)  << "Kept"
                  << "Baseline→LazyLLM output\n";
        std::cout << std::string(80, '-') << "\n";

        std::string line;
        while (std::getline(pf, line) && (args.n_prompts < 0 || n_done < args.n_prompts)) {
            if (line.empty() || line[0] == '#') continue;

            const std::string prompt_text = json_get_str(line, "prompt");
            if (prompt_text.empty()) continue;

            const std::string id = json_get_str(line, "id");
            std::vector<std::string> answers = json_get_str_array(line, "answers");

            // tokenize (with truncation to n_ctx)
            int n_prompt = -llama_tokenize(vocab, prompt_text.c_str(),
                                           (int32_t)prompt_text.size(),
                                           nullptr, 0, true, true);
            if (n_prompt <= 0) continue;
            std::vector<llama_token> tokens(n_prompt);
            llama_tokenize(vocab, prompt_text.c_str(), (int32_t)prompt_text.size(),
                           tokens.data(), n_prompt, true, true);
            // Truncate to n_ctx - max_tokens so generation tokens fit within KV.
            // Position n_prompt is where the first generated token goes; it must
            // be < n_ctx for the KV cache (positions 0..n_ctx-1 are valid).
            // LongBench puts the QUESTION at the END of the prompt, so we use
            // TAIL truncation: keep the last max_prompt tokens, preserving the
            // question. This discards the beginning of the context passages.
            const int max_prompt = (int)args.n_ctx - std::max(args.max_tokens, 1);
            if (n_prompt > max_prompt) {
                tokens.erase(tokens.begin(), tokens.begin() + (n_prompt - max_prompt));
                n_prompt = max_prompt;
            }

            PromptResult res = bench_prompt(ctx, vocab, lz_params, tokens,
                                            args.max_tokens,
                                            args.decode_pruning, args.decode_keep_ratio,
                                            args.repeat,
                                            id.empty() ? std::to_string(n_done) : id,
                                            answers, args.verbose);

            all_baseline.push_back(res.baseline_ttft_ms);
            all_lazyllm.push_back(res.lazyllm_ttft_ms);
            all_speedup.push_back(res.speedup);

            // compact output line
            const std::string bl_snip = res.baseline_output.size() > 25
                ? res.baseline_output.substr(0,25) + "…" : res.baseline_output;
            const std::string lz_snip = res.lazyllm_output.size() > 25
                ? res.lazyllm_output.substr(0,25) + "…" : res.lazyllm_output;

            std::cout << std::left
                      << std::setw(6)  << n_done
                      << std::setw(7)  << res.n_prompt
                      << std::fixed << std::setprecision(0)
                      << std::setw(10) << res.baseline_ttft_ms
                      << std::setw(10) << res.lazyllm_ttft_ms
                      << std::setprecision(2)
                      << std::setw(8)  << res.speedup
                      << std::setw(7)  << res.n_kept
                      << "\"" << bl_snip << "\" → \"" << lz_snip << "\"\n";

            if (csv.is_open()) {
                std::string ans_str;
                for (size_t i = 0; i < res.answers.size(); i++) {
                    if (i) ans_str += "|";
                    ans_str += res.answers[i];
                }
                csv << csv_escape(res.id) << ","
                    << res.n_prompt << ","
                    << res.n_kept << ","
                    << std::fixed << std::setprecision(2)
                    << res.baseline_ttft_ms << ","
                    << res.lazyllm_ttft_ms << ","
                    << std::setprecision(3) << res.speedup << ","
                    << csv_escape(res.baseline_output) << ","
                    << csv_escape(res.lazyllm_output) << ","
                    << csv_escape(ans_str) << "\n";
            }
            n_done++;
        }

        /* ── summary ── */
        if (!all_speedup.empty()) {
            const double med_bl = median(all_baseline);
            const double med_lz = median(all_lazyllm);
            const double med_sp = median(all_speedup);
            const double iqr_sp = iqr(all_speedup);

            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "Prompts evaluated   : " << n_done << "\n";
            std::cout << "Median baseline TTFT: " << std::fixed << std::setprecision(1) << med_bl << " ms\n";
            std::cout << "Median LazyLLM TTFT : " << med_lz << " ms\n";
            std::cout << "Median speedup      : " << std::setprecision(3) << med_sp
                      << "x  (IQR=" << std::setprecision(3) << iqr_sp << "x)\n";
            const char * gate = med_sp >= 2.0 ? "PASS" : (med_sp >= 1.5 ? "PARTIAL" : "FAIL");
            std::cout << "Gate TTFT >= 2.0x   : " << gate << " (" << std::setprecision(2) << med_sp << "x)\n";
            if (!args.out_csv.empty())
                std::cout << "Per-prompt CSV      : " << args.out_csv << "\n";
            std::cout << "(Run scripts/score-f1.py " << args.out_csv << " to compute F1)\n";
        }

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    /* ─────────────────────────────── single-prompt mode ────────────────── */

    std::string prompt_text = resolve_prompt(args.prompt);

    int n_prompt = -llama_tokenize(vocab, prompt_text.c_str(),
                                   (int32_t)prompt_text.size(),
                                   nullptr, 0, true, true);
    if (n_prompt <= 0) { std::cerr << "Tokenisation failed\n"; return 1; }
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt_text.c_str(), (int32_t)prompt_text.size(),
                   tokens.data(), n_prompt, true, true);
    if (n_prompt > (int)args.n_ctx) {
        std::cerr << "Warning: prompt truncated " << n_prompt << " → " << args.n_ctx << " tokens\n";
        tokens.resize(args.n_ctx);
        n_prompt = (int)args.n_ctx;
    }

    std::cout << "Prompt: " << n_prompt << " tokens\n";
    std::cout << "Config: pruning_layers=[";
    for (int i = 0; i < (int)args.pruning_layers.size(); i++) {
        if (i) std::cout << ","; std::cout << args.pruning_layers[i];
    }
    std::cout << "] keep_ratios=[";
    for (int i = 0; i < (int)args.keep_ratios.size(); i++) {
        if (i) std::cout << ","; std::cout << args.keep_ratios[i];
    }
    std::cout << "] pool_size=" << args.pool_size;
    if (args.decode_pruning) std::cout << " decode_pruning=on keep_ratio=" << args.decode_keep_ratio;
    std::cout << "\n";

    /* warmup */
    std::cout << "Warming up...\n";
    for (int w = 0; w < 2; w++) {
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch b = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        llama_decode(ctx, b);
    }

    std::ofstream csv_out;
    if (!args.out_csv.empty()) {
        csv_out.open(args.out_csv);
        if (csv_out) csv_out << "run,mode,ttft_ms,n_kept\n";
    }

    std::vector<double> baseline_ms, lazyllm_ms;

    for (int r = 0; r < args.repeat; r++) {
        /* baseline */
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch bb = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        auto t0 = Clock::now();
        int ret = llama_decode(ctx, bb);
        auto t1 = Clock::now();
        if (ret != 0) { std::cerr << "baseline decode failed\n"; continue; }
        const double b_ms = elapsed_ms(t0, t1);
        baseline_ms.push_back(b_ms);
        if (csv_out.is_open()) csv_out << r << ",baseline," << std::fixed << std::setprecision(2) << b_ms << "," << n_prompt << "\n";

        /* LazyLLM */
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_lazyllm_context * lz_ctx = llama_lazyllm_init_with_params(ctx, lz_params);
        if (!lz_ctx) continue;
        if (args.decode_pruning) llama_lazyllm_decode_pruning_enable(lz_ctx, args.decode_keep_ratio, 0);
        llama_batch bl = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        auto t2 = Clock::now();
        int n_kept = llama_lazyllm_prefill(lz_ctx, bl);
        auto t3 = Clock::now();
        llama_lazyllm_free(lz_ctx);
        if (n_kept < 0) { std::cerr << "lazyllm prefill failed\n"; continue; }
        const double lz_ms = elapsed_ms(t2, t3);
        lazyllm_ms.push_back(lz_ms);
        if (csv_out.is_open()) csv_out << r << ",lazyllm," << std::fixed << std::setprecision(2) << lz_ms << "," << n_kept << "\n";

        std::cout << "  run " << std::setw(2) << r
                  << "  baseline=" << std::fixed << std::setprecision(1) << std::setw(8) << b_ms << " ms"
                  << "  lazyllm="  << std::setw(8) << lz_ms << " ms"
                  << "  speedup="  << std::setprecision(2) << std::setw(5) << (lz_ms > 0 ? b_ms/lz_ms : 0) << "x"
                  << "  kept=" << n_kept << "/" << n_prompt << "\n";
    }

    if (!baseline_ms.empty() && !lazyllm_ms.empty()) {
        const double mb = median(baseline_ms), ml = median(lazyllm_ms);
        const double sp = ml > 0 ? mb/ml : 0;
        std::cout << "\nMedian baseline TTFT : " << std::fixed << std::setprecision(2) << mb << " ms\n";
        std::cout << "Median LazyLLM TTFT  : " << ml << " ms\n";
        std::cout << "Speedup ratio        : " << std::setprecision(3) << sp << "x\n";
        const char * gate = sp >= 2.0 ? "PASS" : (sp >= 1.5 ? "PARTIAL" : "FAIL");
        std::cout << "Gate TTFT >= 2.0x    : " << gate << " (" << std::setprecision(2) << sp << "x)\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
