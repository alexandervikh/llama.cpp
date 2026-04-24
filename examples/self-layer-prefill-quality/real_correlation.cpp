/*
 * Real Correlation Analysis for Self-Layer Prefill
 * 
 * Since ggml reuses compute buffers, we can't extract Q/K from all layers after
 * a single decode. Instead, we measure correlation by:
 * 
 * 1. Running partial decode (N early layers) and extracting importance scores
 * 2. Running full decode and extracting importance from last layer
 * 3. Computing Spearman correlation of token rankings
 * 4. Computing Jaccard similarity of selected token sets at various keep_ratios
 */

#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO] ");
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_bos) {
    int n_tokens = text.length() + 16;
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), n_tokens, add_bos, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_bos, false);
    }
    tokens.resize(n_tokens);
    return tokens;
}

// Compute Spearman rank correlation
static float spearman_correlation(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.size() < 2) return 0.f;
    size_t n = a.size();
    
    // Create indices and sort by values
    std::vector<size_t> idx_a(n), idx_b(n);
    std::iota(idx_a.begin(), idx_a.end(), 0);
    std::iota(idx_b.begin(), idx_b.end(), 0);
    
    std::sort(idx_a.begin(), idx_a.end(), [&a](size_t i, size_t j) { return a[i] < a[j]; });
    std::sort(idx_b.begin(), idx_b.end(), [&b](size_t i, size_t j) { return b[i] < b[j]; });
    
    // Assign ranks
    std::vector<double> rank_a(n), rank_b(n);
    for (size_t i = 0; i < n; i++) {
        rank_a[idx_a[i]] = i + 1;
        rank_b[idx_b[i]] = i + 1;
    }
    
    // Compute Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
    double sum_d2 = 0;
    for (size_t i = 0; i < n; i++) {
        double d = rank_a[i] - rank_b[i];
        sum_d2 += d * d;
    }
    return 1.0f - (6.0 * sum_d2) / (n * (n * n - 1));
}

// Compute Jaccard similarity of top-k selections
static float jaccard_similarity(const std::vector<float>& a, const std::vector<float>& b, float keep_ratio) {
    if (a.size() != b.size() || a.empty()) return 0.f;
    size_t n = a.size();
    size_t k = std::max((size_t)1, (size_t)(n * keep_ratio));
    
    std::vector<size_t> idx_a(n), idx_b(n);
    std::iota(idx_a.begin(), idx_a.end(), 0);
    std::iota(idx_b.begin(), idx_b.end(), 0);
    
    std::sort(idx_a.begin(), idx_a.end(), [&a](size_t i, size_t j) { return a[i] > a[j]; });
    std::sort(idx_b.begin(), idx_b.end(), [&b](size_t i, size_t j) { return b[i] > b[j]; });
    
    std::vector<bool> in_a(n, false), in_b(n, false);
    for (size_t i = 0; i < k; i++) {
        in_a[idx_a[i]] = true;
        in_b[idx_b[i]] = true;
    }
    
    size_t intersection = 0, union_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (in_a[i] && in_b[i]) intersection++;
        if (in_a[i] || in_b[i]) union_count++;
    }
    
    return union_count > 0 ? (float)intersection / union_count : 0.f;
}

// Run self-layer prefill and return which tokens were selected at given keep_ratio
static bool get_selected_tokens_partial(
    llama_context* ctx,
    const llama_token* tokens,
    int n_tokens,
    int n_early,
    float keep_ratio,
    std::vector<bool>& selected
) {
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = keep_ratio;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    // Use the partial decode path which runs only early layers
    int n_kept = llama_self_layer_prefill_partial(ctx, tokens, n_tokens, &params);
    
    if (n_kept < 0) {
        return false;
    }
    
    // We can't directly get which tokens were selected, but we know how many
    // For correlation analysis, we'll use a different approach
    selected.resize(n_tokens);
    return true;
}

// Get token importance ranking using the partial decode approach
// This runs partial decode (early layers only) and scores tokens
static bool get_importance_via_partial(
    llama_context* ctx,
    const llama_token* tokens,
    int n_tokens,
    int n_early,
    std::vector<float>& scores
) {
    // The partial decode path scores tokens using only early layers
    // We use keep_ratio=1.0 to get all scores without filtering
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = 1.0f;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    // Run partial path - this internally does a partial decode and extracts scores
    // The scores come from the Q/K of the early layers only
    std::vector<float> shallow, deep, full;
    float p1, p2;
    
    int r = llama_self_layer_prefill_attention_profiles(
        ctx, tokens, n_tokens, &params,
        shallow, deep, full, &p1, &p2);
    
    if (r != 0 || shallow.empty()) {
        return false;
    }
    
    // Note: Due to buffer reuse, shallow/deep/full will be similar
    // The "shallow" here is actually computed from the same data as "full"
    // This is a known limitation that we document
    scores = shallow;
    return true;
}

// Get prompts from various sources
static std::vector<std::pair<std::string, std::string>> get_test_prompts() {
    return {
        {"QA-Short",
         "Context: Paris is the capital of France. Berlin is the capital of Germany.\n"
         "Question: What is the capital of France?\nAnswer:"},
        
        {"QA-Medium",
         "Context: The Amazon rainforest covers most of the Amazon basin. "
         "60% is in Brazil, 13% in Peru, 10% in Colombia. "
         "It is the largest tropical rainforest in the world.\n"
         "Question: What percentage of the Amazon is in Brazil?\nAnswer:"},
        
        {"Code",
         "def fibonacci(n):\n"
         "    '''Return the n-th Fibonacci number.'''\n"
         "    if n <= 1:\n"
         "        return n\n"
         "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
         "# Test the function:\nprint(fibonacci(10))\n# Output:"},
        
        {"Math",
         "Problem: A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. "
         "What is the total distance traveled?\n\nSolution: "
         "Distance = speed × time. "
         "First leg: 60 × 2 = 120 miles. "
         "Second leg: 80 × 3 = 240 miles. "
         "Total distance ="},
        
        {"Reasoning",
         "Alice is taller than Bob. Bob is taller than Charlie. Charlie is taller than Diana. "
         "Eve is shorter than Diana. Frank is taller than Alice.\n"
         "Who is the tallest?"},
        
        {"Summarize",
         "Text: Machine learning is a subset of artificial intelligence that enables systems "
         "to learn and improve from experience without being explicitly programmed. "
         "It focuses on developing algorithms that can access data, learn from it, and make predictions. "
         "Deep learning, a subset of machine learning, uses neural networks with many layers.\n"
         "Summary:"},
        
        {"Mixed",
         "You are a helpful assistant. Here is some context:\n"
         "The user has been working on a Python project. They want to add tests.\n"
         "Previous conversation:\nUser: How do I write unit tests?\n"
         "Assistant: Use pytest or unittest.\n"
         "User: Show me an example with pytest.\nAssistant:"},
    };
}

// Load real prompts from LongBench
static std::vector<std::pair<std::string, std::string>> load_longbench(const std::string& path, int max_chars = 1000) {
    std::vector<std::pair<std::string, std::string>> prompts;
    std::string content = read_file(path);
    if (content.empty()) return prompts;
    
    size_t pos = 0;
    int count = 0;
    while (pos < content.size() && count < 5) {
        size_t ctx_start = content.find("\"context\": \"", pos);
        if (ctx_start == std::string::npos) break;
        ctx_start += 12;
        
        size_t ctx_end = ctx_start;
        while (ctx_end < content.size() && !(content[ctx_end] == '"' && content[ctx_end-1] != '\\')) ctx_end++;
        
        std::string context = content.substr(ctx_start, std::min(ctx_end - ctx_start, (size_t)max_chars));
        
        size_t inp_start = content.find("\"input\": \"", ctx_end);
        if (inp_start == std::string::npos) break;
        inp_start += 10;
        
        size_t inp_end = inp_start;
        while (inp_end < content.size() && !(content[inp_end] == '"' && content[inp_end-1] != '\\')) inp_end++;
        
        std::string input = content.substr(inp_start, inp_end - inp_start);
        
        auto unescape = [](std::string& s) {
            size_t p = 0;
            while ((p = s.find("\\n", p)) != std::string::npos) { s.replace(p, 2, "\n"); p++; }
            p = 0;
            while ((p = s.find("\\\"", p)) != std::string::npos) { s.replace(p, 2, "\""); p++; }
        };
        unescape(context);
        unescape(input);
        
        prompts.push_back({"LongBench-" + std::to_string(count), context + "\n\nQuestion: " + input + "\nAnswer:"});
        pos = inp_end;
        count++;
    }
    return prompts;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string longbench_path = "datasets/longbench/subset/qasper.json";
    int n_ctx = 1024;
    int n_early = 0;
    int n_gpu_layers = -1;
    bool multi_gpu = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--longbench" && i + 1 < argc) {
            longbench_path = argv[++i];
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--n-early" && i + 1 < argc) {
            n_early = std::stoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--multi-gpu") {
            multi_gpu = true;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Usage: real_correlation --model <gguf> [--n-ctx N] [--n-early N] [--n-gpu-layers N] [--multi-gpu]\n";
        return 1;
    }
    
    log_info("Loading model: %s\n", model_path.c_str());
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (n_gpu_layers != 0) {
        mparams.split_mode = multi_gpu ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
    }
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = (int)llama_model_n_layer(model);
    if (n_early <= 0) n_early = std::max(1, n_layer / 4);
    
    log_info("Model: %d layers, n_early=%d (%.1f%%)\n", n_layer, n_early, 100.0f * n_early / n_layer);
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create context\n";
        llama_model_free(model);
        return 1;
    }
    
    log_info("Context created: n_ctx=%d\n", n_ctx);
    
    // Collect prompts
    auto prompts = get_test_prompts();
    auto lb_prompts = load_longbench(longbench_path, 800);
    prompts.insert(prompts.end(), lb_prompts.begin(), lb_prompts.end());
    
    log_info("Testing %zu prompts\n\n", prompts.size());
    
    printf("=== Token Selection Correlation Analysis ===\n");
    printf("Method: Compare token selection using early-layer scores vs full-model scores\n\n");
    
    printf("%-15s %6s %10s %10s %10s %10s\n", 
           "Prompt", "Tokens", "Spearman", "J@0.3", "J@0.5", "J@0.7");
    printf("%-15s %6s %10s %10s %10s %10s\n", 
           "------", "------", "--------", "-----", "-----", "-----");
    
    double sum_spearman = 0;
    double sum_j30 = 0, sum_j50 = 0, sum_j70 = 0;
    int valid_count = 0;
    
    for (const auto& [name, prompt] : prompts) {
        auto tokens = tokenize(vocab, prompt, true);
        if ((int)tokens.size() > n_ctx - 10) {
            tokens.resize(n_ctx - 10);
        }
        if (tokens.size() < 20) continue;
        
        // Get scores using early layers
        std::vector<float> shallow_scores;
        if (!get_importance_scores(ctx, tokens.data(), (int)tokens.size(), n_early, shallow_scores)) {
            printf("%-15s %6d %10s\n", name.substr(0, 15).c_str(), (int)tokens.size(), "FAILED");
            continue;
        }
        
        // Get scores using all layers (use n_layer as n_early to get "full" attention)
        std::vector<float> full_scores;
        if (!get_importance_scores(ctx, tokens.data(), (int)tokens.size(), n_layer, full_scores)) {
            printf("%-15s %6d %10s\n", name.substr(0, 15).c_str(), (int)tokens.size(), "FAILED");
            continue;
        }
        
        float spearman = spearman_correlation(shallow_scores, full_scores);
        float j30 = jaccard_similarity(shallow_scores, full_scores, 0.3f);
        float j50 = jaccard_similarity(shallow_scores, full_scores, 0.5f);
        float j70 = jaccard_similarity(shallow_scores, full_scores, 0.7f);
        
        printf("%-15s %6d %10.4f %10.4f %10.4f %10.4f\n",
               name.substr(0, 15).c_str(), (int)tokens.size(), spearman, j30, j50, j70);
        
        if (!std::isnan(spearman)) {
            sum_spearman += spearman;
            sum_j30 += j30;
            sum_j50 += j50;
            sum_j70 += j70;
            valid_count++;
        }
    }
    
    printf("\n=== Summary ===\n");
    printf("Valid samples: %d\n", valid_count);
    if (valid_count > 0) {
        printf("Average Spearman Correlation: %.4f\n", sum_spearman / valid_count);
        printf("Average Jaccard@0.3: %.4f\n", sum_j30 / valid_count);
        printf("Average Jaccard@0.5: %.4f\n", sum_j50 / valid_count);
        printf("Average Jaccard@0.7: %.4f\n", sum_j70 / valid_count);
        
        printf("\n{\"model\": \"%s\", \"n_early\": %d, \"n_layer\": %d, ", model_path.c_str(), n_early, n_layer);
        printf("\"avg_spearman\": %.4f, ", sum_spearman / valid_count);
        printf("\"avg_jaccard_0.3\": %.4f, ", sum_j30 / valid_count);
        printf("\"avg_jaccard_0.5\": %.4f, ", sum_j50 / valid_count);
        printf("\"avg_jaccard_0.7\": %.4f, ", sum_j70 / valid_count);
        printf("\"n_samples\": %d}\n", valid_count);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}
