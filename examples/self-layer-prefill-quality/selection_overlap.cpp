/*
 * Selection Overlap Analysis for Self-Layer Prefill
 * 
 * Measures how consistently the early-layer attention selects tokens
 * compared to using different numbers of early layers.
 * 
 * This validates whether the "proxy" (early layers) reliably identifies
 * important tokens by comparing selection at different n_early values.
 */

#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
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

// Run self-layer prefill and return the number of kept tokens
static int run_prefill(
    llama_context* ctx,
    const llama_token* tokens,
    int n_tokens,
    int n_early,
    float keep_ratio
) {
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = keep_ratio;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    return llama_self_layer_prefill_partial(ctx, tokens, n_tokens, &params);
}

// Get benchmark prompts
static std::vector<std::pair<std::string, std::string>> get_prompts() {
    return {
        {"QA-1",
         "Context: The Amazon rainforest covers most of the Amazon basin of South America. "
         "This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. "
         "The majority of the forest is contained within Brazil, with 60% of the rainforest.\n"
         "Question: What percentage of the Amazon rainforest is in Brazil?\nAnswer:"},
        
        {"Code-1",
         "def quicksort(arr):\n"
         "    if len(arr) <= 1:\n"
         "        return arr\n"
         "    pivot = arr[len(arr) // 2]\n"
         "    left = [x for x in arr if x < pivot]\n"
         "    middle = [x for x in arr if x == pivot]\n"
         "    right = [x for x in arr if x > pivot]\n"
         "    return quicksort(left) + middle + quicksort(right)\n\n"
         "# Test:\nprint(quicksort([3, 6, 8, 10, 1, 2, 1]))\n# Output:"},
        
        {"Math-1",
         "Problem: A company has 120 employees. 40% work in engineering, 25% in sales, "
         "20% in marketing, and the rest in administration. How many employees work in administration?\n"
         "Solution:"},
        
        {"Reasoning",
         "Facts: All mammals are warm-blooded. All dogs are mammals. Buddy is a dog.\n"
         "Question: Is Buddy warm-blooded? Explain your reasoning.\nAnswer:"},
        
        {"Summary",
         "Text: Climate change refers to long-term shifts in temperatures and weather patterns. "
         "These shifts may be natural, such as through variations in the solar cycle, but since the 1800s, "
         "human activities have been the main driver of climate change, primarily due to burning fossil fuels "
         "like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a "
         "blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.\n"
         "Key points:"},
    };
}

// Load LongBench samples
static std::vector<std::pair<std::string, std::string>> load_longbench(const std::string& path, int max_chars = 1500) {
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
        
        prompts.push_back({"LongBench-" + std::to_string(count), context + "\n\nQ: " + input + "\nA:"});
        pos = inp_end;
        count++;
    }
    return prompts;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string longbench_path = "datasets/longbench/subset/qasper.json";
    int n_ctx = 2048;
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
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--multi-gpu") {
            multi_gpu = true;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Usage: selection_overlap --model <gguf> [--n-ctx N] [--n-gpu-layers N] [--multi-gpu]\n";
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
    
    log_info("Model: %d layers\n", n_layer);
    
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
    
    // Collect prompts
    auto prompts = get_prompts();
    auto lb_prompts = load_longbench(longbench_path, 1200);
    prompts.insert(prompts.end(), lb_prompts.begin(), lb_prompts.end());
    
    log_info("Testing %zu prompts\n\n", prompts.size());
    
    // Test different n_early values
    std::vector<int> n_early_values;
    for (int ne = 2; ne <= n_layer; ne += std::max(1, n_layer / 8)) {
        n_early_values.push_back(ne);
    }
    if (n_early_values.back() != n_layer) {
        n_early_values.push_back(n_layer);
    }
    
    std::vector<float> keep_ratios = {0.3f, 0.5f, 0.7f};
    
    printf("=== Token Retention Analysis ===\n");
    printf("Compares number of tokens kept at different n_early values\n");
    printf("Higher consistency suggests early layers are good proxies\n\n");
    
    for (float kr : keep_ratios) {
        printf("\n--- keep_ratio = %.1f ---\n", kr);
        printf("%-15s %6s", "Prompt", "Tokens");
        for (int ne : n_early_values) {
            printf(" %6s", ("N" + std::to_string(ne)).c_str());
        }
        printf("\n");
        
        printf("%-15s %6s", "------", "------");
        for (size_t j = 0; j < n_early_values.size(); j++) {
            printf(" %6s", "------");
        }
        printf("\n");
        
        for (const auto& [name, prompt] : prompts) {
            auto tokens = tokenize(vocab, prompt, true);
            if ((int)tokens.size() > n_ctx - 10) {
                tokens.resize(n_ctx - 10);
            }
            if (tokens.size() < 50) continue;
            
            printf("%-15s %6d", name.substr(0, 15).c_str(), (int)tokens.size());
            
            for (int ne : n_early_values) {
                int n_kept = run_prefill(ctx, tokens.data(), (int)tokens.size(), ne, kr);
                printf(" %6d", n_kept);
            }
            printf("\n");
        }
    }
    
    // Compute consistency metrics
    printf("\n=== Consistency Analysis ===\n");
    printf("Measuring variation in n_kept across different n_early values\n\n");
    
    double total_cv = 0;  // Coefficient of variation
    int cv_count = 0;
    
    for (float kr : keep_ratios) {
        double sum_cv = 0;
        int count = 0;
        
        for (const auto& [name, prompt] : prompts) {
            auto tokens = tokenize(vocab, prompt, true);
            if ((int)tokens.size() > n_ctx - 10) tokens.resize(n_ctx - 10);
            if (tokens.size() < 50) continue;
            
            std::vector<int> kept_counts;
            for (int ne : n_early_values) {
                int n_kept = run_prefill(ctx, tokens.data(), (int)tokens.size(), ne, kr);
                if (n_kept > 0) kept_counts.push_back(n_kept);
            }
            
            if (kept_counts.size() >= 2) {
                double mean = 0;
                for (int k : kept_counts) mean += k;
                mean /= kept_counts.size();
                
                double var = 0;
                for (int k : kept_counts) var += (k - mean) * (k - mean);
                var /= kept_counts.size();
                
                double cv = (mean > 0) ? sqrt(var) / mean : 0;
                sum_cv += cv;
                count++;
            }
        }
        
        if (count > 0) {
            double avg_cv = sum_cv / count;
            printf("keep_ratio=%.1f: avg CV = %.4f (lower is more consistent)\n", kr, avg_cv);
            total_cv += avg_cv;
            cv_count++;
        }
    }
    
    printf("\nOverall average CV: %.4f\n", cv_count > 0 ? total_cv / cv_count : 0);
    printf("\nInterpretation:\n");
    printf("- CV < 0.05: Very consistent (early layers are excellent proxies)\n");
    printf("- CV 0.05-0.15: Moderately consistent (reasonable proxy quality)\n");
    printf("- CV > 0.15: High variation (early layers may not be reliable proxies)\n");
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}
