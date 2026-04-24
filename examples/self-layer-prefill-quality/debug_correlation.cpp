/*
 * Debug correlation analysis - inspect raw attention score values
 */

#include "llama.h"
#include "llama-self-layer-prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

static std::string detokenize(const llama_vocab * vocab, llama_token tok) {
    char buf[64] = {0};
    int len = llama_token_to_piece(vocab, tok, buf, 63, 0, false);
    if (len > 0) {
        buf[len] = '\0';
        return std::string(buf);
    }
    return "<?>"; 
}

static float compute_pearson(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.size() < 2) return 0.f;
    double sa = 0, sb = 0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        sa += a[i];
        sb += b[i];
    }
    sa /= n;
    sb /= n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        const double xa = a[i] - sa;
        const double xb = b[i] - sb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    const double eps = 1e-20;
    if (da <= eps || db <= eps) return (da <= eps && db <= eps) ? 1.0f : 0.0f;
    return (float)(num / sqrt(da * db));
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_ctx = 512;
    int n_early = 0;
    int n_gpu_layers = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--n-early" && i + 1 < argc) {
            n_early = std::stoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Usage: debug_correlation --model <gguf> [--n-ctx N] [--n-early N] [--n-gpu-layers N]\n";
        return 1;
    }
    
    log_info("Loading model: %s\n", model_path.c_str());
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (n_gpu_layers != 0) {
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    }
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = (int)llama_model_n_layer(model);
    if (n_early <= 0) n_early = std::max(1, n_layer / 4);
    
    log_info("Model: %d layers, n_early=%d\n", n_layer, n_early);
    
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
    
    log_info("Context created\n");
    
    // Test prompt with diverse content
    std::string prompt = 
        "The quick brown fox jumps over the lazy dog. "
        "Question: What animal jumped?\n"
        "Answer: The fox jumped over the dog.\n"
        "Now let's analyze: the subject was 'fox', the verb was 'jumps', and the object was 'dog'. "
        "Mathematical example: if x = 5 and y = 3, then x + y = 8.\n"
        "Code: def add(a, b): return a + b";
    
    auto tokens = tokenize(vocab, prompt, true);
    log_info("Prompt: %zu tokens\n", tokens.size());
    
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = 1.0f;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    std::vector<float> shallow, deep, full;
    float pearson_sf = 0, pearson_sd = 0;
    
    printf("\nCalling llama_self_layer_prefill_attention_profiles with n_early=%d, n_layer=%d\n", n_early, n_layer);
    
    int r = llama_self_layer_prefill_attention_profiles(
        ctx, tokens.data(), (int)tokens.size(), &params,
        shallow, deep, full, &pearson_sf, &pearson_sd);
    
    if (r != 0) {
        std::cerr << "Failed to compute attention profiles\n";
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("\n=== RAW ATTENTION SCORES ===\n");
    printf("%-4s %-15s %12s %12s %12s\n", "Pos", "Token", "Shallow", "Deep", "Full");
    printf("%-4s %-15s %12s %12s %12s\n", "---", "-----", "-------", "----", "----");
    
    for (size_t i = 0; i < shallow.size() && i < tokens.size(); i++) {
        std::string tok_str = detokenize(vocab, tokens[i]);
        // Escape newlines and tabs for display
        for (auto& c : tok_str) {
            if (c == '\n') c = 'N';
            if (c == '\t') c = 'T';
        }
        if (tok_str.length() > 15) tok_str = tok_str.substr(0, 12) + "...";
        
        printf("%4zu %-15s %12.4f %12.4f %12.4f\n", 
               i, tok_str.c_str(), shallow[i], deep[i], full[i]);
    }
    
    // Statistics
    printf("\n=== STATISTICS ===\n");
    auto stats = [](const std::vector<float>& v, const char* name) {
        float sum = 0, min_v = v[0], max_v = v[0];
        for (float f : v) {
            sum += f;
            min_v = std::min(min_v, f);
            max_v = std::max(max_v, f);
        }
        float mean = sum / v.size();
        float var = 0;
        for (float f : v) var += (f - mean) * (f - mean);
        var /= v.size();
        printf("%s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", 
               name, mean, sqrt(var), min_v, max_v);
    };
    
    stats(shallow, "Shallow");
    stats(deep, "Deep");
    stats(full, "Full");
    
    // Compute correlations manually
    float my_pearson_sf = compute_pearson(shallow, full);
    float my_pearson_sd = compute_pearson(shallow, deep);
    
    printf("\n=== CORRELATIONS ===\n");
    printf("API-returned Pearson(shallow, full): %.6f\n", pearson_sf);
    printf("API-returned Pearson(shallow, deep): %.6f\n", pearson_sd);
    printf("Recomputed   Pearson(shallow, full): %.6f\n", my_pearson_sf);
    printf("Recomputed   Pearson(shallow, deep): %.6f\n", my_pearson_sd);
    
    // Check if arrays are identical
    bool shallow_full_identical = true;
    bool shallow_deep_identical = true;
    float max_diff_sf = 0, max_diff_sd = 0;
    for (size_t i = 0; i < shallow.size(); i++) {
        if (fabs(shallow[i] - full[i]) > 1e-6) {
            shallow_full_identical = false;
            max_diff_sf = std::max(max_diff_sf, (float)fabs(shallow[i] - full[i]));
        }
        if (fabs(shallow[i] - deep[i]) > 1e-6) {
            shallow_deep_identical = false;
            max_diff_sd = std::max(max_diff_sd, (float)fabs(shallow[i] - deep[i]));
        }
    }
    
    printf("\nShallow vs Full: %s (max diff: %.6f)\n", 
           shallow_full_identical ? "IDENTICAL!" : "Different", max_diff_sf);
    printf("Shallow vs Deep: %s (max diff: %.6f)\n", 
           shallow_deep_identical ? "IDENTICAL!" : "Different", max_diff_sd);
    
    // Check if scores are proportional (linear scaling)
    if (!shallow_full_identical && !shallow.empty()) {
        float ratio = full[0] / shallow[0];
        bool proportional = true;
        for (size_t i = 1; i < shallow.size(); i++) {
            if (fabs(shallow[i]) > 1e-6) {
                float r = full[i] / shallow[i];
                if (fabs(r - ratio) > 0.01) {
                    proportional = false;
                    break;
                }
            }
        }
        printf("Shallow vs Full proportional? %s (ratio ~%.4f)\n", 
               proportional ? "YES" : "NO", ratio);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}
