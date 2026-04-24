/*
 * Attention Correlation Analysis on Real Benchmark Prompts
 * 
 * Tests correlation between shallow and full attention scores
 * across diverse real-world prompts from multiple sources.
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

// Real benchmark prompts from various sources
static std::vector<std::pair<std::string, std::string>> get_benchmark_prompts() {
    return {
        // MMLU-style questions
        {"MMLU-History", 
         "Question: The longest war in United States history, sometimes known as the \"forgotten war,\" was\n"
         "A) the Spanish-American War\nB) the Korean War\nC) the Vietnam War\nD) the War in Afghanistan\n"
         "Answer:"},
        
        {"MMLU-Science",
         "Question: Which of the following is NOT a function of the liver?\n"
         "A) Production of bile\nB) Detoxification of harmful substances\n"
         "C) Production of insulin\nD) Storage of glycogen\nAnswer:"},
        
        {"MMLU-Math",
         "Question: If f(x) = 3x^2 - 2x + 1, what is f'(x)?\n"
         "A) 6x - 2\nB) 6x + 2\nC) 3x - 2\nD) 6x^2 - 2\nAnswer:"},
        
        // GSM8K-style math problems
        {"GSM8K-1",
         "Question: A store sells apples for $2 each and oranges for $3 each. "
         "If Maria buys 5 apples and 4 oranges, how much does she spend in total?\n"
         "Let's solve this step by step:\n"},
        
        {"GSM8K-2",
         "Question: A train travels at 60 miles per hour. Another train travels at 80 miles per hour. "
         "If they start from the same point and travel in the same direction, "
         "how far apart will they be after 3 hours?\n"
         "Solution:"},
        
        // Code generation prompts
        {"HumanEval-1",
         "def fibonacci(n: int) -> int:\n"
         "    \"\"\"Return the n-th Fibonacci number.\n"
         "    >>> fibonacci(0)\n"
         "    0\n"
         "    >>> fibonacci(1)\n"
         "    1\n"
         "    >>> fibonacci(10)\n"
         "    55\n"
         "    \"\"\"\n"},
        
        {"HumanEval-2",
         "def is_palindrome(s: str) -> bool:\n"
         "    \"\"\"Check if the given string is a palindrome.\n"
         "    >>> is_palindrome('racecar')\n"
         "    True\n"
         "    >>> is_palindrome('hello')\n"
         "    False\n"
         "    \"\"\"\n"},
        
        // QA prompts (similar to SQuAD/NaturalQuestions)
        {"QA-1",
         "Context: The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest "
         "in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses "
         "7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. The majority of the forest "
         "is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with "
         "10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.\n\n"
         "Question: What percentage of the Amazon rainforest is in Brazil?\nAnswer:"},
        
        {"QA-2",
         "Context: The theory of relativity usually encompasses two interrelated theories by Albert Einstein: "
         "special relativity and general relativity. Special relativity was published in 1905 and general "
         "relativity was published in 1915. The theory transformed theoretical physics and astronomy during "
         "the 20th century, superseding a 200-year-old theory of mechanics created primarily by Isaac Newton.\n\n"
         "Question: When was general relativity published?\nAnswer:"},
        
        // Summarization prompts
        {"Summarize-1",
         "Summarize the following text:\n\n"
         "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural "
         "intelligence displayed by animals including humans. AI research has been defined as the field "
         "of study of intelligent agents, which refers to any system that perceives its environment and "
         "takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' "
         "had previously been used to describe machines that mimic and display 'human' cognitive skills "
         "that are associated with the human mind, such as 'learning' and 'problem-solving'. This definition "
         "has since been rejected by major AI researchers who now describe AI in terms of rationality and "
         "acting rationally, which does not limit how intelligence can be articulated.\n\nSummary:"},
        
        // Translation prompts
        {"Translate-1",
         "Translate the following English text to French:\n\n"
         "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet "
         "and is commonly used for typing practice and font demonstrations.\n\nFrench translation:"},
        
        // Instruction following
        {"Instruction-1",
         "Write a professional email to a colleague requesting a meeting to discuss the quarterly report. "
         "The email should be polite, concise, and include a proposed time.\n\nSubject:"},
        
        // Long context (truncated for token limit)
        {"LongContext-1",
         "The following is a detailed analysis of machine learning algorithms used in natural language processing. "
         "Machine learning has revolutionized how computers process and understand human language. "
         "Traditional rule-based systems required extensive manual coding of linguistic rules, whereas modern "
         "machine learning approaches can learn patterns directly from data. Neural networks, particularly "
         "transformers, have become the dominant architecture for NLP tasks. The attention mechanism allows "
         "these models to weigh the importance of different parts of the input when generating output. "
         "Pre-training on large corpora followed by fine-tuning on specific tasks has proven highly effective. "
         "Models like BERT, GPT, and T5 have achieved state-of-the-art results on numerous benchmarks. "
         "However, these models also face challenges including computational costs, environmental impact, "
         "and issues with bias and fairness. Research continues to address these limitations while pushing "
         "the boundaries of what is possible with language understanding and generation.\n\n"
         "Based on the above, explain the key advantage of transformer models:"},
        
        // Reasoning prompts
        {"Reasoning-1",
         "Alice is older than Bob. Bob is older than Charlie. Charlie is older than Diana. "
         "Who is the youngest?\n\nLet me think through this:"},
        
        {"Reasoning-2",
         "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?\n"
         "Think carefully:"},
    };
}

// Load prompts from LongBench JSON file
static std::vector<std::pair<std::string, std::string>> load_longbench_prompts(
    const std::string & path, int max_context_chars = 2000
) {
    std::vector<std::pair<std::string, std::string>> prompts;
    std::string content = read_file(path);
    if (content.empty()) return prompts;
    
    // Simple extraction of context + input pairs
    size_t pos = 0;
    int count = 0;
    while (pos < content.size() && count < 10) {
        // Find "context": "
        size_t ctx_start = content.find("\"context\": \"", pos);
        if (ctx_start == std::string::npos) break;
        ctx_start += 12;
        
        // Find end of context string
        size_t ctx_end = ctx_start;
        while (ctx_end < content.size()) {
            if (content[ctx_end] == '"' && content[ctx_end-1] != '\\') break;
            ctx_end++;
        }
        
        std::string context = content.substr(ctx_start, std::min(ctx_end - ctx_start, (size_t)max_context_chars));
        
        // Find "input": "
        size_t inp_start = content.find("\"input\": \"", ctx_end);
        if (inp_start == std::string::npos) break;
        inp_start += 10;
        
        size_t inp_end = inp_start;
        while (inp_end < content.size()) {
            if (content[inp_end] == '"' && content[inp_end-1] != '\\') break;
            inp_end++;
        }
        
        std::string input = content.substr(inp_start, inp_end - inp_start);
        
        // Unescape basic sequences
        auto unescape = [](std::string& s) {
            size_t p = 0;
            while ((p = s.find("\\n", p)) != std::string::npos) {
                s.replace(p, 2, "\n"); p++;
            }
            p = 0;
            while ((p = s.find("\\\"", p)) != std::string::npos) {
                s.replace(p, 2, "\""); p++;
            }
        };
        unescape(context);
        unescape(input);
        
        std::string full_prompt = context.substr(0, max_context_chars) + "\n\nQuestion: " + input + "\nAnswer:";
        prompts.push_back({"LongBench-" + std::to_string(count), full_prompt});
        
        pos = inp_end;
        count++;
    }
    
    return prompts;
}

struct CorrelationResult {
    std::string name;
    int n_tokens;
    float pearson_shallow_full;
    float pearson_shallow_deep;
};

int main(int argc, char ** argv) {
    std::string model_path;
    std::string longbench_path = "datasets/longbench/subset/qasper.json";
    int n_ctx = 2048;
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
        std::cerr << "Usage: attention_correlation --model <gguf> [--n-ctx N] [--n-early N] [--n-gpu-layers N] [--multi-gpu] [--longbench path]\n";
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
    
    log_info("Context created: n_ctx=%d\n", n_ctx);
    
    // Collect all prompts
    auto prompts = get_benchmark_prompts();
    auto lb_prompts = load_longbench_prompts(longbench_path, 1500);
    prompts.insert(prompts.end(), lb_prompts.begin(), lb_prompts.end());
    
    log_info("Total prompts to analyze: %zu\n", prompts.size());
    
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = 1.0f;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    std::vector<CorrelationResult> results;
    double sum_sf = 0, sum_sd = 0;
    int valid_count = 0;
    
    printf("\n%-20s %8s %12s %12s\n", "Prompt", "Tokens", "P(shal,full)", "P(shal,deep)");
    printf("%-20s %8s %12s %12s\n", "------", "------", "------------", "------------");
    
    for (const auto & [name, prompt] : prompts) {
        auto tokens = tokenize(vocab, prompt, true);
        
        // Skip if too long
        if ((int)tokens.size() > n_ctx - 10) {
            tokens.resize(n_ctx - 10);
        }
        
        // Skip very short prompts
        if (tokens.size() < 20) continue;
        
        std::vector<float> shallow, deep, full;
        float pearson_sf = 0, pearson_sd = 0;
        
        int r = llama_self_layer_prefill_attention_profiles(
            ctx, tokens.data(), (int)tokens.size(), &params,
            shallow, deep, full, &pearson_sf, &pearson_sd);
        
        if (r == 0) {
            CorrelationResult cr;
            cr.name = name;
            cr.n_tokens = (int)tokens.size();
            cr.pearson_shallow_full = pearson_sf;
            cr.pearson_shallow_deep = pearson_sd;
            results.push_back(cr);
            
            // Check for valid (non-NaN) correlations
            if (!std::isnan(pearson_sf) && !std::isnan(pearson_sd)) {
                sum_sf += pearson_sf;
                sum_sd += pearson_sd;
                valid_count++;
            }
            
            printf("%-20s %8d %12.4f %12.4f\n", 
                   name.substr(0, 20).c_str(), (int)tokens.size(), pearson_sf, pearson_sd);
        } else {
            printf("%-20s %8d %12s %12s\n", 
                   name.substr(0, 20).c_str(), (int)tokens.size(), "FAILED", "FAILED");
        }
    }
    
    printf("\n");
    printf("=== Summary ===\n");
    printf("Valid samples: %d / %zu\n", valid_count, results.size());
    
    if (valid_count > 0) {
        double avg_sf = sum_sf / valid_count;
        double avg_sd = sum_sd / valid_count;
        
        // Calculate std dev
        double var_sf = 0, var_sd = 0;
        for (const auto & r : results) {
            if (!std::isnan(r.pearson_shallow_full)) {
                var_sf += (r.pearson_shallow_full - avg_sf) * (r.pearson_shallow_full - avg_sf);
            }
            if (!std::isnan(r.pearson_shallow_deep)) {
                var_sd += (r.pearson_shallow_deep - avg_sd) * (r.pearson_shallow_deep - avg_sd);
            }
        }
        double std_sf = sqrt(var_sf / valid_count);
        double std_sd = sqrt(var_sd / valid_count);
        
        printf("\nPearson(shallow, full): %.4f ± %.4f\n", avg_sf, std_sf);
        printf("Pearson(shallow, deep): %.4f ± %.4f\n", avg_sd, std_sd);
        
        // Find min/max
        float min_sf = 1.0f, max_sf = -1.0f;
        std::string min_name, max_name;
        for (const auto & r : results) {
            if (!std::isnan(r.pearson_shallow_full)) {
                if (r.pearson_shallow_full < min_sf) { min_sf = r.pearson_shallow_full; min_name = r.name; }
                if (r.pearson_shallow_full > max_sf) { max_sf = r.pearson_shallow_full; max_name = r.name; }
            }
        }
        printf("\nRange: [%.4f (%s), %.4f (%s)]\n", min_sf, min_name.c_str(), max_sf, max_name.c_str());
        
        // JSON output
        printf("\n{\"model\": \"%s\", \"n_early\": %d, \"n_layer\": %d, ", model_path.c_str(), n_early, n_layer);
        printf("\"avg_pearson_shallow_full\": %.4f, \"std_pearson_shallow_full\": %.4f, ", avg_sf, std_sf);
        printf("\"avg_pearson_shallow_deep\": %.4f, \"std_pearson_shallow_deep\": %.4f, ", avg_sd, std_sd);
        printf("\"n_samples\": %d}\n", valid_count);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}
