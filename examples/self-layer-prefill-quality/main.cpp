/*
 * Self-Layer Prefill Quality Evaluation
 * 
 * Implements 5 phases of quality validation:
 * - Phase 1: Perplexity evaluation
 * - Phase 2: LongBench-style QA (simplified)
 * - Phase 3: PassKey retrieval (needle-in-haystack)
 * - Phase 4: Generation comparison
 * - Phase 5: Attention score analysis
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

// ============================================================================
// Utilities
// ============================================================================

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO] ");
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

static void log_result(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    fflush(stdout);
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        return "";
    }
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

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    for (llama_token tok : tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) {
            result.append(buf, n);
        }
    }
    return result;
}

// ============================================================================
// Phase 1: Perplexity Evaluation
// ============================================================================

struct PerplexityResult {
    double ppl;
    double nll;
    int n_tokens;
    double time_ms;
};

static double compute_log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += exp(logits[i] - max_logit);
    }
    return logits[tok] - max_logit - log(sum_exp);
}

static PerplexityResult compute_perplexity_baseline(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    int n_ctx
) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    
    double nll = 0.0;
    int n_evaluated = 0;
    double t0 = now_ms();
    
    int effective_ctx = std::min(n_ctx, (int)tokens.size() - 1);
    if (effective_ctx < 16) {
        return {0.0, 0.0, 0, 0.0};
    }
    
    // Use same split as filtered: 80% context, 20% evaluation
    int ctx_len = (int)(effective_ctx * 0.8);
    int eval_start = ctx_len;
    int eval_end = (int)tokens.size();
    
    llama_memory_clear(llama_get_memory(ctx), false);
    
    // Process context
    llama_batch batch = llama_batch_init(ctx_len, 0, 1);
    for (int j = 0; j < ctx_len; j++) {
        batch.token[batch.n_tokens] = tokens[j];
        batch.pos[batch.n_tokens] = j;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (j == ctx_len - 1);
        batch.n_tokens++;
    }
    
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return {0.0, 0.0, 0, now_ms() - t0};
    }
    llama_batch_free(batch);
    
    // Evaluate continuation predictions
    for (int i = eval_start; i < eval_end; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        
        int next_tok = tokens[i];
        double log_prob = compute_log_softmax(n_vocab, logits, next_tok);
        nll -= log_prob;
        n_evaluated++;
        
        // Feed actual next token
        llama_batch next_batch = llama_batch_get_one(&next_tok, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            break;
        }
    }
    
    double t1 = now_ms();
    double ppl = n_evaluated > 0 ? exp(nll / n_evaluated) : 0.0;
    
    return {ppl, nll, n_evaluated, t1 - t0};
}

static PerplexityResult compute_perplexity_filtered(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    int n_ctx,
    int n_early,
    float keep_ratio
) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    
    double nll = 0.0;
    int n_evaluated = 0;
    double t0 = now_ms();
    
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = keep_ratio;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    params.chunk_size = 32;
    
    int effective_ctx = std::min(n_ctx, (int)tokens.size() - 1);
    if (effective_ctx < 16) {
        return {0.0, 0.0, 0, 0.0};  // Too short for meaningful eval
    }
    
    // Process: use first 80% as context, evaluate on remaining 20%
    int ctx_len = (int)(effective_ctx * 0.8);
    int eval_start = ctx_len;
    int eval_end = (int)tokens.size();
    
    // Process context with filtering
    int n_kept = llama_self_layer_prefill_partial(ctx, tokens.data(), ctx_len, &params);
    
    if (n_kept < 0) {
        return {0.0, 0.0, 0, now_ms() - t0};
    }
    
    // Evaluate continuation predictions
    for (int i = eval_start; i < eval_end; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        
        int next_tok = tokens[i];
        double log_prob = compute_log_softmax(n_vocab, logits, next_tok);
        nll -= log_prob;
        n_evaluated++;
        
        // Feed actual next token
        llama_batch batch = llama_batch_get_one(&next_tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }
    
    double t1 = now_ms();
    double ppl = n_evaluated > 0 ? exp(nll / n_evaluated) : 0.0;
    
    return {ppl, nll, n_evaluated, t1 - t0};
}

static void run_phase1_perplexity(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & test_text,
    int n_ctx,
    int n_early,
    const std::vector<float> & keep_ratios
) {
    log_info("\n=== Phase 1: Perplexity Evaluation ===\n");
    
    auto tokens = tokenize(vocab, test_text, true);
    log_info("Test text: %zu tokens\n", tokens.size());
    
    // Limit to reasonable size
    if (tokens.size() > (size_t)(n_ctx * 10)) {
        tokens.resize(n_ctx * 10);
        log_info("Truncated to %zu tokens\n", tokens.size());
    }
    
    // Baseline
    log_info("Running baseline (kr=1.00)...\n");
    auto baseline = compute_perplexity_baseline(ctx, tokens, n_ctx);
    log_info("  Baseline PPL: %.4f (%.1f ms)\n", baseline.ppl, baseline.time_ms);
    
    log_result("{\n  \"phase\": \"perplexity\",\n  \"n_tokens\": %d,\n  \"n_ctx\": %d,\n  \"n_early\": %d,\n",
               (int)tokens.size(), n_ctx, n_early);
    log_result("  \"baseline_ppl\": %.4f,\n  \"results\": [\n", baseline.ppl);
    
    bool first = true;
    for (float kr : keep_ratios) {
        if (kr >= 1.0f) continue;
        
        log_info("Running kr=%.2f...\n", kr);
        auto result = compute_perplexity_filtered(ctx, tokens, n_ctx, n_early, kr);
        double ppl_ratio = result.ppl / baseline.ppl;
        
        log_info("  PPL: %.4f (ratio: %.2fx, %.1f ms)\n", result.ppl, ppl_ratio, result.time_ms);
        
        if (!first) log_result(",\n");
        first = false;
        log_result("    {\"keep_ratio\": %.2f, \"ppl\": %.4f, \"ppl_ratio\": %.4f, \"time_ms\": %.1f}",
                   kr, result.ppl, ppl_ratio, result.time_ms);
    }
    
    log_result("\n  ]\n}\n");
}

// ============================================================================
// Phase 3: PassKey Retrieval (Needle-in-Haystack)
// ============================================================================

struct PassKeyResult {
    int passkey;
    float depth;
    int n_total;
    int n_kept;
    bool passkey_retrieved;
    std::string response;
};

static std::string generate_filler_text(int n_chars) {
    static const char * sentences[] = {
        "The quick brown fox jumps over the lazy dog. ",
        "A journey of a thousand miles begins with a single step. ",
        "To be or not to be that is the question. ",
        "All that glitters is not gold. ",
        "Actions speak louder than words. ",
        "The early bird catches the worm. ",
        "When in Rome do as the Romans do. ",
        "A picture is worth a thousand words. ",
        "Better late than never but never late is better. ",
        "Knowledge is power but enthusiasm pulls the switch. ",
    };
    static const int n_sentences = sizeof(sentences) / sizeof(sentences[0]);
    
    std::string result;
    result.reserve(n_chars + 100);
    std::mt19937 rng(42);
    
    while ((int)result.size() < n_chars) {
        result += sentences[rng() % n_sentences];
    }
    
    return result;
}

static std::string generate_response(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_past,
    int max_tokens = 32
) {
    std::vector<llama_token> response_tokens;
    int cur_pos = n_past;
    
    for (int i = 0; i < max_tokens; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        
        // Greedy sampling
        int n_vocab = llama_vocab_n_tokens(vocab);
        int best_tok = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                best_tok = j;
            }
        }
        
        // Check for EOS
        if (best_tok == llama_vocab_eos(vocab) || best_tok == llama_vocab_eot(vocab)) {
            break;
        }
        
        response_tokens.push_back(best_tok);
        
        // Decode next token using llama_batch_get_one for simplicity
        llama_batch batch = llama_batch_get_one(&best_tok, 1);
        
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
        cur_pos++;
    }
    
    return detokenize(vocab, response_tokens);
}

static PassKeyResult run_passkey_test(
    llama_context * ctx,
    const llama_vocab * vocab,
    int passkey,
    float depth,
    int n_filler_tokens,
    int n_early,
    float keep_ratio
) {
    PassKeyResult result;
    result.passkey = passkey;
    result.depth = depth;
    result.n_total = 0;
    result.n_kept = 0;
    result.passkey_retrieved = false;
    
    // Build prompt with passkey at specified depth
    std::string filler = generate_filler_text(n_filler_tokens * 4);  // ~4 chars per token
    
    int insert_pos = (int)(filler.size() * depth);
    std::string passkey_str = "The secret passkey is: " + std::to_string(passkey) + ". Remember this number. ";
    filler.insert(insert_pos, passkey_str);
    
    std::string prompt = "Read the following text carefully and remember any important numbers.\n\n"
                        + filler + 
                        "\n\nQuestion: What was the secret passkey mentioned in the text above?\nAnswer: The passkey is ";
    
    auto tokens = tokenize(vocab, prompt, true);
    result.n_total = (int)tokens.size();
    
    // Run with self-layer prefill
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = keep_ratio;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    params.chunk_size = 32;
    
    int n_kept = llama_self_layer_prefill_partial(ctx, tokens.data(), (int)tokens.size(), &params);
    result.n_kept = n_kept > 0 ? n_kept : (int)tokens.size();
    
    // Generate response
    result.response = generate_response(ctx, vocab, result.n_kept, 16);
    
    // Check if passkey is in response
    result.passkey_retrieved = result.response.find(std::to_string(passkey)) != std::string::npos;
    
    return result;
}

static void run_phase3_passkey(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_ctx,
    int n_early,
    const std::vector<float> & keep_ratios
) {
    log_info("\n=== Phase 3: PassKey Retrieval (Needle-in-Haystack) ===\n");
    
    const std::vector<float> depths = {0.1f, 0.25f, 0.5f, 0.75f, 0.9f};
    const int n_filler = std::min(n_ctx - 200, 2000);  // Leave room for prompt/question
    
    std::mt19937 rng(12345);
    
    log_result("{\n  \"phase\": \"passkey\",\n  \"n_filler_tokens\": %d,\n  \"n_early\": %d,\n  \"results\": [\n",
               n_filler, n_early);
    
    bool first = true;
    for (float kr : keep_ratios) {
        int correct = 0;
        int total = 0;
        
        log_info("Testing kr=%.2f:\n", kr);
        
        for (float depth : depths) {
            int passkey = 1000 + (rng() % 9000);  // 4-digit number
            
            auto result = run_passkey_test(ctx, vocab, passkey, depth, n_filler, n_early, kr);
            
            log_info("  depth=%.2f: passkey=%d, kept=%d/%d, retrieved=%s, response=\"%s\"\n",
                     depth, passkey, result.n_kept, result.n_total,
                     result.passkey_retrieved ? "YES" : "NO",
                     result.response.substr(0, 30).c_str());
            
            if (result.passkey_retrieved) correct++;
            total++;
        }
        
        float accuracy = total > 0 ? (float)correct / total : 0.0f;
        log_info("  Overall: %d/%d = %.1f%%\n", correct, total, accuracy * 100);
        
        if (!first) log_result(",\n");
        first = false;
        log_result("    {\"keep_ratio\": %.2f, \"correct\": %d, \"total\": %d, \"accuracy\": %.4f}",
                   kr, correct, total, accuracy);
    }
    
    log_result("\n  ]\n}\n");
}

// ============================================================================
// Phase 4: Generation Comparison
// ============================================================================

struct GenerationResult {
    std::string prompt;
    std::string baseline_response;
    std::string filtered_response;
    bool responses_match;
};

static std::string generate_with_baseline(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    int max_tokens
) {
    llama_memory_clear(llama_get_memory(ctx), false);
    
    llama_batch batch = llama_batch_init((int)prompt_tokens.size(), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = (int)i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == prompt_tokens.size() - 1);
        batch.n_tokens++;
    }
    
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return "[DECODE FAILED]";
    }
    llama_batch_free(batch);
    
    return generate_response(ctx, vocab, (int)prompt_tokens.size(), max_tokens);
}

static std::string generate_with_filter(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    int n_early,
    float keep_ratio,
    int max_tokens
) {
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.keep_ratio = keep_ratio;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    params.chunk_size = 32;
    
    int n_kept = llama_self_layer_prefill_partial(
        ctx, prompt_tokens.data(), (int)prompt_tokens.size(), &params);
    
    if (n_kept < 0) {
        return "[PREFILL FAILED]";
    }
    
    return generate_response(ctx, vocab, n_kept, max_tokens);
}

static void run_phase4_generation(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_early,
    const std::vector<float> & keep_ratios
) {
    log_info("\n=== Phase 4: Generation Comparison ===\n");
    
    // Test prompts of varying complexity
    const std::vector<std::string> prompts = {
        "The capital of France is",
        "Write a haiku about the ocean:\n",
        "Explain why the sky is blue in one sentence:",
        "What is 25 * 4?",
        "Translate 'hello world' to Spanish:",
    };
    
    log_result("{\n  \"phase\": \"generation\",\n  \"n_early\": %d,\n  \"results\": [\n", n_early);
    
    bool first_kr = true;
    for (float kr : keep_ratios) {
        if (kr >= 1.0f) continue;
        
        log_info("Testing kr=%.2f:\n", kr);
        
        if (!first_kr) log_result(",\n");
        first_kr = false;
        log_result("    {\"keep_ratio\": %.2f, \"comparisons\": [\n", kr);
        
        bool first_prompt = true;
        int n_match = 0;
        
        for (const auto & prompt : prompts) {
            auto tokens = tokenize(vocab, prompt, true);
            
            std::string baseline = generate_with_baseline(ctx, vocab, tokens, 32);
            std::string filtered = generate_with_filter(ctx, vocab, tokens, n_early, kr, 32);
            
            // Simple match: first 10 chars match
            bool match = baseline.substr(0, 10) == filtered.substr(0, 10);
            if (match) n_match++;
            
            log_info("  Prompt: \"%s\"\n", prompt.substr(0, 40).c_str());
            log_info("    Baseline: \"%s\"\n", baseline.substr(0, 50).c_str());
            log_info("    Filtered: \"%s\" %s\n", filtered.substr(0, 50).c_str(), match ? "[MATCH]" : "[DIFF]");
            
            if (!first_prompt) log_result(",\n");
            first_prompt = false;
            
            // Escape strings for JSON
            auto escape_json = [](const std::string & s) {
                std::string result;
                for (char c : s) {
                    if (c == '"') result += "\\\"";
                    else if (c == '\\') result += "\\\\";
                    else if (c == '\n') result += "\\n";
                    else if (c == '\r') result += "\\r";
                    else if (c == '\t') result += "\\t";
                    else result += c;
                }
                return result;
            };
            
            log_result("      {\"prompt\": \"%s\", \"baseline\": \"%s\", \"filtered\": \"%s\", \"match\": %s}",
                       escape_json(prompt.substr(0, 50)).c_str(),
                       escape_json(baseline.substr(0, 100)).c_str(),
                       escape_json(filtered.substr(0, 100)).c_str(),
                       match ? "true" : "false");
        }
        
        log_result("\n    ], \"match_rate\": %.2f}", (float)n_match / prompts.size());
        log_info("  Match rate: %d/%zu = %.1f%%\n", n_match, prompts.size(), 100.0f * n_match / prompts.size());
    }
    
    log_result("\n  ]\n}\n");
}

// ============================================================================
// Phase 2: LongBench Tasks
// ============================================================================

struct LongBenchSample {
    std::string input;
    std::string context;
    std::vector<std::string> answers;
    int length;
    std::string dataset;
};

static std::vector<LongBenchSample> load_longbench_json(const std::string & path) {
    std::vector<LongBenchSample> samples;
    std::string content = read_file(path);
    if (content.empty()) return samples;
    
    // Simple JSON array parser (handles our specific format)
    size_t pos = content.find('[');
    if (pos == std::string::npos) return samples;
    
    // Find each object in the array
    size_t start = pos + 1;
    int depth = 0;
    size_t obj_start = std::string::npos;
    
    for (size_t i = start; i < content.size(); i++) {
        if (content[i] == '{') {
            if (depth == 0) obj_start = i;
            depth++;
        } else if (content[i] == '}') {
            depth--;
            if (depth == 0 && obj_start != std::string::npos) {
                std::string obj = content.substr(obj_start, i - obj_start + 1);
                LongBenchSample sample;
                
                // Extract fields (simplified parsing)
                auto extract_string = [&obj](const std::string& key) -> std::string {
                    std::string search = "\"" + key + "\": \"";
                    size_t p = obj.find(search);
                    if (p == std::string::npos) {
                        search = "\"" + key + "\":\"";
                        p = obj.find(search);
                    }
                    if (p == std::string::npos) return "";
                    p += search.length();
                    std::string result;
                    while (p < obj.size() && !(obj[p] == '"' && obj[p-1] != '\\')) {
                        if (obj[p] == '\\' && p + 1 < obj.size()) {
                            char c = obj[p+1];
                            if (c == 'n') result += '\n';
                            else if (c == 't') result += '\t';
                            else if (c == '"') result += '"';
                            else if (c == '\\') result += '\\';
                            else result += c;
                            p += 2;
                        } else {
                            result += obj[p++];
                        }
                    }
                    return result;
                };
                
                auto extract_string_array = [&obj](const std::string& key) -> std::vector<std::string> {
                    std::vector<std::string> result;
                    std::string search = "\"" + key + "\": [";
                    size_t p = obj.find(search);
                    if (p == std::string::npos) {
                        search = "\"" + key + "\":[";
                        p = obj.find(search);
                    }
                    if (p == std::string::npos) return result;
                    p += search.length();
                    
                    while (p < obj.size() && obj[p] != ']') {
                        if (obj[p] == '"') {
                            p++;
                            std::string s;
                            while (p < obj.size() && !(obj[p] == '"' && obj[p-1] != '\\')) {
                                if (obj[p] == '\\' && p + 1 < obj.size()) {
                                    p += 2;
                                } else {
                                    s += obj[p++];
                                }
                            }
                            result.push_back(s);
                            p++;
                        } else {
                            p++;
                        }
                    }
                    return result;
                };
                
                sample.input = extract_string("input");
                sample.context = extract_string("context");
                sample.answers = extract_string_array("answers");
                sample.dataset = extract_string("dataset");
                
                if (!sample.context.empty() && !sample.input.empty()) {
                    samples.push_back(sample);
                }
                obj_start = std::string::npos;
            }
        }
    }
    
    return samples;
}

static float compute_f1(const std::string& prediction, const std::vector<std::string>& answers) {
    if (answers.empty()) return 0.0f;
    
    // Tokenize prediction into words
    auto tokenize_words = [](const std::string& s) -> std::vector<std::string> {
        std::vector<std::string> words;
        std::string word;
        for (char c : s) {
            if (isalnum(c)) {
                word += tolower(c);
            } else if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        if (!word.empty()) words.push_back(word);
        return words;
    };
    
    auto pred_words = tokenize_words(prediction);
    if (pred_words.empty()) return 0.0f;
    
    float best_f1 = 0.0f;
    for (const auto& ans : answers) {
        auto ans_words = tokenize_words(ans);
        if (ans_words.empty()) continue;
        
        // Count common words
        int common = 0;
        for (const auto& pw : pred_words) {
            for (const auto& aw : ans_words) {
                if (pw == aw) { common++; break; }
            }
        }
        
        float precision = (float)common / pred_words.size();
        float recall = (float)common / ans_words.size();
        float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
        best_f1 = std::max(best_f1, f1);
    }
    
    return best_f1;
}

static void run_phase2_longbench(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & data_dir,
    int n_ctx,
    int n_early,
    const std::vector<float> & keep_ratios
) {
    log_info("\n=== Phase 2: LongBench Tasks ===\n");
    
    const std::vector<std::string> tasks = {"qasper", "hotpotqa", "trec"};
    
    log_result("{\n  \"phase\": \"longbench\",\n  \"n_early\": %d,\n  \"n_ctx\": %d,\n  \"tasks\": [\n",
               n_early, n_ctx);
    
    bool first_task = true;
    for (const auto& task : tasks) {
        std::string path = data_dir + "/" + task + ".json";
        auto samples = load_longbench_json(path);
        
        if (samples.empty()) {
            log_info("Skipping %s: no samples found at %s\n", task.c_str(), path.c_str());
            continue;
        }
        
        // Limit samples to fit in context
        std::vector<LongBenchSample> valid_samples;
        for (const auto& s : samples) {
            std::string prompt = s.context + "\n\nQuestion: " + s.input + "\nAnswer:";
            auto toks = tokenize(vocab, prompt, true);
            if ((int)toks.size() <= n_ctx - 64) {  // Leave room for response
                valid_samples.push_back(s);
                if (valid_samples.size() >= 10) break;  // Limit for speed
            }
        }
        
        if (valid_samples.empty()) {
            log_info("Skipping %s: all samples too long for n_ctx=%d\n", task.c_str(), n_ctx);
            continue;
        }
        
        log_info("Task: %s (%zu samples)\n", task.c_str(), valid_samples.size());
        
        if (!first_task) log_result(",\n");
        first_task = false;
        log_result("    {\"task\": \"%s\", \"n_samples\": %zu, \"results\": [\n", task.c_str(), valid_samples.size());
        
        bool first_kr = true;
        for (float kr : keep_ratios) {
            float total_f1 = 0.0f;
            int n_eval = 0;
            
            llama_self_layer_prefill_params params;
            params.n_early_layers = n_early;
            params.keep_ratio = kr;
            params.pool_kernel_size = 1;
            params.use_chunking = false;
            
            for (const auto& sample : valid_samples) {
                std::string prompt = sample.context + "\n\nQuestion: " + sample.input + "\nAnswer:";
                auto tokens = tokenize(vocab, prompt, true);
                
                int n_kept;
                if (kr >= 1.0f) {
                    // Baseline
                    llama_memory_clear(llama_get_memory(ctx), false);
                    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
                    for (size_t i = 0; i < tokens.size(); i++) {
                        batch.token[batch.n_tokens] = tokens[i];
                        batch.pos[batch.n_tokens] = (int)i;
                        batch.n_seq_id[batch.n_tokens] = 1;
                        batch.seq_id[batch.n_tokens][0] = 0;
                        batch.logits[batch.n_tokens] = (i == tokens.size() - 1);
                        batch.n_tokens++;
                    }
                    llama_decode(ctx, batch);
                    llama_batch_free(batch);
                    n_kept = (int)tokens.size();
                } else {
                    n_kept = llama_self_layer_prefill_partial(ctx, tokens.data(), (int)tokens.size(), &params);
                    if (n_kept < 0) continue;
                }
                
                // Generate response (greedy, max 64 tokens)
                std::string response = generate_response(ctx, vocab, n_kept, 64);
                
                // Compute F1
                float f1 = compute_f1(response, sample.answers);
                total_f1 += f1;
                n_eval++;
            }
            
            float avg_f1 = n_eval > 0 ? total_f1 / n_eval : 0.0f;
            log_info("  kr=%.2f: avg_f1=%.4f (%d samples)\n", kr, avg_f1, n_eval);
            
            if (!first_kr) log_result(",\n");
            first_kr = false;
            log_result("      {\"keep_ratio\": %.2f, \"avg_f1\": %.4f, \"n_eval\": %d}", kr, avg_f1, n_eval);
        }
        
        log_result("\n    ]}");
    }
    
    log_result("\n  ]\n}\n");
}

// ============================================================================
// Phase 5: Attention Score Analysis
// ============================================================================

static void run_phase5_attention(
    llama_context * ctx,
    const llama_vocab * vocab,
    int n_early,
    const std::vector<float> & keep_ratios
) {
    (void)keep_ratios;  // Used for analysis but not iterating over
    log_info("\n=== Phase 5: Attention Score Analysis ===\n");
    
    // Test with prompts that have clear important/unimportant tokens
    const std::string prompt = 
        "The answer to the question 'What is the capital of Japan?' is Tokyo. "
        "Some additional filler text that is not important for answering the question. "
        "More irrelevant details about random topics like weather and food. "
        "Remember: the question was about the capital of Japan.";
    
    auto tokens = tokenize(vocab, prompt, true);
    log_info("Test prompt: %zu tokens\n", tokens.size());
    
    llama_self_layer_prefill_params params;
    params.n_early_layers = n_early;
    params.pool_kernel_size = 1;
    params.use_chunking = false;
    
    log_result("{\n  \"phase\": \"attention_analysis\",\n  \"n_early\": %d,\n  \"n_tokens\": %d,\n",
               n_early, (int)tokens.size());
    
    // Get attention profiles
    std::vector<float> shallow, deep, full;
    float pearson_sf = 0, pearson_sd = 0;
    
    params.keep_ratio = 1.0f;  // Don't filter for analysis
    int r = llama_self_layer_prefill_attention_profiles(
        ctx, tokens.data(), (int)tokens.size(), &params,
        shallow, deep, full, &pearson_sf, &pearson_sd);
    
    if (r == 0 && !shallow.empty()) {
        log_info("Pearson(shallow, full): %.4f\n", pearson_sf);
        log_info("Pearson(shallow, deep): %.4f\n", pearson_sd);
        
        log_result("  \"pearson_shallow_full\": %.4f,\n  \"pearson_shallow_deep\": %.4f,\n",
                   pearson_sf, pearson_sd);
        
        // Show top-k important tokens according to shallow vs full
        std::vector<std::pair<float, int>> shallow_ranked, full_ranked;
        for (size_t i = 0; i < shallow.size(); i++) {
            shallow_ranked.push_back({shallow[i], (int)i});
            full_ranked.push_back({full[i], (int)i});
        }
        
        std::sort(shallow_ranked.rbegin(), shallow_ranked.rend());
        std::sort(full_ranked.rbegin(), full_ranked.rend());
        
        int k = std::min(10, (int)shallow.size());
        log_info("Top %d tokens by shallow attention:\n", k);
        log_result("  \"top_shallow\": [");
        for (int i = 0; i < k; i++) {
            int idx = shallow_ranked[i].second;
            char buf[64] = {0};
            int len = llama_token_to_piece(vocab, tokens[idx], buf, sizeof(buf) - 1, 0, false);
            if (len > 0) buf[len] = '\0';
            log_info("  [%d] \"%s\" (score=%.4f)\n", idx, buf, shallow_ranked[i].first);
            if (i > 0) log_result(", ");
            log_result("{\"idx\": %d, \"score\": %.4f}", idx, shallow_ranked[i].first);
        }
        log_result("],\n");
        
        log_info("Top %d tokens by full attention:\n", k);
        log_result("  \"top_full\": [");
        for (int i = 0; i < k; i++) {
            int idx = full_ranked[i].second;
            char buf[64] = {0};
            int len = llama_token_to_piece(vocab, tokens[idx], buf, sizeof(buf) - 1, 0, false);
            if (len > 0) buf[len] = '\0';
            log_info("  [%d] \"%s\" (score=%.4f)\n", idx, buf, full_ranked[i].first);
            if (i > 0) log_result(", ");
            log_result("{\"idx\": %d, \"score\": %.4f}", idx, full_ranked[i].first);
        }
        log_result("]\n");
    } else {
        log_info("Failed to get attention profiles\n");
        log_result("  \"error\": \"failed to get attention profiles\"\n");
    }
    
    log_result("}\n");
}

// ============================================================================
// Main
// ============================================================================

static void usage() {
    std::cerr
        << "llama-self-layer-prefill-quality\n"
        << "  --model <gguf>         Model file (required)\n"
        << "  --test-file <path>     Text file for perplexity test\n"
        << "  --longbench-dir <dir>  LongBench dataset directory (for Phase 2)\n"
        << "  --phase <1-5|all>      Which phase(s) to run (default: all)\n"
        << "  --n-ctx <n>            Context size (default: 2048)\n"
        << "  --n-early <n>          Shallow layers for scoring (default: L/4)\n"
        << "  --n-gpu-layers <n>     GPU layers (default: -1 = all)\n"
        << "  --multi-gpu            Use layer split across multiple GPUs\n"
        << "  --keep-ratios <list>   Comma-separated keep ratios (default: 0.10,0.25,0.50,1.00)\n";
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string test_file;
    std::string longbench_dir = "datasets/longbench/subset";
    std::string phase_str = "all";
    int n_ctx = 2048;
    int n_early = 0;
    int n_gpu_layers = -1;
    bool multi_gpu = false;
    std::vector<float> keep_ratios = {0.10f, 0.25f, 0.50f, 1.00f};
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--test-file" && i + 1 < argc) {
            test_file = argv[++i];
        } else if (arg == "--longbench-dir" && i + 1 < argc) {
            longbench_dir = argv[++i];
        } else if (arg == "--phase" && i + 1 < argc) {
            phase_str = argv[++i];
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--n-early" && i + 1 < argc) {
            n_early = std::stoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--multi-gpu") {
            multi_gpu = true;
        } else if (arg == "--keep-ratios" && i + 1 < argc) {
            keep_ratios.clear();
            std::string kr_str = argv[++i];
            std::stringstream ss(kr_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                keep_ratios.push_back(std::stof(item));
            }
        } else {
            usage();
            return 1;
        }
    }
    
    if (model_path.empty()) {
        usage();
        return 1;
    }
    
    // Load model
    log_info("Loading model: %s\n", model_path.c_str());
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (n_gpu_layers != 0) {
        if (multi_gpu) {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
            log_info("Using multi-GPU layer split\n");
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        }
    }
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = (int)llama_model_n_layer(model);
    if (n_early <= 0) n_early = std::max(1, n_layer / 4);
    
    log_info("Model loaded: %d layers, n_early=%d\n", n_layer, n_early);
    
    // Create context
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
    
    // Determine which phases to run
    bool run_all = (phase_str == "all");
    int phase_num = run_all ? 0 : std::stoi(phase_str);
    
    // Phase 1: Perplexity
    if (run_all || phase_num == 1) {
        std::string test_text;
        if (!test_file.empty()) {
            test_text = read_file(test_file);
        }
        if (test_text.empty()) {
            // Default test text - longer for meaningful perplexity evaluation
            test_text = 
                "The history of artificial intelligence began with ancient myths and legends about artificial beings. "
                "Modern AI research started in the 1950s when computer scientists began exploring whether machines could think. "
                "Alan Turing proposed the Turing test as a measure of machine intelligence in his famous 1950 paper. "
                "The term artificial intelligence was coined by John McCarthy at the Dartmouth conference in 1956. "
                "Early AI research was optimistic, with predictions that human-level AI was just around the corner. "
                "However, progress proved more difficult than expected, leading to periods of reduced funding called AI winters. "
                "Machine learning emerged as a practical approach to AI by allowing computers to learn from data. "
                "Neural networks, inspired by the structure of the brain, became a key technology in modern AI systems. "
                "Deep learning, using neural networks with many layers, achieved breakthroughs in image and speech recognition. "
                "The transformer architecture, introduced in 2017, revolutionized natural language processing. "
                "Large language models like GPT demonstrated remarkable capabilities in text generation and understanding. "
                "These models are trained on vast amounts of text data from the internet and books. "
                "They use attention mechanisms to understand relationships between words in a sequence. "
                "Pre-training on large datasets followed by fine-tuning has become a standard approach. "
                "The field continues to advance rapidly with new architectures and training techniques. "
                "Concerns about AI safety and alignment have become increasingly important research areas. "
                "Researchers work to ensure AI systems are beneficial and aligned with human values. "
                "Applications of AI now span healthcare, finance, transportation, and many other fields. "
                "The technology raises important questions about privacy, employment, and social impact. "
                "Despite challenges, AI holds tremendous potential to solve complex problems facing humanity.";
        }
        run_phase1_perplexity(ctx, vocab, test_text, n_ctx, n_early, keep_ratios);
    }
    
    // Phase 2: LongBench
    if (run_all || phase_num == 2) {
        run_phase2_longbench(ctx, vocab, longbench_dir, n_ctx, n_early, keep_ratios);
    }
    
    // Phase 3: PassKey
    if (run_all || phase_num == 3) {
        run_phase3_passkey(ctx, vocab, n_ctx, n_early, keep_ratios);
    }
    
    // Phase 4: Generation
    if (run_all || phase_num == 4) {
        run_phase4_generation(ctx, vocab, n_early, keep_ratios);
    }
    
    // Phase 5: Attention
    if (run_all || phase_num == 5) {
        run_phase5_attention(ctx, vocab, n_early, keep_ratios);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    log_info("\n=== Quality evaluation complete ===\n");
    return 0;
}
