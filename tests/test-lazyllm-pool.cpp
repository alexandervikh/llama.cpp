// tests/test-lazyllm-pool.cpp
// Unit tests for llama_lazyllm_apply_pooling — no model required.
// arXiv:2407.14057 §3.2 average-pool smoothing.
#include "llama-lazyllm.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static void test_pool_identity() {
    // k=1 → no change
    std::vector<float> s = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> orig = s;
    llama_lazyllm_apply_pooling(s, 1);
    for (size_t i = 0; i < s.size(); i++) {
        assert(float_eq(s[i], orig[i]));
    }
    printf("PASS: test_pool_identity\n");
}

static void test_pool_k3() {
    // k=3, half=1
    // s = {1, 2, 3, 4, 5}
    // i=0: avg(s[0..1])   = avg(1,2)     = 1.5
    // i=1: avg(s[0..2])   = avg(1,2,3)   = 2.0
    // i=2: avg(s[1..3])   = avg(2,3,4)   = 3.0
    // i=3: avg(s[2..4])   = avg(3,4,5)   = 4.0
    // i=4: avg(s[3..4])   = avg(4,5)     = 4.5
    std::vector<float> s = {1.f, 2.f, 3.f, 4.f, 5.f};
    llama_lazyllm_apply_pooling(s, 3);
    assert(float_eq(s[0], 1.5f));
    assert(float_eq(s[1], 2.0f));
    assert(float_eq(s[2], 3.0f));
    assert(float_eq(s[3], 4.0f));
    assert(float_eq(s[4], 4.5f));
    printf("PASS: test_pool_k3\n");
}

static void test_pool_empty() {
    std::vector<float> s;
    llama_lazyllm_apply_pooling(s, 13);
    assert(s.empty());
    printf("PASS: test_pool_empty\n");
}

static void test_pool_single() {
    // Single element — any kernel size → unchanged.
    std::vector<float> s = {42.f};
    llama_lazyllm_apply_pooling(s, 13);
    assert(float_eq(s[0], 42.f));
    printf("PASS: test_pool_single\n");
}

static void test_pool_uniform() {
    // Uniform scores — pooling of any size should give same value.
    std::vector<float> s(20, 1.0f);
    llama_lazyllm_apply_pooling(s, 13);
    for (float v : s) {
        assert(float_eq(v, 1.0f));
    }
    printf("PASS: test_pool_uniform\n");
}

int main() {
    test_pool_identity();
    test_pool_k3();
    test_pool_empty();
    test_pool_single();
    test_pool_uniform();
    printf("ALL PASS: test-lazyllm-pool\n");
    return 0;
}
