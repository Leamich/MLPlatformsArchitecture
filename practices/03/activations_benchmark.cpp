#include <benchmark/benchmark.h>
#include <immintrin.h>

#include <algorithm>
#include <random>
#include <vector>

constexpr size_t seed = 42;
constexpr size_t n = 1024;
std::vector<float> vec(n);

void fill_vector(std::vector<float>& target) {
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::generate(target.begin(), target.end(),
                  [&]() { return distribution(generator); });
}

static void do_setup([[maybe_unused]] const benchmark::State& state) {
    fill_vector(vec);
}

struct sigmoid {
    float operator()(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

// A SIMD-friendly linear loop: no branch, no recursion, element-independant.
// The compiler SHOULD auto-vectorize this one.
static void sigmoid_vectorized(benchmark::State& state) {
    sigmoid activation;
    for (auto _ : state) {
        // #pragma clang loop vectorize(enable)
        for (auto& v : vec) {
            v = activation(v);
        }
    }
}
BENCHMARK(sigmoid_vectorized)->Setup(do_setup);

// Same linear loop, but WITHOUT auto-vectorization.
__attribute__((optimize("no-tree-vectorize"))) static void
sigmoid_non_vectorized(benchmark::State& state) {
    sigmoid activation;
    for (auto _ : state) {
        // #pragma clang loop vectorize(disable)
        for (auto& v : vec) {
            v = activation(v);
        }
    }
}
BENCHMARK(sigmoid_non_vectorized)->Setup(do_setup);

// Manual Sigmoid implementation
void simd_sigmoid(float* data, size_t n) {
    // Chebyshev polynomial coefficients for a rational approximation of
    // sigmoid.
    const __m256 c0 = _mm256_set1_ps(0.00949107017368078f);
    const __m256 c1 = _mm256_set1_ps(0.0654858946800232f);
    const __m256 c2 = _mm256_set1_ps(0.231547489762306f - 0.00949107017368078f);
    const __m256 c3 = _mm256_set1_ps(0.530778527259827f);
    const __m256 c4 = _mm256_set1_ps(0.855334937572479f);
    const __m256 c5 = _mm256_set1_ps(0.500000894069672f);

    const __m256 d0 = _mm256_set1_ps(0.130970627069473f);
    const __m256 d1 = _mm256_set1_ps(3.99615288415589e-07f);
    const __m256 d2 = _mm256_set1_ps(1.06155431270599f - 0.130970627069473f);
    const __m256 d3 = _mm256_set1_ps(1.35144250634767e-06f);
    const __m256 d4 = _mm256_set1_ps(1.0f);

    const __m256 invtwelve = _mm256_set1_ps(1.0f / 12.0f);
    const __m256 lo = _mm256_set1_ps(-11.5f);
    const __m256 hi = _mm256_set1_ps(11.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        auto f = _mm256_min_ps(hi, _mm256_max_ps(lo, v));
        f = _mm256_mul_ps(f, invtwelve);
        auto f2 = _mm256_mul_ps(f, f);

        auto a1 = _mm256_fmadd_ps(f2, c0, c1);
        auto a2 = _mm256_fmadd_ps(f2, a1, c2);
        auto a3 = _mm256_sub_ps(_mm256_fmadd_ps(f2, a2, c3), a1);
        auto a4 = _mm256_sub_ps(_mm256_fmadd_ps(f2, a3, c4), a2);
        auto f0 = _mm256_sub_ps(_mm256_fmadd_ps(f, a4, c5), a3);
        auto b1 = _mm256_fmadd_ps(f2, d0, d1);
        auto b2 = _mm256_fmadd_ps(f2, b1, d2);
        auto b3 = _mm256_sub_ps(_mm256_fmadd_ps(f2, b2, d3), b1);
        auto f1 = _mm256_sub_ps(_mm256_fmadd_ps(f, b3, d4), b2);
        _mm256_storeu_ps(&data[i], _mm256_div_ps(f0, f1));
    }
}

static void sigmoid_vectorized_manually(benchmark::State& state) {
    for (auto _ : state) {
        simd_sigmoid(vec.data(), vec.size());
    }
}
BENCHMARK(sigmoid_vectorized_manually)->Setup(do_setup);

static void sigmoid_data_dependent(benchmark::State& state) {
    sigmoid activation;

    for (auto _ : state) {
        float prev = 0;
        for (auto& v : vec) {
            // IF DEPENDENCE
            if (prev > 1.2f)
                v = activation(v);
            else
                v = activation(v) + 1;

            // PREV DEPENDENCE
            // v = activation(v) + prev;

            prev = v;
        }
    }
}
BENCHMARK(sigmoid_data_dependent)->Setup(do_setup);

struct relu {
    float operator()(float x) { return std::max(0.f, x); }
};

__attribute__((optimize("no-tree-vectorize"))) static void relu_non_vectorized(
    benchmark::State& state) {
    relu activation;
    for (auto _ : state) {
        // #pragma clang loop vectorize(disable)
        for (auto& v : vec) {
            v = activation(v);
        }
    }
}
BENCHMARK(relu_non_vectorized)->Setup(do_setup);

static void relu_vectorized(benchmark::State& state) {
    relu activation;
    for (auto _ : state) {
        for (auto& v : vec) {
            v = activation(v);
        }
    }
}
BENCHMARK(relu_vectorized)->Setup(do_setup);

static void relu_data_dependent(benchmark::State& state) {
    for (auto _ : state) {
        float prev = 0;
        for (auto& v : vec) {
            v = std::max(0.f, v) + prev;
            prev = v;
        }
    }
}
BENCHMARK(relu_data_dependent)->Setup(do_setup);

void simd_relu(float* data, size_t n) {
    // TODO: implement ReLU with SIMD. relu(x) = max(x, 0). Use:
    // _mm256_loadu_ps
    // _mm256_set1_ps
    // _mm256_storeu_ps
    // _mm256_max_ps

    __m256 thr = _mm256_set1_ps(0.0f);
    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_max_ps(thr, _mm256_loadu_ps(&data[i]));
        _mm256_storeu_ps(&data[i], v);
    }
}

static void relu_vectorized_manually(benchmark::State& state) {
    for (auto _ : state) {
        simd_relu(vec.data(), vec.size());
    }
}
BENCHMARK(relu_vectorized_manually)->Setup(do_setup);