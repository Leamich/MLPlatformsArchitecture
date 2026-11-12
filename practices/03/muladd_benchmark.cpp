#include <hwy/aligned_allocator.h>

#include "bench_utils.hpp"
#include "data_generators.hpp"
#include "muladd.h"

using namespace bench;

template <class T>
static void SimdMulAdd(benchmark::State& state) {
    auto N = state.range(0);
    auto a = hwy::AllocateAligned<T>(N);
    auto b = hwy::AllocateAligned<T>(N);
    auto result = hwy::AllocateAligned<T>(N);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0, 1.);
    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        result[i] = 3;
    }

    for (auto _ : state) {
        simd_math::mul_add(a.get(), b.get(), result.get(), N);
    }
}

BENCHMARK_TEMPLATE1(SimdMulAdd, float)
    ->Apply(problems)
    ->MinWarmUpTime(1.0)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE1(SimdMulAdd, double)
    ->Apply(problems)
    ->MinWarmUpTime(1.0)
    ->Unit(benchmark::kMicrosecond);
