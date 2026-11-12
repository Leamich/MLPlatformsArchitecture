#include "bench_utils.hpp"
#include "data_generators.hpp"

using namespace bench;

template <class T>
// TODO: uncomment and compare performance
// __attribute__((optimize("no-tree-vectorize")))
static void BaselineMulAdd(benchmark::State& state) {
    auto N = state.range(0);
    auto [a, b] = make_test_data<T>(seed, N);
    std::vector<T> result(N, 3);

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            result[i] = a[i] * result[i] + b[i];
        }
    }
}

BENCHMARK_TEMPLATE1(BaselineMulAdd, float)
    ->Apply(problems)
    ->MinWarmUpTime(1.0)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE1(BaselineMulAdd, double)
    ->Apply(problems)
    ->MinWarmUpTime(1.0)
    ->Unit(benchmark::kMicrosecond);
