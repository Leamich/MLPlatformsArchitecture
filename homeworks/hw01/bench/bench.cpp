#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <random>
#include <thread>

#include "array.hpp"
#include "engine.hpp"
#include "executor.hpp"
#include "scalar_kernels.hpp"
#include "simd_kernels.hpp"

// Every benchmark has Args {size, threads}; threads = 0 means SequentialExecutor.
// Results are reported as items/s (elements, or FLOP for matmul).

template <typename T>
Array<T> random_array(Shape shape) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(1, 10);  // positive -> valid divisor
    auto out = Array<T>::uninitialized(shape);
    for (T& x : out) x = T(dist(gen));
    return out;
}

// Times op(engine), where `items` is the work done per call.
template <typename T, typename Kernels, typename Op>
void run(benchmark::State& state, double items, Op op) {
    const int threads = static_cast<int>(state.range(1));
    std::unique_ptr<IExecutor> executor;
    if (threads == 0) {
        executor = std::make_unique<SequentialExecutor>();
    } else {
        executor = std::make_unique<TbbExecutor>(threads);
    }
    Kernels kernels;
    Engine<T> engine(kernels, *executor);

    for (auto _ : state) {
        auto result = op(engine);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * items));
    state.SetLabel(engine.name());
}

template <typename T, typename Kernels>
void BM_Add(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n}), b = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.add(a, b); });
}

template <typename T, typename Kernels>
void BM_Sub(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n}), b = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.sub(a, b); });
}

template <typename T, typename Kernels>
void BM_Mul(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n}), b = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.mul(a, b); });
}

template <typename T, typename Kernels>
void BM_Div(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n}), b = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.div(a, b); });
}

template <typename T, typename Kernels>
void BM_Sum(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.sum(a); });
}

template <typename T, typename Kernels>
void BM_Mean(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.mean(a); });
}

template <typename T, typename Kernels>
void BM_Silu(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.silu(a); });
}

template <typename T, typename Kernels>
void BM_Gelu(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({1, n});
    run<T, Kernels>(state, n, [&](const Engine<T>& e) { return e.gelu(a); });
}

// N x N matrices, 2 * N^3 FLOP
template <typename T, typename Kernels>
void BM_Matmul(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_array<T>({n, n}), b = random_array<T>({n, n});
    run<T, Kernels>(state, 2.0 * n * n * n, [&](const Engine<T>& e) { return e.matmul(a, b); });
}

void add_args(benchmark::internal::Benchmark* b, std::initializer_list<std::int64_t> sizes) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    for (std::int64_t n : sizes) {
        b->Args({n, 0});
        for (int t = 1; t < hw; t *= 2) b->Args({n, t});
        b->Args({n, hw});
    }
    b->ArgNames({"n", "threads"});
}

void VectorArgs(benchmark::internal::Benchmark* b) { add_args(b, {1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26}); }
void MatrixArgs(benchmark::internal::Benchmark* b) { add_args(b, {64, 256, 1024}); }

// Registers BM for all 3 types x 2 kernel sets.
#define REGISTER_ALL(BM, ARGS)                                                           \
    BENCHMARK_TEMPLATE(BM, int32_t, ScalarKernels<int32_t>)->Apply(ARGS)->UseRealTime(); \
    BENCHMARK_TEMPLATE(BM, int32_t, SimdKernels<int32_t>)->Apply(ARGS)->UseRealTime();   \
    BENCHMARK_TEMPLATE(BM, float, ScalarKernels<float>)->Apply(ARGS)->UseRealTime();     \
    BENCHMARK_TEMPLATE(BM, float, SimdKernels<float>)->Apply(ARGS)->UseRealTime();       \
    BENCHMARK_TEMPLATE(BM, double, ScalarKernels<double>)->Apply(ARGS)->UseRealTime();   \
    BENCHMARK_TEMPLATE(BM, double, SimdKernels<double>)->Apply(ARGS)->UseRealTime()

REGISTER_ALL(BM_Add, VectorArgs);
REGISTER_ALL(BM_Sub, VectorArgs);
REGISTER_ALL(BM_Mul, VectorArgs);
REGISTER_ALL(BM_Div, VectorArgs);
REGISTER_ALL(BM_Sum, VectorArgs);
REGISTER_ALL(BM_Mean, VectorArgs);
REGISTER_ALL(BM_Silu, VectorArgs);
REGISTER_ALL(BM_Gelu, VectorArgs);
REGISTER_ALL(BM_Matmul, MatrixArgs);
