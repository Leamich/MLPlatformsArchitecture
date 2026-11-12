#include "bench_utils.hpp"
#include "data_generators.hpp"

#include "matmul.h"
#include <hwy/aligned_allocator.h>

#include <vector>

using namespace bench;

// The baseline is intentionally NOT auto-vectorized so you can see a clean
// scalar reference point. Your hand-written SIMD in matmul-inl.h must beat it.
// TODO: Also comment the __attribute__ to see if these loops can be
// auto-vectorized
__attribute__((optimize("no-tree-vectorize"))) static void
BaselineMatMul(benchmark::State &state) {
  auto n = state.range(0);
  auto [a, b] = make_test_data<float>(seed, n * n);
  std::vector<float> c(n * n, 0.f);

  for (auto _ : state) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < n; ++k) {
        const float a_ik = a[i * n + k];
        for (size_t j = 0; j < n; ++j) {
          c[i * n + j] += a_ik * b[k * n + j];
        }
      }
    }
  }
}
BENCHMARK(BaselineMatMul)->Apply(matmul_problems)->MinWarmUpTime(1.0);

static void SimdMatMul(benchmark::State &state) {
  auto n = state.range(0);
  auto a = hwy::AllocateAligned<float>(n * n);
  auto b = hwy::AllocateAligned<float>(n * n);
  auto c = hwy::AllocateAligned<float>(n * n);
  std::fill(c.get(), c.get() + n * n, 0.f);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0, 1.);
  for (int i = 0; i < n * n; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }

  for (auto _ : state) {
    simd_math::matmul(a.get(), b.get(), c.get(), n);
  }
}
BENCHMARK(SimdMatMul)
    ->Apply(matmul_problems)
    ->MinWarmUpTime(1.0)
    ->Unit(benchmark::kMicrosecond);