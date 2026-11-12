#pragma once
#include <benchmark/benchmark.h>

namespace bench {

constexpr size_t seed = 42;

inline void problems(benchmark::Benchmark *b) {
  const std::vector<long> problem_size = {1024, 4096, 1024 * 1024,
                                          1024 * 1024 * 3};
  for (auto size : problem_size) {
    b->Args({size});
  }
}

// Sizes for dense matrix mult: N x N matrix, O(N^3) flops and O(N^2) data.
inline void matmul_problems(benchmark::Benchmark *b) {
  const std::vector<long> matrix_size = {64, 128, 256, 512};
  for (auto size : matrix_size) {
    b->Args({size});
  }
}
} // namespace bench