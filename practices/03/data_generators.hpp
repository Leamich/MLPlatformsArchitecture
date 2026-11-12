
#pragma once

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

template <typename T> using Data = std::vector<T>;

template <typename T, typename Rng>
void fill_random_data(Rng &rng, Data<T> &data) {
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(0, 1.);
    for (auto &v : data) {
      v = dist(rng);
    }
  } else {
    std::uniform_int_distribution<T> dist(0, data.size());
    for (auto &v : data) {
      v = dist(rng);
    }
  }
}

template <typename T>
std::tuple<Data<T>, Data<T>> make_test_data(size_t seed, size_t num_elements) {
  std::mt19937 rng(seed);
  Data<T> x(num_elements);
  fill_random_data<T>(rng, x);
  Data<T> y(num_elements);
  fill_random_data<T>(rng, y);
  return {x, y};
}
