#pragma once
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "array.hpp"
#include "engine.hpp"
#include "executor.hpp"
#include "scalar_kernels.hpp"
#include "simd_kernels.hpp"

using TestTypes = ::testing::Types<int32_t, float, double>;

// Sizes around SIMD widths catch tail bugs; 100003 is prime -> uneven chunks.
inline const std::vector<std::size_t> kSizes = {0, 1, 3, 7, 8, 9, 17, 33, 1023, 100'003};

inline std::ostream& operator<<(std::ostream& os, const Shape& s) {
    return os << "{" << s.rows << ", " << s.cols << "}";
}

template <typename T>
Array<T> random_array(Shape shape, std::uint32_t seed, T lo = T(-100), T hi = T(100)) {
    std::mt19937 gen(seed);
    auto out = Array<T>::uninitialized(shape);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(lo, hi);
        for (T& x : out) x = dist(gen);
    } else {
        std::uniform_real_distribution<T> dist(lo, hi);
        for (T& x : out) x = dist(gen);
    }
    return out;
}

// |actual - expected| <= atol + rtol * |expected|; NaN == NaN. Reports only the first mismatch.
template <typename T>
void expect_array_near(const Array<T>& actual, const Array<T>& expected, double rtol = 0, double atol = 0) {
    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double got = actual[i], exp = expected[i];
        const bool ok = (std::isnan(got) && std::isnan(exp)) || got == exp ||
                        std::abs(got - exp) <= atol + rtol * std::abs(exp);
        if (!ok) {
            ADD_FAILURE() << "index " << i << ": got " << got << ", expected " << exp;
            return;
        }
    }
}

template <typename T>
struct Backend {
    std::string name;
    std::unique_ptr<IKernels<T>> kernels;
    std::unique_ptr<IExecutor> executor;
    std::unique_ptr<Engine<T>> engine;
};

// [0] is scalar/seq = the reference.
// Small grain so that even small arrays are split into many chunks.
template <typename T>
std::vector<Backend<T>> make_backends(std::size_t grain = 64) {
    std::vector<Backend<T>> out;
    auto add = [&](std::unique_ptr<IKernels<T>> k, std::unique_ptr<IExecutor> e) {
        Backend<T> b;
        b.kernels = std::move(k);
        b.executor = std::move(e);
        b.engine = std::make_unique<Engine<T>>(*b.kernels, *b.executor, grain);
        b.name = b.engine->name();
        out.push_back(std::move(b));
    };
    add(std::make_unique<ScalarKernels<T>>(), std::make_unique<SequentialExecutor>());
    add(std::make_unique<ScalarKernels<T>>(), std::make_unique<TbbExecutor>(4));
    add(std::make_unique<SimdKernels<T>>(),   std::make_unique<SequentialExecutor>());
    add(std::make_unique<SimdKernels<T>>(),   std::make_unique<TbbExecutor>(4));
    return out;
}
