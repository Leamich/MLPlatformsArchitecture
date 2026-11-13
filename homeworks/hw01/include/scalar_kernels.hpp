#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fast_math.hpp"
#include "kernels.hpp"

// Reference implementation: plain loops (auto-vectorization is disabled in CMake).
template <typename T>
class ScalarKernels final : public IKernels<T> {
public:
    std::string name() const override { return "scalar"; }

    // int32 math is done in int64 and truncated back, so overflow wraps around
    // (same as SIMD) instead of being undefined behaviour.
    using Wide = std::conditional_t<std::is_integral_v<T>, std::int64_t, T>;

    void add(const T* a, const T* b, T* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = T(Wide(a[i]) + Wide(b[i]));
    }

    void sub(const T* a, const T* b, T* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = T(Wide(a[i]) - Wide(b[i]));
    }

    void mul(const T* a, const T* b, T* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = T(Wide(a[i]) * Wide(b[i]));
    }

    void div(const T* a, const T* b, T* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::is_integral_v<T> && b[i] == 0) throw std::domain_error("integer division by zero");
            out[i] = T(Wide(a[i]) / Wide(b[i]));
        }
    }

    Accum<T> sum(const T* a, std::size_t n) const override {
        Accum<T> s = 0;
        for (std::size_t i = 0; i < n; ++i) s += a[i];
        return s;
    }

    void silu(const T* a, ActOut<T>* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = fm::silu(ActOut<T>(a[i]));
    }

    void gelu(const T* a, ActOut<T>* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = fm::gelu(ActOut<T>(a[i]));
    }

    // i-k-j order: the inner loop walks rows of B and C, which is cache friendly.
    void matmul_rows(const T* a, const T* b, T* c, std::size_t K, std::size_t N,
                     std::size_t row_begin, std::size_t row_end) const override {
        std::fill(c + row_begin * N, c + row_end * N, T(0));
        for (std::size_t i = row_begin; i < row_end; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const T aik = a[i * K + k];
                for (std::size_t j = 0; j < N; ++j) c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
};
