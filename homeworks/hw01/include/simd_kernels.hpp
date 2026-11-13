#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

#include "fast_math.hpp"
#include "kernels.hpp"
#include "scalar_kernels.hpp"

namespace hn = hwy::HWY_NAMESPACE;

// Google Highway, static dispatch (target chosen by -march=native).
// Every method processes full vectors, then passes the leftover tail (< 1 vector)
// to the scalar kernel.
template <typename T>
class SimdKernels final : public IKernels<T> {
public:
    std::string name() const override { return std::string("simd-") + hwy::TargetName(HWY_TARGET); }

    void add(const T* a, const T* b, T* out, std::size_t n) const override {
        const std::size_t i = map2(a, b, out, n, [](auto x, auto y) { return hn::Add(x, y); });
        scalar_.add(a + i, b + i, out + i, n - i);
    }

    void sub(const T* a, const T* b, T* out, std::size_t n) const override {
        const std::size_t i = map2(a, b, out, n, [](auto x, auto y) { return hn::Sub(x, y); });
        scalar_.sub(a + i, b + i, out + i, n - i);
    }

    void mul(const T* a, const T* b, T* out, std::size_t n) const override {
        const std::size_t i = map2(a, b, out, n, [](auto x, auto y) { return hn::Mul(x, y); });
        scalar_.mul(a + i, b + i, out + i, n - i);
    }

    // x86 has no int32 SIMD division, and we need the zero check -> scalar for int32.
    void div(const T* a, const T* b, T* out, std::size_t n) const override {
        std::size_t i = 0;
        if constexpr (std::is_floating_point_v<T>) {
            i = map2(a, b, out, n, [](auto x, auto y) { return hn::Div(x, y); });
        }
        scalar_.div(a + i, b + i, out + i, n - i);
    }

    Accum<T> sum(const T* a, std::size_t n) const override {
        const hn::ScalableTag<T> d;
        const std::size_t L = hn::Lanes(d);
        std::size_t i = 0;
        Accum<T> s = 0;

        if constexpr (std::is_same_v<T, std::int32_t>) {
            // widen int32 lanes to int64 so the sum can't overflow
            const hn::Repartition<std::int64_t, decltype(d)> d64;
            auto acc = hn::Zero(d64);
            for (; i + L <= n; i += L) {
                const auto v = hn::LoadU(d, a + i);
                acc = hn::Add(acc, hn::PromoteLowerTo(d64, v));
                acc = hn::Add(acc, hn::PromoteUpperTo(d64, v));
            }
            s = hn::ReduceSum(d64, acc);
        } else {
            auto acc = hn::Zero(d);
            for (; i + L <= n; i += L) acc = hn::Add(acc, hn::LoadU(d, a + i));
            s = hn::ReduceSum(d, acc);
        }
        return s + scalar_.sum(a + i, n - i);
    }

    void silu(const T* a, ActOut<T>* out, std::size_t n) const override {
        const std::size_t i = map1(a, out, n, [](auto d, auto x) { return silu_vec(d, x); });
        scalar_.silu(a + i, out + i, n - i);
    }

    void gelu(const T* a, ActOut<T>* out, std::size_t n) const override {
        const std::size_t i = map1(a, out, n, [](auto d, auto x) { return gelu_vec(d, x); });
        scalar_.gelu(a + i, out + i, n - i);
    }

    // Same i-k-j loops as ScalarKernels, but the inner loop does one vector of C at a time:
    // c[i][j..j+L) += a[i][k] * b[k][j..j+L)
    void matmul_rows(const T* a, const T* b, T* c, std::size_t K, std::size_t N,
                     std::size_t row_begin, std::size_t row_end) const override {
        const hn::ScalableTag<T> d;
        const std::size_t L = hn::Lanes(d);
        std::fill(c + row_begin * N, c + row_end * N, T(0));
        for (std::size_t i = row_begin; i < row_end; ++i) {
            T* c_row = c + i * N;
            for (std::size_t k = 0; k < K; ++k) {
                const T aik = a[i * K + k];
                const T* b_row = b + k * N;
                const auto va = hn::Set(d, aik);
                std::size_t j = 0;
                for (; j + L <= N; j += L) {
                    const auto vc = hn::Add(hn::LoadU(d, c_row + j), hn::Mul(va, hn::LoadU(d, b_row + j)));
                    hn::StoreU(vc, d, c_row + j);
                }
                for (; j < N; ++j) c_row[j] += aik * b_row[j];
            }
        }
    }

private:
    // out[i] = op(a[i], b[i]) for full vectors. Returns how many elements were done.
    template <typename Op>
    static std::size_t map2(const T* a, const T* b, T* out, std::size_t n, Op op) {
        const hn::ScalableTag<T> d;
        const std::size_t L = hn::Lanes(d);
        std::size_t i = 0;
        for (; i + L <= n; i += L) {
            hn::StoreU(op(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        return i;
    }

    // out[i] = f(a[i]) for full vectors. Returns how many elements were done.
    template <typename Op>
    static std::size_t map1(const T* a, ActOut<T>* out, std::size_t n, Op f) {
        const hn::ScalableTag<ActOut<T>> df;
        const std::size_t L = hn::Lanes(df);
        std::size_t i = 0;
        for (; i + L <= n; i += L) hn::StoreU(f(df, load_as_float(a + i)), df, out + i);
        return i;
    }

    // Loads one vector; int32 lanes are converted to float.
    static auto load_as_float(const T* p) {
        const hn::ScalableTag<ActOut<T>> df;
        if constexpr (std::is_integral_v<T>) {
            const hn::Rebind<T, hn::ScalableTag<ActOut<T>>> di;  // same lane count as df
            return hn::ConvertTo(df, hn::LoadU(di, p));
        } else {
            return hn::LoadU(df, p);
        }
    }

    // Same steps as fm::silu
    template <class D, class V>
    static V silu_vec(D d, V x) {
        using F = hn::TFromD<D>;
        const V limit = hn::Set(d, F(fm::kExpLimit));
        const V one = hn::Set(d, F(1));
        const V t = hn::Min(hn::Max(hn::Neg(x), hn::Neg(limit)), limit);
        const V sigmoid = hn::Div(one, hn::Add(one, hn::Exp(d, t)));
        return hn::IfThenZeroElse(hn::Lt(x, hn::Neg(limit)), hn::Mul(x, sigmoid));
    }

    // Same steps as fm::gelu
    template <class D, class V>
    static V gelu_vec(D d, V x) {
        using F = hn::TFromD<D>;
        const V limit = hn::Set(d, F(fm::kTanhLimit));
        const V x3 = hn::Mul(hn::Mul(x, x), x);
        const V u = hn::Mul(hn::Set(d, F(fm::kSqrt2OverPi)), hn::MulAdd(hn::Set(d, F(fm::kGeluCubic)), x3, x));
        const V t = hn::Tanh(d, hn::Min(hn::Max(u, hn::Neg(limit)), limit));
        const V result = hn::Mul(hn::Mul(hn::Set(d, F(0.5)), x), hn::Add(hn::Set(d, F(1)), t));
        return hn::IfThenZeroElse(hn::Lt(x, hn::Neg(limit)), result);
    }

    ScalarKernels<T> scalar_;
};
