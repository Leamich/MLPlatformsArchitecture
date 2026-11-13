#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// sum(int32) is accumulated in int64
template <typename T> struct AccumType          { using type = T; };
template <>           struct AccumType<int32_t> { using type = int64_t; };
template <typename T> using Accum = typename AccumType<T>::type;

// silu/gelu(int32) returns float
template <typename T> struct ActivationType          { using type = T; };
template <>           struct ActivationType<int32_t> { using type = float; };
template <typename T> using ActOut = typename ActivationType<T>::type;

// Kernels work on one contiguous chunk and know nothing about threads.
template <typename T>
class IKernels {
public:
    virtual ~IKernels() = default;

    virtual std::string name() const = 0;

    virtual void add(const T* a, const T* b, T* out, std::size_t n) const = 0;
    virtual void sub(const T* a, const T* b, T* out, std::size_t n) const = 0;
    virtual void mul(const T* a, const T* b, T* out, std::size_t n) const = 0;
    virtual void div(const T* a, const T* b, T* out, std::size_t n) const = 0;

    virtual Accum<T> sum(const T* a, std::size_t n) const = 0;

    virtual void silu(const T* a, ActOut<T>* out, std::size_t n) const = 0;
    virtual void gelu(const T* a, ActOut<T>* out, std::size_t n) const = 0;

    // Computes rows [row_begin, row_end) of C = A (M x K) * B (K x N).
    // a, b, c point to the start of the full matrices; the rows of c are overwritten.
    virtual void matmul_rows(const T* a, const T* b, T* c,
                             std::size_t K, std::size_t N,
                             std::size_t row_begin, std::size_t row_end) const = 0;
};
