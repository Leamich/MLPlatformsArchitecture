#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "array.hpp"
#include "executor.hpp"
#include "kernels.hpp"

// Public API: checks shapes, allocates the result, splits the work into chunks.
// Kernels do the math on a chunk, the executor runs the chunks.
//
//   ScalarKernels<float> k;  TbbExecutor ex(4);
//   Engine<float> engine(k, ex);
//   Array<float> c = engine.add(a, b);
template <typename T>
class Engine {
public:
    static constexpr std::size_t kDefaultGrain = 1 << 14;  // elements per chunk

    Engine(const IKernels<T>& kernels, const IExecutor& executor, std::size_t grain = kDefaultGrain)
        : kernels_(kernels), executor_(executor), grain_(std::max<std::size_t>(grain, 1)) {}

    std::string name() const { return kernels_.name() + "/" + executor_.name(); }

    Array<T> add(const Array<T>& a, const Array<T>& b) const { return binary(a, b, &IKernels<T>::add); }
    Array<T> sub(const Array<T>& a, const Array<T>& b) const { return binary(a, b, &IKernels<T>::sub); }
    Array<T> mul(const Array<T>& a, const Array<T>& b) const { return binary(a, b, &IKernels<T>::mul); }
    Array<T> div(const Array<T>& a, const Array<T>& b) const { return binary(a, b, &IKernels<T>::div); }

    // Each chunk writes its own partial sum, then they are added in order.
    // Chunks don't depend on the thread count, so the result is always the same.
    Accum<T> sum(const Array<T>& a) const {
        const std::size_t n = a.size();
        std::vector<Accum<T>> partial((n + grain_ - 1) / grain_);
        for_chunks(n, grain_, [&](std::size_t begin, std::size_t end) {
            partial[begin / grain_] = kernels_.sum(a.data() + begin, end - begin);
        });
        Accum<T> total = 0;
        for (Accum<T> p : partial) total += p;
        return total;
    }

    double mean(const Array<T>& a) const {
        if (a.empty()) throw std::invalid_argument("mean of an empty array");
        return double(sum(a)) / double(a.size());
    }

    Array<ActOut<T>> silu(const Array<T>& a) const { return activation(a, &IKernels<T>::silu); }
    Array<ActOut<T>> gelu(const Array<T>& a) const { return activation(a, &IKernels<T>::gelu); }

    // Parallel over rows of the result, about grain_ output elements per chunk.
    Array<T> matmul(const Array<T>& a, const Array<T>& b) const {
        if (a.cols() != b.rows()) throw std::invalid_argument("matmul: a.cols() != b.rows()");
        const std::size_t M = a.rows(), K = a.cols(), N = b.cols();

        auto out = Array<T>::uninitialized(Shape{M, N});
        if (out.empty()) return out;

        const std::size_t rows_per_chunk = std::max<std::size_t>(1, grain_ / N);
        for_chunks(M, rows_per_chunk, [&](std::size_t row_begin, std::size_t row_end) {
            kernels_.matmul_rows(a.data(), b.data(), out.data(), K, N, row_begin, row_end);
        });
        return out;
    }

private:
    using BinaryOp = void (IKernels<T>::*)(const T*, const T*, T*, std::size_t) const;
    using ActivationOp = void (IKernels<T>::*)(const T*, ActOut<T>*, std::size_t) const;

    Array<T> binary(const Array<T>& a, const Array<T>& b, BinaryOp op) const {
        if (a.shape() != b.shape()) throw std::invalid_argument("element-wise op: shape mismatch");
        auto out = Array<T>::uninitialized(a.shape());
        for_chunks(a.size(), grain_, [&](std::size_t begin, std::size_t end) {
            (kernels_.*op)(a.data() + begin, b.data() + begin, out.data() + begin, end - begin);
        });
        return out;
    }

    Array<ActOut<T>> activation(const Array<T>& a, ActivationOp op) const {
        auto out = Array<ActOut<T>>::uninitialized(a.shape());
        for_chunks(a.size(), grain_, [&](std::size_t begin, std::size_t end) {
            (kernels_.*op)(a.data() + begin, out.data() + begin, end - begin);
        });
        return out;
    }

    // Runs fn(begin, end) for chunks [0, grain), [grain, 2 * grain), ... of [0, n).
    template <typename F>
    void for_chunks(std::size_t n, std::size_t grain, F&& fn) const {
        const std::size_t num_chunks = (n + grain - 1) / grain;
        executor_.run(num_chunks, [&](std::size_t i) {
            const std::size_t begin = i * grain;
            fn(begin, std::min(n, begin + grain));
        });
    }

    const IKernels<T>& kernels_;
    const IExecutor& executor_;
    std::size_t grain_;
};
