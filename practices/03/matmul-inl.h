#if defined(MATMUL_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef MATMUL_INL_H_
#undef MATMUL_INL_H_
#else
#define MATMUL_INL_H_
#endif

// It is fine to #include normal or *-inl headers.
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace simd_math {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// C = A * B, one output row of C at a time via rank-1 updates.
// For each (i, k): C[i][*] += A[i][k] * B[k][*].
//
// Lesson: unlike mul_add, matmul is ARITHMETIC-INTENSIVE. It does O(N^3)
// fused multiply-adds but only touches O(N^2) data. That high arithmetic
// intensity (FLOPs per byte) is exactly what lets SIMD show a big win:
// every A[i][k] value is reused across all n columns of B, so the vector
// register stays busy doing math instead of waiting on memory.
//
// The version below is the plain SCALAR triple loop. It is correct but slow,
// and is compiled with auto-vectorization disabled so it is a clean scalar
// reference. Your task is to replace the innermost j-loop with a SIMD
// rank-1 update (see TODO below) that clearly beats this scalar baseline.
template <typename T>
__attribute__((optimize("no-tree-vectorize"))) HWY_MAYBE_UNUSED void matmul(
    const T* const HWY_RESTRICT a, const T* const HWY_RESTRICT b,
    T* const HWY_RESTRICT c, const size_t n) {
    using D = hn::ScalableTag<T>;
    constexpr D d;
    const size_t lanes = hn::Lanes(d);
    // Scaffolding: `d`/`lanes` are unused until the SIMD TODO below is done.
    // (void)d;
    // (void)lanes;
    // (Zero the C output first so the FMA accumulate in the SIMD version
    //  starts from 0.)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) c[i * n + j] = 0;

    for (size_t i = 0; i < n; ++i) {      // row of C
        for (size_t k = 0; k < n; ++k) {  // shared index (col of A / row of B)
            const T a_ik = a[i * n + k];  // Use hn::Set(d, a_ik
            const T* HWY_RESTRICT brow = b + k * n;  // row k of B
            T* HWY_RESTRICT crow = c + i * n;        // row i of C
            // for (size_t j = 0; j < n; ++j) {  // Use steps size equals Lanes
            for (size_t j = 0; j < n;
                 j += lanes) {  // Use steps size equals Lanes
                // TODO: SIMD rank-1 update in place of this scalar line.
                auto acc = hn::LoadU(d, crow + j);
                auto b_row = hn::LoadU(d, brow + j);
                acc = hn::MulAdd(hn::Set(d, a_ik), b_row, acc);
                hn::StoreU(acc, d, crow + j);
                // Remember: fill `crow` in chunks of `lanes` and handle the
                // scalar TAIL when (n % lanes) != 0. Thread-safe here because
                // each (i) row of C is written by exactly one i iteration.
                // crow[j] += a_ik * brow[j];
            }
            if (n % lanes != 0) {
                const size_t rem = n % lanes;
                auto acc = hn::LoadN(d, crow + (n - rem), rem);
                auto b_row = hn::LoadN(d, brow + (n - rem), rem);
                acc = hn::MulAdd(hn::Set(d, a_ik), b_row, acc);
                hn::StoreN(acc, d, crow + (n - rem), rem);
            }
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace simd_math
HWY_AFTER_NAMESPACE();
#endif