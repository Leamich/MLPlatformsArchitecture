#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "muladd.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
#include <hwy/per_target.h>

#include "muladd-inl.h"

#if HWY_ONCE
#include <oneapi/tbb.h>

#include "muladd.h"

namespace simd_math {

template <typename T>
void mul_add_impl(const T* a, const T* b, T* c, size_t n) {
    // Serial SIMD version (your TODO kernel in muladd-inl.h).
    // HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(mul_add<T>)
    // (a, b, c, n);

    // -----------------------------------------------------------------
    // TODO: the function above is memory-bound, so per-core
    // SIMD cannot beat DRAM bandwidth. If you want to go faster, parallelize
    // across cores. Split the range into grains that fit
    // in L1/L2 and run the SIMD kernel on each block:
    //
    constexpr size_t grain_size = (48 * 1024) / 2;  // L1 bytes / 2
    constexpr auto gs = grain_size / (3 * sizeof(T));
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, n, gs),
        [a, b, c](const oneapi::tbb::blocked_range<size_t>& r) {
            HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(mul_add<T>)
            (a + r.begin(), b + r.begin(), c + r.begin(), r.end() - r.begin());
        });
    // NOTE: that passes the wrong pointer (`a`) unless you offset by r.begin().
    // Make sure each block processes a[i + r.begin()] etc.
    // -----------------------------------------------------------------
}

void mul_add(const float* a, const float* b, float* c, size_t n) {
    mul_add_impl(a, b, c, n);
}

void mul_add(const double* a, const double* b, double* c, size_t n) {
    mul_add_impl(a, b, c, n);
}

}  // namespace simd_math
#endif  // HWY_ONCE