#if defined(MULADD_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef MULADD_INL_H_
#undef MULADD_INL_H_
#else
#define MULADD_INL_H_
#endif

// It is fine to #include normal or *-inl headers.
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace simd_math {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_MAYBE_UNUSED void mul_add(const T* HWY_RESTRICT mul_array,
                              const T* HWY_RESTRICT add_array,
                              T* HWY_RESTRICT out_array, const size_t size) {
    using D = hn::ScalableTag<T>;
    constexpr D d;

    // TODO: implement the SIMD kernel and delete this scalar based
    // implementation.
    // Use:  hn::Load, hn:MulAdd, hn::Store
    const size_t lanes = hn::Lanes(d);
    for (size_t i = 0; i < size; i+=lanes) {
        auto mul_a = hn::Load(d, &mul_array[i]);
        auto add_a = hn::Load(d, &add_array[i]);
        auto out_a = hn::Load(d, &out_array[i]);
        auto res = hn::MulAdd(mul_a, out_a, add_a);
        hn::Store(res, d, &out_array[i]);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace simd_math
HWY_AFTER_NAMESPACE();
#endif