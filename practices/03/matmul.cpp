#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "matmul.cpp"
#include <hwy/foreach_target.h>

#include "matmul-inl.h"
#include <hwy/highway.h>
#include <hwy/per_target.h>

#if HWY_ONCE
#include "matmul.h"

namespace simd_math {

void matmul(const float *a, const float *b, float *c, size_t n) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(matmul<float>)(a, b, c, n);
}

} // namespace simd_math
#endif // HWY_ONCE