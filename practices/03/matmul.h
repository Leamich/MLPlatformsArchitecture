#pragma once

#include <stddef.h>

namespace simd_math {

void matmul(const float *a, const float *b, float *c, size_t n);

} // namespace simd_math