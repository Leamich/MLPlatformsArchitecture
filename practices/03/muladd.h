#pragma once

#include <stddef.h>

namespace simd_math {

void mul_add(const float *a, const float *b, float *c, size_t n);
void mul_add(const double *a, const double *b, double *c, size_t n);

} // namespace simd_math