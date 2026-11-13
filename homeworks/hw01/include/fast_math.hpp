#pragma once
#include <algorithm>
#include <cmath>

// The SIMD kernels do exactly the same steps with hn::Exp / hn::Tanh.
namespace fm {

// exp(80) still fits in float. Below x = -80, silu(x) is ~1e-33, so we return 0
// (this also makes silu(-inf) = 0 instead of -inf * 0 = NaN).
constexpr double kExpLimit = 80.0;

// tanh(20) == 1 exactly in float and double. Below x = -20, gelu(x) is ~1e-88.
constexpr double kTanhLimit = 20.0;

constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kGeluCubic = 0.044715;

template <typename F>
F sigmoid(F x) {
    const F t = std::clamp(-x, F(-kExpLimit), F(kExpLimit));
    return F(1) / (F(1) + std::exp(t));
}

template <typename F>
F silu(F x) {
    if (x < F(-kExpLimit)) return F(0);
    return x * sigmoid(x);
}

// tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
template <typename F>
F gelu(F x) {
    if (x < F(-kTanhLimit)) return F(0);
    const F u = F(kSqrt2OverPi) * (x + F(kGeluCubic) * x * x * x);
    const F t = std::tanh(std::clamp(u, F(-kTanhLimit), F(kTanhLimit)));
    return F(0.5) * x * (F(1) + t);
}

}  // namespace fm
