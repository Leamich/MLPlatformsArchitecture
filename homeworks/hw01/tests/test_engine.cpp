#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include "test_utils.hpp"

template <typename T>
class EngineTest : public ::testing::Test {};
TYPED_TEST_SUITE(EngineTest, TestTypes);

// Runs op(engine, a, b) on every backend and compares with backends[0] (scalar/seq).
template <typename T, typename Op>
void check_backends_agree(Op op, double rtol, double atol) {
    auto backends = make_backends<T>();
    for (std::size_t n : kSizes) {
        const auto a = random_array<T>({1, n}, 1, T(-10), T(10));
        auto b = random_array<T>({1, n}, 2, T(-10), T(10));
        for (T& x : b) {
            if (x == T(0)) x = T(1);  // valid divisor
        }
        const auto expected = op(*backends[0].engine, a, b);
        for (std::size_t i = 1; i < backends.size(); ++i) {
            SCOPED_TRACE(backends[i].name + ", n = " + std::to_string(n));
            expect_array_near(op(*backends[i].engine, a, b), expected, rtol, atol);
        }
    }
}

TYPED_TEST(EngineTest, ElementwiseKnownValues) {
    using T = TypeParam;
    for (auto& be : make_backends<T>()) {
        SCOPED_TRACE(be.name);
        const Engine<T>& e = *be.engine;
        const Array<T> a{1, 2, 3}, b{4, 5, 6};
        expect_array_near(e.add(a, b), Array<T>{5, 7, 9});
        expect_array_near(e.sub(a, b), Array<T>{-3, -3, -3});
        expect_array_near(e.mul(a, b), Array<T>{4, 10, 18});
        expect_array_near(e.div(Array<T>{8, 9, 3}, Array<T>{2, 3, 4}), Array<T>{4, 3, T(3) / T(4)});
    }
}

TYPED_TEST(EngineTest, ElementwiseBackendsAgree) {
    using T = TypeParam;
    check_backends_agree<T>([](const auto& e, const auto& a, const auto& b) { return e.add(a, b); }, 0, 0);
    check_backends_agree<T>([](const auto& e, const auto& a, const auto& b) { return e.sub(a, b); }, 0, 0);
    check_backends_agree<T>([](const auto& e, const auto& a, const auto& b) { return e.mul(a, b); }, 0, 0);
    check_backends_agree<T>([](const auto& e, const auto& a, const auto& b) { return e.div(a, b); }, 0, 0);
}

TYPED_TEST(EngineTest, ShapeMismatchThrows) {
    using T = TypeParam;
    for (auto& be : make_backends<T>()) {
        EXPECT_THROW(be.engine->add(Array<T>(2, 3), Array<T>(3, 2)), std::invalid_argument);
        EXPECT_THROW(be.engine->matmul(Array<T>(2, 3), Array<T>(2, 3)), std::invalid_argument);
    }
}

TYPED_TEST(EngineTest, SumAndMean) {
    using T = TypeParam;
    Array<T> a(100);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = T(i + 1);

    for (auto& be : make_backends<T>()) {
        SCOPED_TRACE(be.name);
        EXPECT_EQ(be.engine->sum(a), 5050);
        EXPECT_EQ(be.engine->sum(Array<T>()), 0);
        EXPECT_DOUBLE_EQ(be.engine->mean(a), 50.5);
        EXPECT_THROW(be.engine->mean(Array<T>()), std::invalid_argument);
    }
}

TYPED_TEST(EngineTest, SumBackendsAgree) {
    using T = TypeParam;
    auto backends = make_backends<T>();
    for (std::size_t n : kSizes) {
        const auto a = random_array<T>({1, n}, 3);
        const double expected = double(backends[0].engine->sum(a));
        // float sums in a different order differ by rounding
        const double tol = std::is_same_v<T, float> ? 1e-5 * (std::abs(expected) + n) : 1e-9;
        for (std::size_t i = 1; i < backends.size(); ++i) {
            SCOPED_TRACE(backends[i].name + ", n = " + std::to_string(n));
            EXPECT_NEAR(double(backends[i].engine->sum(a)), expected, tol);
        }
    }
}

// Checks every backend against the exact formula on x in [-20, 20] and prints the max error.
template <typename T, typename Op>
void check_vs_exact(Op op, double (*exact)(double), const char* op_name, double atol) {
    Array<T> x(4001);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = T(-20 + 0.01 * double(i));  // int32: whole numbers

    for (auto& be : make_backends<T>()) {
        SCOPED_TRACE(be.name);
        const auto y = op(*be.engine, x);
        double max_err = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double err = std::abs(double(y[i]) - exact(double(x[i])));
            max_err = std::max(max_err, err);
            EXPECT_LE(err, atol) << op_name << "(" << double(x[i]) << ")";
        }
        std::cout << op_name << " " << ::testing::UnitTest::GetInstance()->current_test_info()->type_param()
                  << " " << be.name << ": max abs error " << max_err << "\n";
    }
}

static double exact_silu(double x) { return x / (1.0 + std::exp(-x)); }
static double exact_gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

TYPED_TEST(EngineTest, ActivationsVsExact) {
    using T = TypeParam;
    // SiLU uses the exact formula (only rounding error); the GELU tanh approximation
    // itself is off by up to ~5e-4.
    check_vs_exact<T>([](const auto& e, const auto& x) { return e.silu(x); }, exact_silu, "silu", 2e-5);
    check_vs_exact<T>([](const auto& e, const auto& x) { return e.gelu(x); }, exact_gelu, "gelu", 1e-3);
}

TYPED_TEST(EngineTest, ActivationsBackendsAgree) {
    using T = TypeParam;
    check_backends_agree<T>([](const auto& e, const auto& a, const auto&) { return e.silu(a); }, 1e-5, 1e-5);
    check_backends_agree<T>([](const auto& e, const auto& a, const auto&) { return e.gelu(a); }, 1e-5, 1e-5);
}

TYPED_TEST(EngineTest, ActivationsOfInfinity) {
    using T = TypeParam;
    if constexpr (std::is_floating_point_v<T>) {
        const T inf = std::numeric_limits<T>::infinity();
        Array<T> x(37);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = i % 2 ? inf : -inf;
        Array<T> expected(37);
        for (std::size_t i = 0; i < x.size(); ++i) expected[i] = i % 2 ? inf : 0;

        for (auto& be : make_backends<T>()) {
            SCOPED_TRACE(be.name);
            expect_array_near(be.engine->silu(x), expected);
            expect_array_near(be.engine->gelu(x), expected);
        }
    }
}

TYPED_TEST(EngineTest, Matmul) {
    using T = TypeParam;
    const Array<T> a{{1, 2}, {3, 4}};
    const Array<T> b{{5, 6}, {7, 8}};
    for (auto& be : make_backends<T>()) {
        SCOPED_TRACE(be.name);
        expect_array_near(be.engine->matmul(a, b), Array<T>{{19, 22}, {43, 50}});
    }
}

TYPED_TEST(EngineTest, MatmulBackendsAgree) {
    using T = TypeParam;
    auto backends = make_backends<T>();
    for (std::size_t n : {1, 7, 8, 9, 17, 65}) {
        // whole numbers -> the result is exact in any summation order
        auto a = random_array<T>({n, n + 3}, 4, T(-10), T(10));
        auto b = random_array<T>({n + 3, n + 1}, 5, T(-10), T(10));
        for (T& x : a) x = T(std::round(x));
        for (T& x : b) x = T(std::round(x));

        const auto expected = backends[0].engine->matmul(a, b);
        for (std::size_t i = 1; i < backends.size(); ++i) {
            SCOPED_TRACE(backends[i].name + ", n = " + std::to_string(n));
            expect_array_near(backends[i].engine->matmul(a, b), expected);
        }
    }
}

TEST(Int32Test, OverflowWrapsAndDivisionByZeroThrows) {
    const int32_t max = std::numeric_limits<int32_t>::max();
    const Shape shape{1, 37};
    for (auto& be : make_backends<int32_t>()) {
        SCOPED_TRACE(be.name);
        const Engine<int32_t>& e = *be.engine;
        const auto sum = e.add(Array<int32_t>(shape, max), Array<int32_t>(shape, 1));
        expect_array_near(sum, Array<int32_t>(shape, std::numeric_limits<int32_t>::min()));
        EXPECT_EQ(e.sum(Array<int32_t>(shape, max)), 37 * int64_t(max));
        EXPECT_THROW(e.div(Array<int32_t>(shape, 1), Array<int32_t>(shape, 0)), std::domain_error);
    }
}
