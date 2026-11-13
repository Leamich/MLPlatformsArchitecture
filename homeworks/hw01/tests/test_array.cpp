#include <cstdint>
#include <utility>

#include "test_utils.hpp"

template <typename T>
class ArrayTest : public ::testing::Test {};
TYPED_TEST_SUITE(ArrayTest, TestTypes);

TYPED_TEST(ArrayTest, Constructors) {
    using T = TypeParam;
    Array<T> a(5);
    EXPECT_EQ(a.shape(), (Shape{1, 5}));
    for (T x : a) EXPECT_EQ(x, T(0));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % HWY_ALIGNMENT, 0u);

    EXPECT_EQ(Array<T>(2, 3).shape(), (Shape{2, 3}));
    EXPECT_TRUE(Array<T>().empty());

    Array<T> m{{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(m.shape(), (Shape{2, 3}));
    EXPECT_EQ(m(1, 0), T(4));
    EXPECT_THROW((Array<T>{{1, 2}, {3}}), std::invalid_argument);
}

TYPED_TEST(ArrayTest, AtThrowsOutOfRange) {
    using T = TypeParam;
    Array<T> a(2, 3);
    EXPECT_NO_THROW((void)a.at(1, 2));
    EXPECT_THROW((void)a.at(2, 0), std::out_of_range);
    EXPECT_THROW((void)a.at(0, 3), std::out_of_range);
}

TYPED_TEST(ArrayTest, CopyIsDeep) {
    using T = TypeParam;
    Array<T> a{1, 2, 3};
    Array<T> b = a;
    b[0] = T(42);
    EXPECT_EQ(a[0], T(1));

    Array<T> c(10);
    c = a;
    EXPECT_EQ(c.shape(), a.shape());
    EXPECT_NE(c.data(), a.data());
}

TYPED_TEST(ArrayTest, MoveLeavesSourceEmpty) {
    using T = TypeParam;
    Array<T> a(2, 3);
    const T* p = a.data();
    Array<T> b = std::move(a);
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(a.shape(), (Shape{0, 0}));

    Array<T> c;
    c = std::move(b);
    EXPECT_EQ(c.data(), p);
    EXPECT_TRUE(b.empty());
}
