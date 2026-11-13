#pragma once
#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <utility>

// A 1D array of n elements has shape {1, n}.
struct Shape {
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::size_t size() const noexcept { return rows * cols; }
    bool operator==(const Shape& o) const noexcept { return rows == o.rows && cols == o.cols; }
    bool operator!=(const Shape& o) const noexcept { return !(*this == o); }
};

// Owning, aligned, row-major 1D/2D array.
//   Array<float> a(5);                   5 zeros
//   Array<float> m(2, 3);                2x3 zeros
//   Array<float> v{1, 2, 3};             values
//   Array<float> w{{1, 2}, {3, 4}};      2x2 values
//   Array<float>::uninitialized({2, 3})  not zeroed
template <typename T>
class Array {
public:
    using value_type = T;

    Array() = default;
    explicit Array(std::size_t n) : Array(Shape{1, n}, T(0)) {}
    Array(std::size_t rows, std::size_t cols) : Array(Shape{rows, cols}, T(0)) {}
    explicit Array(Shape shape) : Array(shape, T(0)) {}

    Array(Shape shape, T fill) {
        allocate(shape);
        std::fill(begin(), end(), fill);
    }

    Array(std::initializer_list<T> values) {
        allocate(Shape{1, values.size()});
        std::copy(values.begin(), values.end(), begin());
    }

    Array(std::initializer_list<std::initializer_list<T>> rows) {
        const std::size_t cols = rows.size() ? rows.begin()->size() : 0;
        for (const auto& row : rows) {
            if (row.size() != cols) throw std::invalid_argument("Array: rows have different lengths");
        }
        allocate(Shape{rows.size(), cols});
        T* p = data();
        for (const auto& row : rows) p = std::copy(row.begin(), row.end(), p);
    }

    static Array uninitialized(Shape shape) {
        Array a;
        a.allocate(shape);
        return a;
    }

    Array(const Array& other) {
        allocate(other.shape_);
        std::copy(other.begin(), other.end(), begin());
    }

    Array& operator=(const Array& other) {
        if (this != &other) {
            allocate(other.shape_);
            std::copy(other.begin(), other.end(), begin());
        }
        return *this;
    }

    Array(Array&& other) noexcept
        : shape_(std::exchange(other.shape_, Shape{})), data_(std::move(other.data_)) {}

    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = std::exchange(other.shape_, Shape{});
        }
        return *this;
    }

    const Shape& shape() const noexcept { return shape_; }
    std::size_t rows() const noexcept { return shape_.rows; }
    std::size_t cols() const noexcept { return shape_.cols; }
    std::size_t size() const noexcept { return shape_.size(); }
    bool empty() const noexcept { return size() == 0; }

    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    T& operator()(std::size_t r, std::size_t c) noexcept { return data_[r * cols() + c]; }
    const T& operator()(std::size_t r, std::size_t c) const noexcept { return data_[r * cols() + c]; }

    T& at(std::size_t r, std::size_t c) {
        check_index(r, c);
        return (*this)(r, c);
    }
    const T& at(std::size_t r, std::size_t c) const {
        check_index(r, c);
        return (*this)(r, c);
    }

    T* begin() noexcept { return data(); }
    T* end() noexcept { return data() + size(); }
    const T* begin() const noexcept { return data(); }
    const T* end() const noexcept { return data() + size(); }

private:
    void check_index(std::size_t r, std::size_t c) const {
        if (r >= rows() || c >= cols()) throw std::out_of_range("Array::at: index out of range");
    }

    // Allocates uninitialized memory for shape.size() elements.
    void allocate(Shape shape) {
        const std::size_t n = shape.size();
        if (n > 0) {
            data_ = hwy::AllocateAligned<T>(n);
            if (!data_) throw std::bad_alloc();
        } else {
            data_.reset();
        }
        shape_ = shape;
    }

    Shape shape_{};
    hwy::AlignedFreeUniquePtr<T[]> data_;
};
