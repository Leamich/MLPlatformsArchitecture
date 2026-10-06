#include <Eigen/Dense>
#include <chrono>
#include <iostream>

using Eigen::MatrixXd;

MatrixXd naive_mul(const MatrixXd& A, const MatrixXd& B) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(B.cols());
    const int k = static_cast<int>(A.cols());
    MatrixXd C(m, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double val = 0;
            for (int l = 0; l < k; l++) {
                val += A(i, l) * B(l, j);
            }
            C(i, j) = val;
        }
    return C;
}

int main() {
    const int N = 256;
    const MatrixXd A = MatrixXd::Random(N, N);
    const MatrixXd B = MatrixXd::Random(N, N);

    auto t0 = std::chrono::steady_clock::now();
    const MatrixXd C1 = A * B;
    auto t1 = std::chrono::steady_clock::now();
    const MatrixXd C2 = naive_mul(A, B);
    auto t2 = std::chrono::steady_clock::now();

    const auto expr = A * B + (A + B) * A;
    auto t3 = std::chrono::steady_clock::now();
    const MatrixXd C3 = expr;  // TODO: materialize the `expr` value
    auto t4 = std::chrono::steady_clock::now();

    const auto ms = [](auto x, auto y) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(y - x)
            .count();
    };
    std::cout << "Eigen A*B           : " << ms(t0, t1) << " ms\n";
    std::cout << "Naive triple loop   : " << ms(t1, t2) << " ms\n";
    std::cout << "Lazy expr + .eval() : " << ms(t3, t4) << " ms\n";
    std::cout << "Max |Eigen - naive| : " << (C1 - C2).cwiseAbs().maxCoeff()
              << "\n";
    std::cout << "C3(0,0) (expr)      : " << C3(0, 0) << "\n";
    return 0;
}