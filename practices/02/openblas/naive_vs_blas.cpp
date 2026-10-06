// Task 2: naive GEMM vs cblas_sgemm, correctness check + GFLOPs for several
// sizes.
#include <cblas-openblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Row-major triple-loop matmul: C[m x n] += A[m x k] * B[k x n].
void naive_gemm(const int m, const int n, const int k, const float* A,
                const float* B, float* C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float val = 0.0f;
            for (int l = 0; l < k; l++) val += A[i * k + l] * B[l * n + j];
            C[i * n + j] = val;
        }
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-1.f, 1.f);

    auto fill_matrix = [&rng, &d](std::vector<float>& A, const int n,
                                  const int m) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) A[i * m + j] = d(rng);
    };

    for (const int sz : {128, 512, 1024}) {
        const int m = sz + 1, n = sz, k = sz - 1;
        std::vector<float> A(m * k), B(k * n), C(m * n, 0.f), Cb(m * n, 0.f);

        fill_matrix(A, m, k);
        fill_matrix(B, k, n);

        auto t0 = std::chrono::steady_clock::now();
        // TODO: use cblas_sgemm to calculate GEMM use correct memory order and
        // LDA

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                    A.data(), k, B.data(), n, 0.0f, Cb.data(), n);

        auto t1 = std::chrono::steady_clock::now();
        naive_gemm(m, n, k, A.data(), B.data(), C.data());
        auto t2 = std::chrono::steady_clock::now();

        float maxdiff = 0.f;
        for (int i = 0; i < m * n; ++i)
            maxdiff = std::max(maxdiff, std::fabs(C[i] - Cb[i]));

        const double flops = 2.0 * m * n * k;
        const double t_blas = std::chrono::duration<double>(t1 - t0).count();
        const double t_naive = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "size " << sz << "x" << sz
                  << ":  cblas = " << (flops / t_blas / 1e9)
                  << " GFLOPs,  naive = " << (flops / t_naive / 1e9)
                  << " GFLOPs,  maxdiff = " << maxdiff << "\n";
    }
    return 0;
}