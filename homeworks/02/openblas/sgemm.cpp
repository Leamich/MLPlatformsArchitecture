#include <cblas-openblas.h>
#include <iostream>
#include <vector>

int main() {
  try {
    blasint m = 128;
    blasint k = 128;
    blasint n = 128;
    std::vector<float> matrix_a(m * k);
    std::vector<float> matrix_b(k * n);
    std::vector<float> matrix_c(m * n);

    /*
      The leading dimension in BLAS (Basic Linear Algebra Subprograms) refers
      to the stride between consecutive elements of the same row (in
      column-major layout) or the same column (in row-major layout) within a
      matrix stored in a one-dimensional array.
    */
    blasint lda(k);
    blasint ldb(n);
    blasint ldc(n);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                /*alpha*/ 1.f, matrix_a.data(), lda, matrix_b.data(), ldb,
                /*beta*/ 1.f, matrix_c.data(), ldc);

  } catch (const std::exception &ex) {
    std::cerr << "Application failed with exception: " << ex.what()
              << std::endl;
    exit(1);
  }
  return 0;
}