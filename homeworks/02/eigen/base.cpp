#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

int main() {
  MatrixXd m(2, 2); // dynamic sized matrix
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  Eigen::Matrix2f matrix_a; // static sized matrix
  matrix_a << 1.0f, 2.0f, 3.0f, 4.0f;
  Eigen::Matrix2f matrix_b;
  matrix_b << 1.0f, 2.0f, 3.0f, 4.0f;
  // auto matrix_c = matrix_a * matrix_b; // lazy operation
  Eigen::Matrix2f matrix_c = matrix_a * matrix_b; // lazy operation
  Eigen::VectorXf vector(2);                      // dynamic sized vector
  vector << 3.f, 5.f;
  matrix_c.colwise() += vector; // broadcasting

  std::cout << matrix_c.eval() << std::endl;

  std::cout << m << std::endl;
}