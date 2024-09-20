#ifndef SPQR_GAMMA2MONTECARLO_H_
#define SPQR_GAMMA2MONTECARLO_H_

#include <Eigen/Dense>
#include <cmath>

void gamma2MC(const Eigen::VectorXd &gamma, Eigen::MatrixXd *Q,
                     Eigen::VectorXd *W, int q) {
  int num_samples = 1 << q;
  W->resize(num_samples);
  Q->resize(gamma.size(), num_samples);
  for (auto i = 0; i < num_samples; ++i) {
    Q->col(i) = Eigen::VectorXd::Random(gamma.size());
    (*W)(i) = double(1) / double(num_samples);
  }
  return;
}
#endif
