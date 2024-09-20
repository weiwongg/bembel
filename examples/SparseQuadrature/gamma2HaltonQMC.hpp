#ifndef SPQR_GAMMA2HALTONQUADRATURE_H_
#define SPQR_GAMMA2HALTONQUADRATURE_H_

#include <Eigen/Dense>
#include <cmath>

#include "HaltonSet.hpp"

template <typename T>
struct myCompareInc {
  const T &m;
  myCompareInc(const T &p) : m(p){};

  bool operator()(const int &i, const int &j) { return (m(i) > m(j)); }
};

void gamma2HaltonQMC(const Eigen::VectorXd &gamma, Eigen::MatrixXd *Q,
                     Eigen::VectorXd *W, int q) {
  int num_samples = 1 << q;
  HaltonSet HS(gamma.size(), 100);
  Eigen::MatrixXd sortQ(gamma.size(), num_samples);
  Eigen::VectorXi sort =
      Eigen::ArrayXi::LinSpaced((int)gamma.size(), 0, (int)gamma.size() - 1);

  std::sort(sort.data(), sort.data() + sort.size(),
            myCompareInc<Eigen::VectorXd>(gamma));
  W->resize(num_samples);
  for (auto i = 0; i < num_samples; ++i) {
    sortQ.col(i) = 2 * HS.get_HaltonVector().array() - 1;
    HS.next();
    (*W)(i) = double(1) / double(num_samples);
  }
  Q->resize(gamma.size(), num_samples);
  for (auto i = 0; i < sort.size(); ++i) Q->row(sort(i)) = sortQ.row(i);
  return;
}
#endif
