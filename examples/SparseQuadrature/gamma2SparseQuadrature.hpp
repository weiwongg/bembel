#ifndef SPQR_GAMMA2SPARSEQUADRATURE_H_
#define SPQR_GAMMA2SPARSEQUADRATURE_H_

#include <Eigen/Dense>
#include <cmath>

#include "GaussLegendreQuadratureProb.hpp"
#include "SparseQuadrature.hpp"
#include "TDindexSet.hpp"

//
//  It's an older code, sir, but it checks out.
//
//
//
void gamma2SparseQuadrature(const Eigen::VectorXd &gamma, Eigen::MatrixXd *Q,
                            Eigen::VectorXd *W, int q) {
  TDindexSet TD;
  SparseQuadrature SQ;
  GaussLegendreQuadratureProb GLP;
  Eigen::VectorXd w(gamma.size());

  double kappa = 0;
  int maxLvl = 0;
  /**
   *  kappa = 2\tau_n/|\Gamma_n|+sqrt(4\tau_n^2/|\Gamma_n|^2+1)
   *  here, we have |Gamma_n| = 2 and, therefore since \tau_n\sim 1/\gamma_n,
   **/
  for (int i = 0; i < gamma.size(); ++i) {
    kappa = 1. / gamma(i) + std::sqrt(1. + 1. / gamma(i) / gamma(i));
    w(i) = std::log(kappa);
  }
  maxLvl = std::ceil(q / w.minCoeff());
  GLP.initQuadrature(maxLvl);
  TD.computeIndexSet(q, w);
  SQ.computeSparseQuadrature(TD, GLP);
  SQ.purgeSparseQuadrature();
  auto sort = TD.get_sortW();
  Q->resize(SQ.get_qPoints().rows(), SQ.get_qPoints().cols());
  *W = SQ.get_qWeights();
  for (auto i = 0; i < sort.size(); ++i)
    Q->row(sort(i)) = SQ.get_qPoints().row(i);
  return;
}
#endif
