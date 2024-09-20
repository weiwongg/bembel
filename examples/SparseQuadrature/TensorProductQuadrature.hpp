#ifndef __TENSORQUADRATURE__CLASS__
#define __TENSORQUADRATURE__CLASS__

#include <Eigen/Dense>

#include "UnivariateQuadrature.hpp"

class TensorProductQuadrature {
 public:
  TensorProductQuadrature(void);
  TensorProductQuadrature(const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl, UnivariateQuadrature &Q);
  void initQuadrature(const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl, UnivariateQuadrature &Q);
  const Eigen::VectorXd &get_weights(void) const;
  const Eigen::MatrixXd &get_points(void) const;
  long long int get_nPts(void) const;
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> computeBase(const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl);

 protected:
  Eigen::VectorXd _weights;
  Eigen::MatrixXd _points;
  long long int _nPts;
};

#endif
