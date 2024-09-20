#include "TensorProductQuadrature.hpp"

/**    \brief default constructor without arguments
*
*/
TensorProductQuadrature::TensorProductQuadrature(void) {}

/**    \brief constructor with quadrature degrees calls init_quadrature
*      \param[in] degs maximum quadrature degrees in each dimension
*
*/
TensorProductQuadrature::TensorProductQuadrature(const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl,
                                                 UnivariateQuadrature &Q) {
  initQuadrature(lvl, Q);
}

/**    \brief initializes the tensor product quadrature based on the
*             Gauss Legendre quadrature (C routine)
*      \param[in] degs maximum quadrature degrees in each dimension
*
*/
void TensorProductQuadrature::initQuadrature(const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl,
                                              UnivariateQuadrature &Q) {
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> base;
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> nPtsInQuad;
  long long int remainder = 0;
  long long int quotient = 0;
  long long int maxLvl = 0;

  for (long long int i = 0; i < lvl.size(); ++i)
    if (maxLvl < lvl(i)) maxLvl = lvl(i);

  Q.resizeQuadrature(maxLvl);
  nPtsInQuad.resize(lvl.size());
  const std::vector<Quadrature> &refQ = Q.get_Q();

  for (long long int i = 0; i < lvl.size(); ++i) nPtsInQuad(i) = refQ[lvl(i)].w.size();

  _nPts = nPtsInQuad.prod();
  _points.resize(lvl.size(), _nPts);
  _weights.resize(_nPts);

  base = computeBase(nPtsInQuad);
  for (long long int i = 0; i < _nPts; ++i) {
    _weights(i) = 1.;
    remainder = i;
    for (long long int j = 0; j < lvl.size(); ++j) {
      quotient = remainder / base(j);
      _points(j, i) = refQ[lvl(j)].xi(quotient);
      _weights(i) *= refQ[lvl(j)].w(quotient);
      remainder -= base(j) * quotient;
    }
  }
}

/**   \brief this function needs a comment
*
*/
Eigen::Matrix<long long int, Eigen::Dynamic, 1> TensorProductQuadrature::computeBase(
    const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &lvl) {
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> base;
  base = Eigen::Matrix<long long int, Eigen::Dynamic, 1>::Ones(lvl.size());
  for (long long int i = (long long int)lvl.size() - 1; i >= 0; --i)
    base(lvl.size() - i - 1) = lvl.tail(i).prod();
  return base;
}

/**   \brief make an educated guess, what this function does...
*
*/
const Eigen::VectorXd &TensorProductQuadrature::get_weights(void) const {
  return _weights;
}

/**   \brief make an educated guess, what this function does...
*
*/
const Eigen::MatrixXd &TensorProductQuadrature::get_points(void) const {
  return _points;
}

/**   \brief make an educated guess, what this function does...
*
*/
long long int TensorProductQuadrature::get_nPts(void) const { return _nPts; }

