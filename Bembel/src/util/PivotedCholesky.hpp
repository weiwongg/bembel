#ifndef __PIVOTEDCHOLESKY__CLASS__
#define __PIVOTEDCHOLESKY__CLASS__

#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <iostream>

template <typename Scalar>
class PivotedCholesky {
 public:
  // void constructor
  PivotedCholesky() {}
  // non-void constructor
  template <class Kernel>
  PivotedCholesky(const Kernel &C, double tol) {
    compute(C, tol);
  }
  /*
   *   \brief approximates the approximation error of the pivoted Cholesky
   *   decomposition by Monte Carlo sampling
   */
  template <class Kernel>
  void sampleError(const Kernel &C, int samples) const {
    double colOp = 0;
    double colL = 0;
    double maxError = 0;
    double aveError = 0;
    double error = 0;
    int col = 0;
    int dim = C.get_dim();
    std::srand(std::time(NULL));
    // sample random columns from C and compare them to the
    // respective column
    // from L * L'
    for (int i = 0; i < samples; ++i) {
      error = 0;
      col = std::rand() % dim;
      for (int j = 0; j < dim; ++j) {
        colOp = C(j, col);
        colL = _L.row(j).dot(_L.row(col));
        error += (colOp - colL) * (colOp - colL);
      }
      error = sqrt(error);
      if (maxError < error) maxError = error;
      aveError += error / samples;
    }

    std::cout << "maximum error: " << maxError << " average error: " << aveError
              << " samples: " << samples << std::endl;
  }
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &get_L(
      void) const {
    return _L;
  }
  double get_tol(void) const { return _tol; }
  /*
   *    \brief computes the pivoted Cholesky decomposition of a positive
   *           semi definite function provided by the class Kernel
   */
  template <class Kernel>
  void compute(const Kernel &C, double tol) {
    Eigen::VectorXd D;
    Eigen::Index pivot = 0;
    double tr = 0;
    constexpr int allocBSize = 100;
    int actBSize = 0;
    int dim = C.get_dim();
    _tol = tol;
    // compute the diagonal and the trace
    D = C.diagonal();
    eigen_assert(D.minCoeff() >= 0 &&
                 "PivotedCholesky: Kernel not pos.sem.def");
    D.maxCoeff(&pivot);
    tr = D.sum();
    tol *= tr;
    // allocate memory for L
    actBSize = allocBSize;
    _L.conservativeResize(dim, actBSize);
    _L.setZero();
    // perform pivoted Cholesky decomposition
    int step = 0;
    while ((step < dim) && (tol < tr)) {
      // check memory requirements
      if (actBSize - 1 <= step) {
        actBSize += allocBSize;
        _L.conservativeResize(dim, actBSize);
      }
      // get new column from C
      _L.col(step) = C.col(pivot);
      // update column with the current matrix _L
      _L.col(step) -=
          _L.block(0, 0, dim, step) * _L.row(pivot).head(step).transpose();
      _L.col(step) /= sqrt(_L(pivot, step));
      // update the diagonal and the trace
      D.array() -= _L.col(step).array().square();
      D.maxCoeff(&pivot);
      tr = D.sum();
      std::cout << tr << std::endl;
      ++step;
    }
    // crop L to its actual size
    _L.conservativeResize(dim, step);
  }
  void compute(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &C,
               double tol) {
    Eigen::VectorXd D;
    Eigen::Index pivot = 0;
    double tr = 0;
    D = C.diagonal().real();
    eigen_assert(C.diagonal().imag().norm() < 1e-10 &&
                 D.cwiseAbs().minCoeff() >= 0 && C.rows() == C.cols() &&
                 "PivotedCholesky: Kernel not pos.sem.def or hermitian");
    D.maxCoeff(&pivot);
    tr = D.sum();
    tol *= tr;
    // allocate memory for L
    _L.resize(C.rows(), C.cols());
    _L.setZero();
    // perform pivoted Cholesky decomposition
    int step = 0;
    while ((step < C.cols()) && (tol < tr)) {
      // get new column from C
      _L.col(step) = C.col(pivot);
      // update column with the current matrix _L
      _L.col(step) -=
          _L.block(0, 0, _L.rows(), step) * _L.row(pivot).head(step).adjoint();
      _L.col(step) /= sqrt(_L(pivot, step));
      // update the diagonal and the trace
      D.array() -= _L.col(step).array().abs().square();
      D.maxCoeff(&pivot);
      tr = D.sum();
      ++step;
    }
    // crop L to its actual size
    _L.conservativeResize(_L.rows(), step);
  }

 private:
  double _tol;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _L;
};

#endif
