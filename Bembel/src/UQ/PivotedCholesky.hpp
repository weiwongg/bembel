// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_UQ_PIVOTEDCHOLESKY_H_
#define BEMBEL_UQ_PIVOTEDCHOLESKY_H_

namespace Bembel {
namespace UQ {

/**
 * \ingroup UQ
 * \brief computes a diagonally pivoted Cholesky decomposition */

class PivotedCholesky {
 public:
  //////////////////////////////////////////////////////////////////////////////
  static constexpr int allocBSize = 100;
  // void constructor
  PivotedCholesky() {}
  // non-void constructor
  template <class covKernel>
  PivotedCholesky(const covKernel &C, double tol) {
    compute(C, tol);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// getters
  //////////////////////////////////////////////////////////////////////////////
  const Eigen::MatrixXd &get_L(void) const { return Lmatrix_; }

  const Eigen::MatrixXd &get_Lambda(void) const { return LambdaMatrix_; }

  const Eigen::MatrixXd &get_Q(void) const { return Qmatrix_; }

  const Eigen::VectorXi &get_indices(void) const { return indices_; }

  double get_tol(void) const { return tol_; }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *    \brief computes the pivoted Cholesky decomposition of a positive
   *           semi definite function provided by the class covKernel
   */
  template <class Kernel>
  void compute(const Kernel &C, double tol) {
    Eigen::VectorXd D;
    Eigen::Index pivot = 0;
    double tr = 0;
    int actBSize = 0;
    int dim = C.get_dim();
    tol_ = tol;
    // compute the diagonal and the trace
    D = C.diagonal();
    eigen_assert(D.minCoeff() >= 0 &&
                 "PivotedCholesky: Kernel not pos.sem.def");
    tr = D.sum();
    tol *= tr;
    // perform pivoted Cholesky decomposition
    int step = 0;
    while ((step < dim) && (tol < tr)) {
      // check memory requirements
      if (actBSize - 1 <= step) {
        actBSize += allocBSize;
        Lmatrix_.conservativeResize(dim, actBSize);
        indices_.conservativeResize(actBSize);
      }
      D.maxCoeff(&pivot);
      indices_(step) = pivot;
      // get new column from C
      Lmatrix_.col(step) = C.col(pivot);
      // update column with the current matrix Lmatrix_
      Lmatrix_.col(step) -= Lmatrix_.block(0, 0, dim, step) *
                            Lmatrix_.row(pivot).head(step).transpose();
      Lmatrix_.col(step) /= sqrt(Lmatrix_(pivot, step));
      // update the diagonal and the trace
      D.array() -= Lmatrix_.col(step).array().square();
      tr = D.sum();
      ++step;
    }
    // crop L to its actual size
    Lmatrix_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
  }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *    \brief computes the spectral decomposition of LL' in low-rank format
   *
   *    at the moment, it is exploited that eig(LL') = eig(L'L). However,
   *    it could be more stable to first compute a QR decomposition of L and
   *    then to compute the spectral decomposition of RR'...
   */
  void computeSpectralDecomposition() {
    Eigen::MatrixXd C = Lmatrix_.transpose() * Lmatrix_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);

    Qmatrix_ = Lmatrix_ * es.eigenvectors();
    // save eigenvalues in decreasing order
    LambdaMatrix_ = es.eigenvalues().reverse().asDiagonal();
    // save eigenvectors accordingly
    for (auto i = 0; i < Qmatrix_.cols() / 2; ++i)
      Qmatrix_.col(i).swap(Qmatrix_.col(Qmatrix_.cols() - 1 - i));
    // normalize Q vectors
    for (auto i = 0; i < Qmatrix_.cols(); ++i)
      Qmatrix_.col(i) /= Qmatrix_.col(i).norm();
  }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *   \brief approximates the approximation error of the pivoted Cholesky
   *   decomposition by Monte Carlo sampling
   */
  template <class covKernel>
  void sampleError(const covKernel &C, int samples) const {
    Eigen::VectorXd colOp;
    Eigen::VectorXd colL;
    double maxError = 0;
    double aveError = 0;
    double error = 0;
    int sampleCol = 0;
    int dim = C.get_dim();
    std::srand(std::time(NULL));
    // sample random columns from C and compare them to the respective column
    // from L * L'
    for (auto i = 0; i < samples; ++i) {
      error = 0;
      sampleCol = std::rand() % dim;
      colOp = C.col(sampleCol);
      colL = Lmatrix_ * Lmatrix_.row(sampleCol).transpose();
      error = (colOp - colL).norm() / colOp.norm();
      if (maxError < error) maxError = error;
      aveError += error / samples;
    }

    std::cout << "max err: " << maxError << " avrg err: " << aveError
              << " samples: " << samples << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  double tol_;
  Eigen::MatrixXd Lmatrix_;
  Eigen::MatrixXd Qmatrix_;
  Eigen::MatrixXd LambdaMatrix_;
  Eigen::VectorXi indices_;
};

}  // namespace UQ
}  // namespace Bembel

#endif
