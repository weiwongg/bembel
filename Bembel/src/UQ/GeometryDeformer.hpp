
// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_UQ_GEOMETRYDEFORMER_H_
#define BEMBEL_UQ_GEOMETRYDEFORMER_H_

namespace Bembel {
namespace UQ {

class GeometryDeformer {
 public:
  GeometryDeformer(){};
  //////////////////////////////////////////////////////////////////////////////
  GeometryDeformer(const std::string &filename, double scale = 1) {
    init_GeometryDeformer(filename, scale);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init_GeometryDeformer(const std::string &filename, double scale = 1) {
    IO::readPointsAscii(&PE_, &p_, &m_, filename);
    n_ = (1 << m_) + 1;
    points_.resize(3, p_ * n_ * n_);
    for (auto i = 0; i < p_; ++i)
      points_.block(0, i * n_ * n_, 3, n_ * n_) = PE_[i];
    points_ *= scale;
    {
      KernelMatrix<Eigen::Matrix<double, 3, Eigen::Dynamic>> K(points_);
      PivotedCholesky piv_chol(K, 1e-8);
      piv_chol.computeSpectralDecomposition();
      deformation_basis_ = piv_chol.get_Q();
      singular_values_ = piv_chol.get_Lambda().cwiseSqrt();
      gamma_ = singular_values_ *
               deformation_basis_.cwiseAbs().colwise().maxCoeff().transpose();
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  int get_parameterDimension() const { return deformation_basis_.cols(); }
  const Eigen::VectorXd &get_gamma() const { return gamma_; }
  //////////////////////////////////////////////////////////////////////////////
  int set_parameterDimension(int dim) {
    if (dim < deformation_basis_.cols()) {
      deformation_basis_.conservativeResize(deformation_basis_.rows(), dim);
      singular_values_ = singular_values_.block(0, 0, dim, dim);
      gamma_ = singular_values_ *
               deformation_basis_.cwiseAbs().colwise().maxCoeff().transpose();
    }
    return deformation_basis_.cols();
  }
  //////////////////////////////////////////////////////////////////////////////
  Geometry get_geometryRealization(const Eigen::VectorXd &param) {
    eigen_assert(param.size() == deformation_basis_.cols() &&
                 "parameter dimension mismatch");
    V = deformation_basis_ * param;
    Eigen::Map<Eigen::MatrixXd> Vfield(V.data(), points_.cols(), 3);
    for (auto i = 0; i < p_; ++i)
      PE_[i] = points_.block(0, i * n_ * n_, 3, n_ * n_) +
               Vfield.block(i * n_ * n_, 0, n_ * n_, 3).transpose();
    return Geometry(PE_);
  }
  //////////////////////////////////////////////////////////////////////////////
  Geometry get_DisplacementVector(const Eigen::Index i) {
    eigen_assert(i < deformation_basis_.cols() && i > 0 &&
                 "parameter dimension mismatch");
    V = deformation_basis_.col(i);
    Eigen::Map<Eigen::MatrixXd> Vfield(V.data(), points_.cols(), 3);
    for (auto i = 0; i < p_; ++i)
      PE_[i] = Vfield.block(i * n_ * n_, 0, n_ * n_, 3).transpose();
    return Geometry(PE_);
  }
  //////////////////////////////////////////////////////////////////////////////
  const Eigen::MatrixXd &get_DisplacementFields() const {
    return deformation_basis_;
  }
  Eigen::VectorXd get_maxPerturbation() const {
    // compute maximum pointwise deformation
    Eigen::VectorXd retval =
        (deformation_basis_.cwiseAbs() * singular_values_.diagonal());
    Eigen::Map<Eigen::MatrixXd> max_def(retval.data(), points_.cols(), 3);
    retval = max_def.rowwise().norm();
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Eigen::MatrixXd &get_SingularValues() const { return singular_values_; }
  //////////////////////////////////////////////////////////////////////////////

 private:
  std::vector<Eigen::MatrixXd> PE_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> points_;
  Eigen::MatrixXd deformation_basis_;
  Eigen::MatrixXd singular_values_;
  Eigen::VectorXd gamma_;
  Eigen::MatrixXd V;
  int p_;
  int m_;
  int n_;
};
}  // namespace UQ
}  // namespace Bembel

#endif
