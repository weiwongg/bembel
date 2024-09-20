
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
  GeometryDeformer(const std::string &filename, double scale = 1,
                   Eigen::Vector3d shift = Eigen::VectorXd::Zero(3)) {
    init_GeometryDeformer(filename, scale, shift);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init_GeometryDeformer(const std::string &filename, double scale = 1,
                             Eigen::Vector3d shift = Eigen::VectorXd::Zero(3)) {
    IO::readPointsAscii(&PE_, &p_, &m_, filename);
    n_ = m_;
    points_.resize(3, p_ * n_ * n_);
    for (auto i = 0; i < p_; ++i)
      points_.block(0, i * n_ * n_, 3, n_ * n_) = PE_[i];
    points_ = scale * (points_ + shift.replicate(1, points_.cols()));
    {
      std::vector<int> I;
      for (auto i = 0; i < points_.cols(); ++i) I.push_back(i);
      std::sort(I.begin(), I.end(), lexiCompareInd(points_));
      std::vector<Eigen::Triplet<double>> trips_map;
      int current = I[0];
      int idx = 0;
      std::vector<int> J;
      J.push_back(current);
      for (std::vector<int>::iterator it = I.begin(); it != I.end(); ++it) {
        if ((points_.col(*(it)) - points_.col(current)).norm() >
            Bembel::Constants::pt_comp_tolerance) {
          current = *it;
          J.push_back(current);
          ++idx;
        }
        trips_map.push_back(Eigen::Triplet<double>(idx, *it, 1));
      }
      std::cout << "number of points is " << points_.cols() << std::endl;
      std::cout << "number of unique points is " << J.size() << std::endl;
      invJ_.conservativeResize(J.size(), points_.cols());
      invJ_.setFromTriplets(trips_map.begin(), trips_map.end());
      Eigen::VectorXd degree_vector =
          invJ_ * Eigen::VectorXd::Ones(points_.cols());
      unique_pts_ = points_ * invJ_.transpose() *
                    degree_vector.cwiseInverse().asDiagonal();
      std::cout << "scatter error: " << (unique_pts_ * invJ_ - points_).norm()
                << std::endl;
      // IO::bin2Mat("scaled_basis.bin",&scaled_deformation_basis_);
      IO::read_binary("scaled_basis.bin", scaled_deformation_basis_);
    }
  }

  int get_parameterDimension() const {
    return scaled_deformation_basis_.cols();
  }
  //////////////////////////////////////////////////////////////////////////////
  Geometry get_geometryRealization(const Eigen::VectorXd &param,
                                   double scale_m = 1.0) {
    eigen_assert(param.size() == (scaled_deformation_basis_.cols()) &&
                 "parameter dimension mismatch");

    V = scale_m * (scaled_deformation_basis_ * param);
    eigen_assert(V.rows() == (unique_pts_.cols() * 3) &&
                 "parameter dimension mismatch");
    V.resize(unique_pts_.cols(), 3);
    Eigen::MatrixXd Vfield = invJ_.transpose() * V;
    // Eigen::Map<Eigen::MatrixXd> Vfield(V.data(), points_.cols(), 3);
    for (auto i = 0; i < p_; ++i)
      PE_[i] = points_.block(0, i * n_ * n_, 3, n_ * n_) +
               Vfield.block(i * n_ * n_, 0, n_ * n_, 3).transpose();
    return Geometry(PE_);
  }

  Eigen::VectorXd get_maxPerturbation(double scale_m = 1.0) const {
    // compute maximum pointwise deformation
    Eigen::VectorXd retval =
        (scale_m * (scaled_deformation_basis_).cwiseAbs().rowwise().sum());
    Eigen::Map<Eigen::MatrixXd> max_box(retval.data(), points_.cols(), 3);
    max_box = max_box + points_.cwiseAbs().transpose();
    retval = max_box.colwise().maxCoeff();
    return retval;
  }

 private:
  std::vector<Eigen::MatrixXd> PE_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> points_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> unique_pts_;
  Eigen::SparseMatrix<double> invJ_;
  Eigen::MatrixXd scaled_deformation_basis_;
  Eigen::MatrixXd V;
  int p_;
  int m_;
  int n_;
};
}  // namespace UQ
}  // namespace Bembel

#endif
