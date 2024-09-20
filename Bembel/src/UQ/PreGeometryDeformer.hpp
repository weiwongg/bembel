
// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_UQ_PREGEOMETRYDEFORMER_H_
#define BEMBEL_UQ_PREGEOMETRYDEFORMER_H_

namespace Bembel {
namespace UQ {

struct lexiCompareInd {
  const double *array;
  int N;
  lexiCompareInd(const Eigen::MatrixXd &A)
      : array(&(A(0, 0))), N((int)A.rows()){};

  bool operator()(const int &i, const int &j) {
    return std::lexicographical_compare(array + i * N, array + i * N + N,
                                        array + j * N, array + j * N + N);
  }
};

class PreGeometryDeformer {
 public:
  PreGeometryDeformer(){};
  //////////////////////////////////////////////////////////////////////////////
  PreGeometryDeformer(const std::string &filename, double scale = 1,
                      Eigen::Vector3d shift = Eigen::VectorXd::Zero(3)) {
    init_PreGeometryDeformer(filename, scale, shift);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init_PreGeometryDeformer(
      const std::string &filename, double scale = 1,
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
      KernelMatern<Eigen::Matrix<double, 3, Eigen::Dynamic>> K_m(unique_pts_);
      PivotedCholesky piv_chol_m(K_m, 1e-3);
      piv_chol_m.computeSpectralDecomposition();
      deformation_basis_m_ = piv_chol_m.get_Q();
      singular_values_m_ = piv_chol_m.get_Lambda().cwiseSqrt();
      gamma_m_ =
          singular_values_m_ *
          deformation_basis_m_.cwiseAbs().colwise().maxCoeff().transpose();
      std::ofstream file_evalues;
      file_evalues.open("evalues.txt");
      for (auto l = 0; l < singular_values_m_.cols(); ++l)
        file_evalues << l << "\t" << singular_values_m_(l, l) << std::endl;
      file_evalues.close();

      // Bembel::UQ::OnlineCor cor(num_steps, max_k, 1e-10);
      Eigen::MatrixXd temp = deformation_basis_m_ * singular_values_m_;
      // IO::print2bin("scaled_basis.bin", temp);
      IO::write_binary("scaled_basis.bin", temp);
    }
  }

  int get_parameterDimension() const { return deformation_basis_m_.cols(); }
  //////////////////////////////////////////////////////////////////////////////
  Geometry get_geometryRealization(const Eigen::VectorXd &param,
                                   double scale_m = 1.0) {
    eigen_assert(param.size() == (deformation_basis_m_.cols()) &&
                 "parameter dimension mismatch");

    V = scale_m * deformation_basis_m_ * (singular_values_m_ * param);
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
        (scale_m * (deformation_basis_m_ * singular_values_m_.diagonal())
                       .cwiseAbs()
                       .rowwise()
                       .sum());
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
  Eigen::MatrixXd deformation_basis_m_;
  Eigen::MatrixXd singular_values_m_;
  Eigen::VectorXd gamma_m_;
  Eigen::MatrixXd V;
  int p_;
  int m_;
  int n_;
};
}  // namespace UQ
}  // namespace Bembel

#endif
