// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_INTERPOLATION_H_
#define BEMBEL_ANSATZSPACE_INTERPOLATION_H_

namespace Bembel {

/**
 *  \ingroup AnsatzSpace
 *  \brief The interpolation solves interpolation problems
 * in the multigrid method.
 */
template <typename Derived>
class Interpolation {
  typedef typename LinearOperatorTraits<Derived>::Scalar Scalar;

 public:
  //////////////////////////////////////////////////////////////////////////////
  /// constructors
  //////////////////////////////////////////////////////////////////////////////
  Interpolation() {}
  Interpolation(const SuperSpace<Derived>& super_space) {
    init_Interpolation(super_space);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// init
  //////////////////////////////////////////////////////////////////////////////
  void init_Interpolation(const SuperSpace<Derived>& super_space) {
    makeInterpolationMatrix(super_space);
    dofs_before_interpolation_ = interpolation_matrix_.rows();
    dofs_after_interpolation_ = interpolation_matrix_.cols();
    return;
  }
  // global interpolation
  void makeInterpolationMatrix(const SuperSpace<Derived>& super_space) {
    int num_patches = super_space.get_number_of_patches();
    int polynomial_degree = super_space.get_polynomial_degree();
    int level = super_space.get_refinement_level();
    std::vector<double> knots =
        Spl::MakeUniformKnotVector(polynomial_degree + 1, pow(2, level) - 1);
    std::vector<double> uniform_refined_knots = Spl::MakeUniformKnotVector(
        polynomial_degree + 1, pow(2, level + 1) - 1);
    int num_basis = pow(2, level) + polynomial_degree;
    int num_fine_basis = pow(2, level + 1) + polynomial_degree;
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_patches * num_fine_basis * num_fine_basis *
                        (polynomial_degree + 1) * (polynomial_degree + 1));
    // the global basis functions in 2D are gotten by the tensor product of the
    // B-spline basis functions in 1D.
    for (int iy = 0; iy < num_fine_basis; ++iy) {
      for (int jx = 0; jx < num_fine_basis; ++jx) {
        int dy;
        Eigen::MatrixXd non_zero_alpha_y;
        int dx;
        Eigen::MatrixXd non_zero_alpha_x;
        std::tie(non_zero_alpha_y, dy) = knots_insertion_row(
            knots, uniform_refined_knots, polynomial_degree, level, iy);
        std::tie(non_zero_alpha_x, dx) = knots_insertion_row(
            knots, uniform_refined_knots, polynomial_degree, level, jx);
        // the non zeros entities
        for (int i = dy - polynomial_degree; i <= dy; ++i) {
          for (int j = dx - polynomial_degree; j <= dx; ++j) {
            // loop for the pathes (the local interpolation matrices are the
            // same for different patches)
            for (int k = 0; k < num_patches; ++k) {
              tripletList.push_back(
                  T(k * num_fine_basis * num_fine_basis + iy * num_fine_basis +
                        jx,
                    k * num_basis * num_basis + i * num_basis + j,
                    non_zero_alpha_y(i - dy + polynomial_degree) *
                        non_zero_alpha_x(j - dx + polynomial_degree)));
            }
          }
        }
      }
    }
    interpolation_matrix_.resize(num_patches * num_fine_basis * num_fine_basis,
                                 num_patches * num_basis * num_basis);
    interpolation_matrix_.setFromTriplets(tripletList.begin(),
                                          tripletList.end());
  }

  // local interpolation on each patch.
  Eigen::SparseMatrix<double> makeInterpolationMatrix2D(
      const SuperSpace<Derived>& super_space) {
    int polynomial_degree = super_space.get_polynomial_degree();
    int level = super_space.get_refinement_level();
    std::vector<double> knots =
        Spl::MakeUniformKnotVector(polynomial_degree + 1, pow(2, level) - 1);
    std::vector<double> uniform_refined_knots = Spl::MakeUniformKnotVector(
        polynomial_degree + 1, pow(2, level + 1) - 1);
    int num_basis = pow(2, level) + polynomial_degree;
    int num_fine_basis = pow(2, level + 1) + polynomial_degree;
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_fine_basis * num_fine_basis *
                        (polynomial_degree + 1) * (polynomial_degree + 1));
    // the global basis functions in 2D are gotten by the tensor product of the
    // B-spline basis functions in 1D.
    for (int iy = 0; iy < num_fine_basis; ++iy) {
      for (int jx = 0; jx < num_fine_basis; ++jx) {
        int dy;
        Eigen::MatrixXd non_zero_alpha_y;
        int dx;
        Eigen::MatrixXd non_zero_alpha_x;

        std::tie(non_zero_alpha_y, dy) = knots_insertion_row(
            knots, uniform_refined_knots, polynomial_degree, level, iy);
        std::tie(non_zero_alpha_x, dx) = knots_insertion_row(
            knots, uniform_refined_knots, polynomial_degree, level, jx);
        // the non zeros entities
        for (int i = dy - polynomial_degree; i <= dy; ++i) {
          for (int j = dx - polynomial_degree; j <= dx; ++j) {
            tripletList.push_back(
                T(iy * num_fine_basis + jx, i * num_basis + j,
                  non_zero_alpha_y(i - dy + polynomial_degree) *
                      non_zero_alpha_x(j - dx + polynomial_degree)));
          }
        }
      }
    }
    Eigen::SparseMatrix<double> A;
    A.resize(num_fine_basis * num_fine_basis, num_basis * num_basis);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
  }

  // Oslo-Algorithm for the knots insertion problem
  // ref:
  // https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v07/undervisningsmateriale/kap4.pdf
  Eigen::SparseMatrix<double> makeLocalInterpolationMatrices1D(
      const SuperSpace<Derived>& super_space) {
    int polynomial_degree = super_space.get_polynomial_degree();
    int level = super_space.get_refinement_level();
    std::vector<double> knots =
        Spl::MakeUniformKnotVector(polynomial_degree + 1, pow(2, level) - 1);
    std::vector<double> uniform_refined_knots = Spl::MakeUniformKnotVector(
        polynomial_degree + 1, pow(2, level + 1) - 1);
    int num_basis = pow(2, level) + polynomial_degree;
    int num_fine_basis = pow(2, level + 1) + polynomial_degree;
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_fine_basis * (polynomial_degree + 1));
    for (int i = 0; i < num_fine_basis; ++i) {
      for (int j = 0; j < polynomial_degree + 1; ++j) {
        int d;
        Eigen::MatrixXd non_zero_alpha;
        std::tie(non_zero_alpha, d) = knots_insertion_row(
            knots, uniform_refined_knots, polynomial_degree, level, i);
        tripletList.push_back(
            T(i, d - polynomial_degree + j, non_zero_alpha(j)));
      }
    }
    Eigen::SparseMatrix<double> A;
    A.resize(num_fine_basis, num_basis);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
  }
  std::tuple<Eigen::MatrixXd, int> knots_insertion_row(
      std::vector<double> knots, std::vector<double> uniform_refined_knots,
      int polynomial_degree, int level, int row_id) {
    int d;
    if (row_id <= polynomial_degree)
      d = polynomial_degree;
    else
      d = (row_id - polynomial_degree) / 2 + polynomial_degree;
    Eigen::MatrixXd non_zero_alpha = Eigen::MatrixXd::Identity(1, 1);
    for (int i = 0; i < polynomial_degree; ++i) {
      Eigen::MatrixXd R = Eigen::MatrixXd::Zero(i + 1, i + 2);
      for (int j = 0; j < i + 1; ++j) {
        R(j, j) = (knots[d + j + 1] - uniform_refined_knots[row_id + i + 1]) /
                  (knots[d + j + 1] - knots[d + j - i]);
        R(j, j + 1) = 1.0 - R(j, j);
      }
      non_zero_alpha = non_zero_alpha * R;
    }

    return std::make_tuple(non_zero_alpha, d);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// getter
  //////////////////////////////////////////////////////////////////////////////
  const Eigen::SparseMatrix<double>& get_interpolation_matrix() {
    return interpolation_matrix_;
  }

 private:
  Eigen::SparseMatrix<double> interpolation_matrix_;
  int dofs_before_interpolation_;
  int dofs_after_interpolation_;
};
}  // namespace Bembel

#endif
