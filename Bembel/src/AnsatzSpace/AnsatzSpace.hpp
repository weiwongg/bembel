// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_ANSATZSPACE_H_
#define BEMBEL_ANSATZSPACE_ANSATZSPACE_H_

namespace Bembel {
/**
 *  \ingroup AnsatzSpace
 *  \brief The AnsatzSpace is the class that handles the assembly of the
 *discrete basis.
 *
 *	It invokes a superspace and uses the Glue and Projektor class to
 *assemble a transformation matrix, which relates the superspace to the desired
 *basis.
 */
template <typename Derived>
class AnsatzSpace {
  typedef typename LinearOperatorTraits<Derived>::Scalar Scalar;

 public:
  enum { Form = LinearOperatorTraits<Derived>::Form };
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  AnsatzSpace() {}
  AnsatzSpace(const AnsatzSpace &other) {
    super_space_ = other.super_space_;
    knot_repetition_ = other.knot_repetition_;
    glue_matrix_ = other.glue_matrix_;
    transformation_matrix_ = other.transformation_matrix_;
    interpolation_matrix_ = other.interpolation_matrix_;
    prolongation_matrix_ = other.prolongation_matrix_;
  }
  AnsatzSpace(AnsatzSpace &&other) {
    super_space_ = other.super_space_;
    knot_repetition_ = other.knot_repetition_;
    glue_matrix_ = other.glue_matrix_;
    transformation_matrix_ = other.transformation_matrix_;
    interpolation_matrix_ = other.interpolation_matrix_;
    prolongation_matrix_ = other.prolongation_matrix_;
  }
  AnsatzSpace &operator=(AnsatzSpace other) {
    super_space_ = other.super_space_;
    knot_repetition_ = other.knot_repetition_;
    glue_matrix_ = other.glue_matrix_;
    transformation_matrix_ = other.transformation_matrix_;
    interpolation_matrix_ = other.interpolation_matrix_;
    prolongation_matrix_ = other.prolongation_matrix_;
    return *this;
  }

  AnsatzSpace(const Geometry &geometry, int refinement_level,
              int polynomial_degree, int knot_repetition = 1) {
    init_AnsatzSpace(geometry, refinement_level, polynomial_degree,
                     knot_repetition);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    init_Ansatzspace
  //////////////////////////////////////////////////////////////////////////////
  void init_AnsatzSpace(const Geometry &geometry, int refinement_level,
                        int polynomial_degree, int knot_repetition) {
    knot_repetition_ = knot_repetition;
    super_space_.init_SuperSpace(geometry, refinement_level, polynomial_degree);
    Projector<Derived> proj(super_space_, knot_repetition_);
    Glue<Derived> glue(super_space_, proj);
    glue_matrix_ = glue.get_glue_matrix();
    transformation_matrix_ =
        proj.get_projection_matrix() * glue.get_glue_matrix();
    Interpolation<Derived> interpolation(super_space_);
    interpolation_matrix_ = interpolation.get_interpolation_matrix();
    return;
  }
  void compute_prolongation() {
    AnsatzSpace<Derived> refined_ansatz_space(
        super_space_.get_geometry(), super_space_.get_refinement_level() + 1,
        super_space_.get_polynomial_degree());

    // g_l^T * g_l (diagonal matrix)
    Eigen::SparseMatrix<double> degree_diag =
        refined_ansatz_space.get_glue_matrix().transpose() *
        refined_ansatz_space.get_glue_matrix();

    // (g_l^T * g_l)^{-1} (diagonal matrix)
    Eigen::SparseMatrix<double> inverse_diag;
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(refined_ansatz_space.get_glue_matrix().cols());
    for (int i = 0; i < degree_diag.cols(); ++i) {
      tripletList.push_back(T(i, i, 1.0 / (degree_diag.coeff(i, i))));
    }
    inverse_diag.resize(degree_diag.rows(), degree_diag.cols());
    inverse_diag.setFromTriplets(tripletList.begin(), tripletList.end());

    // prolongation procedure
    // 0.5 * (g_{l+1}^T * g_{l+1})^{-1} * g_{l+1}^T * I_l^{l+1} * g_l *
    // coarse_global_fun_ the reason we multiply 0.5 in the begining is because
    // we use the scaled B-spline basis function in the Bembel
    prolongation_matrix_ = 0.5 * inverse_diag *
                           refined_ansatz_space.get_glue_matrix().transpose() *
                           interpolation_matrix_ * glue_matrix_;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    getters
  //////////////////////////////////////////////////////////////////////////////
  const SuperSpace<Derived> &get_superspace() const { return super_space_; }
  int get_knot_repetition() const { return knot_repetition_; }
  int get_refinement_level() const {
    return super_space_.get_refinement_level();
  }
  int get_polynomial_degree() const {
    return super_space_.get_polynomial_degree();
  }
  int get_number_of_elements() const {
    return super_space_.get_number_of_elements();
  }
  int get_number_of_patches() const {
    return super_space_.get_number_of_patches();
  }
  int get_number_of_dofs() const { return transformation_matrix_.cols(); }
  const PatchVector &get_geometry() const {
    return super_space_.get_geometry();
  }
  const Eigen::SparseMatrix<double> &get_glue_matrix() const {
    return glue_matrix_;
  }
  const Eigen::SparseMatrix<double> &get_transformation_matrix() const {
    return transformation_matrix_;
  }
  const Eigen::SparseMatrix<double> &get_interpolation_matrix() const {
    return interpolation_matrix_;
  }
  const Eigen::SparseMatrix<double> &get_prolongation_matrix() const {
    return prolongation_matrix_;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    prolongate and restrict
  //////////////////////////////////////////////////////////////////////////////
  // prolongate
  Eigen::Matrix<
      typename LinearOperatorTraits<Derived>::Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>()>
  prolongate_one_step(Eigen::Matrix<Scalar, Eigen::Dynamic,
                                    getFunctionSpaceVectorDimension<
                                        LinearOperatorTraits<Derived>::Form>()>
                          coarse_global_fun_) {
    auto fine_global_fun_ = prolongation_matrix_ * coarse_global_fun_;
    return fine_global_fun_;
  }

  //////////////////////////////////////////////////////////////////////////////
  //    private member variables
  //////////////////////////////////////////////////////////////////////////////
 private:
  Eigen::SparseMatrix<double> glue_matrix_;
  Eigen::SparseMatrix<double> transformation_matrix_;
  Eigen::SparseMatrix<double> interpolation_matrix_;
  Eigen::SparseMatrix<double> prolongation_matrix_;
  SuperSpace<Derived> super_space_;
  int knot_repetition_;
};
}  // namespace Bembel
#endif
