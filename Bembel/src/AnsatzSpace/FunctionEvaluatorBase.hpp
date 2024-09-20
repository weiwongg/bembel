// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_FUNCTIONEVALUATORBASE_H_
#define BEMBEL_ANSATZSPACE_FUNCTIONEVALUATORBASE_H_

namespace Bembel {
/**
 *  \ingroup AnsatzSpace
 *  \brief The FunctionEvaluatorBase provides means to evaluate coefficient
 * vectors as functions on the geometry.
 */
template <typename Derived, int output_dimension>
class FunctionEvaluatorBase {
 public:
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  FunctionEvaluatorBase() {}
  FunctionEvaluatorBase(const FunctionEvaluatorBase &other) = default;
  FunctionEvaluatorBase(FunctionEvaluatorBase &&other) = default;
  FunctionEvaluatorBase &operator=(FunctionEvaluatorBase other) {
    ansatz_space_ = other.ansatz_space_;
    fun_ = other.fun_;
    polynomial_degree_plus_one_squared_ =
        other.polynomial_degree_plus_one_squared_;
    return *this;
  }
  FunctionEvaluatorBase(const AnsatzSpace<Derived> &ansatz_space) {
    init_FunctionEvaluatorBase(ansatz_space);
    return;
  }
  FunctionEvaluatorBase(
      const AnsatzSpace<Derived> &ansatz_space,
      const Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                          Eigen::Dynamic, 1> &fun) {
    init_FunctionEvaluatorBase(ansatz_space, fun);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    init_Ansatzspace
  //////////////////////////////////////////////////////////////////////////////
  void init_FunctionEvaluatorBase(const AnsatzSpace<Derived> &ansatz_space) {
    ansatz_space_ = ansatz_space;
    auto polynomial_degree = ansatz_space_.get_polynomial_degree();
    polynomial_degree_plus_one_squared_ =
        (polynomial_degree + 1) * (polynomial_degree + 1);
    reordering_vector_ = ansatz_space_.get_superspace()
                             .get_mesh()
                             .get_element_tree()
                             .computeReorderingVector();
    return;
  }
  void init_FunctionEvaluatorBase(
      const AnsatzSpace<Derived> &ansatz_space,
      const Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                          Eigen::Dynamic, 1> &fun) {
    init_FunctionEvaluatorBase(ansatz_space);
    set_function(fun);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    setters
  //////////////////////////////////////////////////////////////////////////////
  void set_function(
      Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                    Eigen::Dynamic, Eigen::Dynamic>
          fun) {
    assert(fun.rows() == ansatz_space_.get_number_of_dofs());
    if (output_dimension == getFunctionSpaceVectorDimension<
                                       LinearOperatorTraits<Derived>::Form>()) {
      assert(fun.cols() == 1);
      auto longfun = (ansatz_space_.get_transformation_matrix() * fun).eval();
      fun_ = Eigen::Map<
          Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                        Eigen::Dynamic, output_dimension>>(
          longfun.data(), longfun.rows() / output_dimension, output_dimension);
    } else {
      assert(fun.cols() == output_dimension);
      fun_ = (ansatz_space_.get_transformation_matrix() * fun).eval();
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  //    private member variables
  //////////////////////////////////////////////////////////////////////////////
 protected:
  std::vector<int> reordering_vector_;
  AnsatzSpace<Derived> ansatz_space_;
  Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar, Eigen::Dynamic,
                output_dimension>
      fun_;
  int polynomial_degree_plus_one_squared_;
  FunctionEvaluatorEval<typename LinearOperatorTraits<Derived>::Scalar,
                        LinearOperatorTraits<Derived>::Form, Derived>
      eval_;
};

}  // namespace Bembel
#endif
