// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_DISCRETEFUNCTIONINIT_H_
#define BEMBEL_ANSATZSPACE_DISCRETEFUNCTIONINIT_H_

namespace Bembel {
template <typename Scalar, unsigned int DF, typename LinOp>
struct DiscreteFunctionInit {};

// continuous
template <typename Scalar, typename LinOp>
struct DiscreteFunctionInit<Scalar, DifferentialForm::Continuous, LinOp> {
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<LinOp>::Form>()>
  interpolation(const AnsatzSpace<LinOp> &ansatz_space, int level,
                int polynomial_degree, Eigen::SparseMatrix<Scalar> mass,
                std::function<Scalar(const Eigen::Vector3d &)> func) const {
    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>>
        solver;
    // Compute the ordering permutation vector from the structural pattern of
    solver.analyzePattern(mass);
    // Compute the numerical factorization
    solver.factorize(mass);
    AnsatzSpace<MassMatrixScalarCont> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLinearForm<DirichletTrace<Scalar>, MassMatrixScalarCont> disc_lf(
        temp_ansatz_space);
    disc_lf.get_linear_form().set_function(func);
    disc_lf.compute();
    // Use the factors to solve the linear system
    return solver.solve(disc_lf.get_discrete_linear_form());
  };
  Eigen::SparseMatrix<Scalar> get_mass(const AnsatzSpace<LinOp> &ansatz_space,
                                       int level, int polynomial_degree) {
    AnsatzSpace<MassMatrixScalarCont> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLocalOperator<MassMatrixScalarCont> disc_identity_op(
        temp_ansatz_space);
    disc_identity_op.compute();
    return disc_identity_op.get_discrete_operator();
  };
  Eigen::SparseMatrix<Scalar> get_stiffness(
      const AnsatzSpace<LinOp> &ansatz_space, int level,
      int polynomial_degree) {
    AnsatzSpace<StiffnessMatrixScalarCont> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLocalOperator<StiffnessMatrixScalarCont> disc_lb_op(
        temp_ansatz_space);
    disc_lb_op.compute();
    return disc_lb_op.get_discrete_operator();
  };
};

// discontinuous
template <typename Scalar, typename LinOp>
struct DiscreteFunctionInit<Scalar, DifferentialForm::Discontinuous, LinOp> {
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<LinOp>::Form>()>
  interpolation(const AnsatzSpace<LinOp> &ansatz_space, int level,
                int polynomial_degree, Eigen::SparseMatrix<Scalar> mass,
                std::function<Scalar(const Eigen::Vector3d &)> func) const {
    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>>
        solver;
    // Compute the ordering permutation vector from the structural pattern of
    solver.analyzePattern(mass);
    // Compute the numerical factorization
    solver.factorize(mass);
    AnsatzSpace<MassMatrixScalarDisc> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLinearForm<DirichletTrace<Scalar>, MassMatrixScalarDisc> disc_lf(
        temp_ansatz_space);
    disc_lf.get_linear_form().set_function(func);
    disc_lf.compute();
    // Use the factors to solve the linear system
    return solver.solve(disc_lf.get_discrete_linear_form());
  };
  Eigen::SparseMatrix<Scalar> get_mass(const AnsatzSpace<LinOp> &ansatz_space,
                                       int level, int polynomial_degree) {
    AnsatzSpace<MassMatrixScalarDisc> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLocalOperator<MassMatrixScalarDisc> disc_identity_op(
        temp_ansatz_space);
    disc_identity_op.compute();
    return disc_identity_op.get_discrete_operator();
  };
  Eigen::SparseMatrix<Scalar> get_stiffness(
      const AnsatzSpace<LinOp> &ansatz_space, int level,
      int polynomial_degree) {
    AnsatzSpace<StiffnessMatrixScalarDisc> temp_ansatz_space(
        ansatz_space.get_geometry(), level, polynomial_degree);
    DiscreteLocalOperator<StiffnessMatrixScalarDisc> disc_lb_op(
        temp_ansatz_space);
    disc_lb_op.compute();
    return disc_lb_op.get_discrete_operator();
  };
};
}  // namespace Bembel
#endif
