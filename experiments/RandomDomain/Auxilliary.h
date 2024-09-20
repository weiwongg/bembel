#ifndef RANDOMDOMAIN_AUXILLIARY_H_
#define RANDOMDOMAIN_AUXILLIARY_H_

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

// Bembel
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/src/util/FormalSum.hpp>
#include <Bembel/src/util/GenericMatrix.hpp>

///////////////////////////////////////////////////////////////////////////////
// define integrand
///////////////////////////////////////////////////////////////////////////////
///
std::vector<Eigen::MatrixXcd> integrand(
    const Bembel::Geometry &deformed_geometry,
    const Eigen::Matrix<double, -1, 3> &gridpoints,
    const int refinement_level_BEM, const int polynomial_degree_BEM,
    const std::complex<double> &wavenumber) {
  using namespace Bembel;
  using namespace Eigen;
  Vector3d d(1.0, 0.0, 0.0);
  const std::function<std::complex<double>(Vector3d)> fun = [wavenumber,
                                                             d](Vector3d pt) {
    return std::exp(-std::complex<double>(0, 1) * wavenumber * pt.dot(d));
  };
  const std::function<Vector3cd(Vector3d)> fun_grad = [wavenumber, d,
                                                       &fun](Vector3d pt) {
    return (-std::complex<double>(0, 1) * wavenumber * fun(pt)) * d;
  };
  // put potential and its gradient into an std::vector
  std::vector<MatrixXcd> pot;
  // Build ansatz space
  AnsatzSpace<MassMatrixScalarDisc> ansatz_space_mass(
      deformed_geometry, refinement_level_BEM, polynomial_degree_BEM);
  AnsatzSpace<HelmholtzColtonKressOperatorDisc> ansatz_space_ck(
      deformed_geometry, refinement_level_BEM, polynomial_degree_BEM);

  // Set up load vector
  DiscreteLinearForm<HelmholtzColtonKressRhs, HelmholtzColtonKressOperatorDisc>
      disc_lf_ck(ansatz_space_ck);
  disc_lf_ck.get_linear_form().set_dirichlet(fun);
  disc_lf_ck.get_linear_form().set_neumann(fun_grad);
  disc_lf_ck.get_linear_form().set_wavenumber(wavenumber);
  disc_lf_ck.compute();

  DiscreteOperator<SparseMatrix<double>, MassMatrixScalarDisc> disc_op_mass(
      ansatz_space_mass);
  disc_op_mass.compute();

  SparseMatrix<std::complex<double>> mass =
      (0.5 * disc_op_mass.get_discrete_operator().cast<std::complex<double>>())
          .eval();

  //std::cout << "DOFs " << mass.rows() << std::endl;
  Eigen::VectorXcd rho;

  if (mass.rows() > 20000) {
    // Set up and compute discrete operator
    DiscreteOperator<H2Matrix<std::complex<double>>,
                     HelmholtzColtonKressOperatorDisc>
        disc_op_ck(ansatz_space_ck);
    disc_op_ck.get_linear_operator().set_wavenumber(wavenumber);
    disc_op_ck.compute();

    FormalSum<SparseMatrix<std::complex<double>>,
              H2Matrix<std::complex<double>>>
        system_matrix_ck(mass, disc_op_ck.get_discrete_operator());
    // solve system
    GMRES<FormalSum<SparseMatrix<std::complex<double>>,
                    H2Matrix<std::complex<double>>>,
          IdentityPreconditioner>
        gmres_ck;
    gmres_ck.set_restart(300);
    gmres_ck.setTolerance(1e-8);
    gmres_ck.compute(system_matrix_ck);
    rho = gmres_ck.solve(disc_lf_ck.get_discrete_linear_form());
  } else {
    // Set up and compute discrete operator
    DiscreteOperator<Eigen::MatrixXcd, HelmholtzColtonKressOperatorDisc>
        disc_op_ck(ansatz_space_ck);
    disc_op_ck.get_linear_operator().set_wavenumber(wavenumber);
    disc_op_ck.compute();
    // solve system
    PartialPivLU<MatrixXcd> lu;
    lu.compute(mass + disc_op_ck.get_discrete_operator());
    rho = lu.solve(disc_lf_ck.get_discrete_linear_form());
  }
  // evaluate potential
  DiscretePotential<
      HelmholtzSingleLayerPotential<HelmholtzColtonKressOperatorDisc>,
      HelmholtzColtonKressOperatorDisc>
      disc_pot(ansatz_space_ck);
  disc_pot.get_potential().set_wavenumber(wavenumber);
  disc_pot.set_cauchy_data(rho);
  DiscretePotential<
      HelmholtzSingleLayerPotentialGradient<HelmholtzColtonKressOperatorDisc>,
      HelmholtzColtonKressOperatorDisc>
      disc_pot_grad(ansatz_space_ck);
  disc_pot_grad.get_potential().set_wavenumber(wavenumber);
  disc_pot_grad.set_cauchy_data(rho);
  pot.push_back(disc_pot.evaluate(gridpoints));
  pot.push_back(disc_pot_grad.evaluate(gridpoints));
  return pot;
}  // end integrand
#endif
