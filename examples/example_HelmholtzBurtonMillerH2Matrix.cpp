
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>

#include "Bembel/src/util/FormalSum.hpp"

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;

  std::complex<double> wavenumber(1., 0.);

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry geometry("sphere.dat");

  // Define evaluation points for scattered field, sphere of radius 2, 10*10
  // points.
  MatrixXd gridpoints = Util::makeSphereGrid(2, 10);

  // Define analytical solution using lambda function, in this case the
  // Helmholtz fundamental solution centered on 0, see Data.hpp
  const std::function<std::complex<double>(Vector3d)> fun =
      [wavenumber](Vector3d pt) {
        return Data::HelmholtzFundamentalSolution(pt, wavenumber,
                                                  Vector3d(0., 0.2, 0.));
      };
  const std::function<Vector3cd(Vector3d)> funGrad = [wavenumber](Vector3d pt) {
    return Data::HelmholtzFundamentalSolutionGrad(pt, wavenumber,
                                                  Vector3d(0., 0.2, 0.));
  };

  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3}) {
    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      /////////////////////////////////////////////////////////////////////////
      // Build ansatz spaces
      /////////////////////////////////////////////////////////////////////////
      AnsatzSpace<HelmholtzBurtonMillerKW> a_KW(geometry, refinement_level,
                                                polynomial_degree);
      AnsatzSpace<HelmholtzBurtonMillerSA> a_SA(geometry, refinement_level,
                                                polynomial_degree);
      AnsatzSpace<MassMatrixScalarCont> a_M(geometry, refinement_level,
                                            polynomial_degree);

      /////////////////////////////////////////////////////////////////////////
      // Assemble matrices
      /////////////////////////////////////////////////////////////////////////
      // Burton Miller KW
      DiscreteOperator<H2Matrix<std::complex<double>>, HelmholtzBurtonMillerKW>
          // DiscreteOperator<MatrixXcd, HelmholtzBurtonMillerKW>
          do_KW(a_KW);
      do_KW.get_linear_operator().set_wavenumber(wavenumber);
      do_KW.compute();
      // Burton Miller SA
      DiscreteOperator<H2Matrix<std::complex<double>>, HelmholtzBurtonMillerSA>
          // DiscreteOperator<MatrixXcd, HelmholtzBurtonMillerSA>
          do_SA(a_SA);
      do_SA.get_linear_operator().set_wavenumber(wavenumber);
      do_SA.compute();
      // Mass matrix
      DiscreteLocalOperator<MassMatrixScalarCont> do_M(a_M);
      do_M.compute();
      SparseMatrix<std::complex<double>> mass =
          (do_M.get_discrete_operator().cast<std::complex<double>>()).eval();

      /////////////////////////////////////////////////////////////////////////
      // Set up load vector
      /////////////////////////////////////////////////////////////////////////
      // test
      DiscreteLinearForm<NeumannTrace<std::complex<double>>,
                         HelmholtzBurtonMillerKW>
          disc_lf(a_KW);
      disc_lf.get_linear_form().set_function(funGrad);
      disc_lf.compute();
      // apply inverse mass matrix
      SparseLU<SparseMatrix<std::complex<double>>, COLAMDOrdering<int>> solver;
      solver.analyzePattern(mass);
      solver.factorize(mass);
      VectorXcd neumann = solver.solve(disc_lf.get_discrete_linear_form());

      /////////////////////////////////////////////////////////////////////////
      // build Burton Miller matrices
      /////////////////////////////////////////////////////////////////////////
      SparseMatrix<std::complex<double>> IKW_mass = -0.5 * mass;
      FormalSum<SparseMatrix<std::complex<double>>,
                H2Matrix<std::complex<double>>>
          IKW(IKW_mass, do_KW.get_discrete_operator());
      SparseMatrix<std::complex<double>> ISA_mass =
          -0.5 * do_SA.get_linear_operator().get_ieta() * mass;
      FormalSum<SparseMatrix<std::complex<double>>,
                H2Matrix<std::complex<double>>>
          ISA(ISA_mass, do_SA.get_discrete_operator());

      /////////////////////////////////////////////////////////////////////////
      // solve system
      /////////////////////////////////////////////////////////////////////////
      GMRES<FormalSum<SparseMatrix<std::complex<double>>,
                      H2Matrix<std::complex<double>>>,
            IdentityPreconditioner>
          gmres;
      gmres.compute(IKW);
      // check this
      VectorXcd dirichlet = gmres.solve(ISA * neumann);

      /////////////////////////////////////////////////////////////////////////
      // reference dirichlet
      /////////////////////////////////////////////////////////////////////////
      DiscreteLinearForm<DirichletTrace<std::complex<double>>,
                         HelmholtzBurtonMillerKW>
          disc_lf_d(a_KW);
      disc_lf_d.get_linear_form().set_function(fun);
      disc_lf_d.compute();
      VectorXcd dirichlet_ref =
          solver.solve(disc_lf_d.get_discrete_linear_form());

      // MatrixXcd showit(dirichlet.size(), 2);
      // showit << dirichlet, dirichlet_ref;
      // std::cout << showit << std::endl;

      /////////////////////////////////////////////////////////////////////////
      // evaluate scattered wave
      /////////////////////////////////////////////////////////////////////////
      DiscretePotential<HelmholtzSingleLayerPotential<HelmholtzBurtonMillerKW>,
                        HelmholtzBurtonMillerKW>
          pot_s(a_KW);
      pot_s.get_potential().set_wavenumber(wavenumber);
      pot_s.set_cauchy_data(neumann);
      DiscretePotential<HelmholtzDoubleLayerPotential<HelmholtzBurtonMillerKW>,
                        HelmholtzBurtonMillerKW>
          pot_d(a_KW);
      pot_d.get_potential().set_wavenumber(wavenumber);
      pot_d.set_cauchy_data(dirichlet);
      VectorXcd pot = pot_d.evaluate(gridpoints) - pot_s.evaluate(gridpoints);

      /////////////////////////////////////////////////////////////////////////
      // print error
      /////////////////////////////////////////////////////////////////////////
      std::cout << maxPointwiseError<std::complex<double>>(pot, gridpoints, fun)
                << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
