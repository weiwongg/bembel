
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
                                                  Vector3d(0., 0., 0.));
      };
  const std::function<Vector3cd(Vector3d)> funGrad = [wavenumber](Vector3d pt) {
    return Data::HelmholtzFundamentalSolutionGrad(pt, wavenumber,
                                                  Vector3d(0., 0., 0.));
  };

  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {0, 1, 2, 3}) {
    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<HelmholtzAdjointDoubleLayerOperator> ansatz_space_helm(
          geometry, refinement_level, polynomial_degree);
      AnsatzSpace<MassMatrixScalarDisc> ansatz_space_mass(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<NeumannTrace<std::complex<double>>,
                         HelmholtzAdjointDoubleLayerOperator>
          disc_lf(ansatz_space_helm);
      disc_lf.get_linear_form().set_function(funGrad);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<H2Matrix<std::complex<double>>,
                       HelmholtzAdjointDoubleLayerOperator>
          disc_op_double(ansatz_space_helm);
      disc_op_double.get_linear_operator().set_wavenumber(wavenumber);
      disc_op_double.compute();
      DiscreteLocalOperator<MassMatrixScalarDisc> disc_op_mass(
          ansatz_space_mass);
      disc_op_mass.compute();

      SparseMatrix<std::complex<double>> mass =
          (-0.5 *
           disc_op_mass.get_discrete_operator().cast<std::complex<double>>())
              .eval();
      FormalSum<SparseMatrix<std::complex<double>>,
                H2Matrix<std::complex<double>>>
          system_matrix(mass, disc_op_double.get_discrete_operator());

      // solve system
      GMRES<FormalSum<SparseMatrix<std::complex<double>>,
                      H2Matrix<std::complex<double>>>,
            IdentityPreconditioner>
          gmres;
      gmres.compute(system_matrix);
      VectorXcd rho = gmres.solve(disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<
          HelmholtzSingleLayerPotential<HelmholtzAdjointDoubleLayerOperator>,
          HelmholtzAdjointDoubleLayerOperator>
          disc_pot(ansatz_space_helm);
      disc_pot.get_potential().set_wavenumber(wavenumber);
      disc_pot.set_cauchy_data(rho);
      VectorXcd pot = disc_pot.evaluate(gridpoints);

      // print error
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
