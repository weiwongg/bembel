
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>

#include "Bembel/src/util/FormalSum.hpp"

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry geometry("sphere.dat");

  // Define evaluation points for potential field, a tensor product grid of
  // 7*7*7 points in [-.1,.1]^3
  MatrixXd gridpoints = Util::makeTensorProductGrid(
      VectorXd::LinSpaced(10, -.25, .25), VectorXd::LinSpaced(10, -.25, .25),
      VectorXd::LinSpaced(10, -.25, .25));

  // Define analytical solution using lambda function, in this case a harmonic
  // function, see Data.hpp
  std::function<double(Vector3d)> fun = [](Vector3d in) {
    return Data::HarmonicFunction(in);
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
      AnsatzSpace<LaplaceDoubleLayerOperator> ansatz_space_helm(
          geometry, refinement_level, polynomial_degree);
      AnsatzSpace<MassMatrixScalarDisc> ansatz_space_mass(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<double>, LaplaceDoubleLayerOperator>
          disc_lf(ansatz_space_helm);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<H2Matrix<double>, LaplaceDoubleLayerOperator>
          disc_op_double(ansatz_space_helm);
      disc_op_double.compute();
      DiscreteLocalOperator<MassMatrixScalarDisc> disc_op_mass(ansatz_space_mass);
      disc_op_mass.compute();

      SparseMatrix<double> mass =
          (-0.5 * disc_op_mass.get_discrete_operator()).eval();
      FormalSum<SparseMatrix<double>, H2Matrix<double>> system_matrix(
          mass, disc_op_double.get_discrete_operator());

      // solve system
      GMRES<FormalSum<SparseMatrix<double>, H2Matrix<double>>,
            IdentityPreconditioner>
          gmres;
      gmres.compute(system_matrix);
      auto rho = gmres.solve(disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<LaplaceDoubleLayerPotential<LaplaceDoubleLayerOperator>,
                        LaplaceDoubleLayerOperator>
          disc_pot(ansatz_space_helm);
      disc_pot.set_cauchy_data(rho);
      auto pot = disc_pot.evaluate(gridpoints);

      // print error
      std::cout << maxPointwiseError<double>(pot, gridpoints, fun) << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
