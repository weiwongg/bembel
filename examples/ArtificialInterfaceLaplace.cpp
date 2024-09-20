
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LinearOperator>

#include "Data.hpp"
#include "Grids.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;

  int interface_polynomial_degree = 10;
  Geometry geometry("../geo/sphere.dat");
  Geometry interface_geometry("../geo/centered_cube.dat");

  // Define analytical solution using lambda function, in this case the
  // Laplace fundamental solution centered on (0.1,0.1,0.1), see Data.hpp
  Vector3d center = Vector3d(0.1, 0.1, 0.1);
  const std::function<double(Vector3d)> fun = [center](Vector3d pt) {
    return Data::LaplaceFundamentalSolution(pt, center);
  };

  // Define grid points
  MatrixXd gridpoints = Util::makeSphereGrid(6., 3);
  ArtificialInterface<LaplaceSingleLayerOperator,
                      LaplaceSingleLayerPotential<LaplaceSingleLayerOperator>,
                      LaplaceDoubleLayerOperator,
                      LaplaceDoubleLayerPotential<LaplaceDoubleLayerOperator>>
      artificial_interface(interface_geometry, interface_polynomial_degree);
  Matrix<double, Dynamic, 3> interfacepoints =
      artificial_interface.get_gridpoints();

  // Iterate over polynomial degree.
  for (int polynomial_degree : {0, 1, 2, 3}) {
    std::cout << std::endl;
    // Initialize and log-file
    // Iterate over refinement levels
    for (int refinement_level = 0; refinement_level < 4 + 1 - polynomial_degree;
         ++refinement_level) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<LaplaceSingleLayerOperator> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<double>, LaplaceSingleLayerOperator>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<H2Matrix<double>, LaplaceSingleLayerOperator> disc_op(
          ansatz_space);
      disc_op.compute();

      // solve system
      ConjugateGradient<H2Matrix<double>, Lower | Upper, IdentityPreconditioner>
          cg;
      cg.compute(disc_op.get_discrete_operator());
      VectorXd rho = cg.solve(disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<LaplaceSingleLayerPotential<LaplaceSingleLayerOperator>,
                        LaplaceSingleLayerOperator>
          disc_pot(ansatz_space);
      disc_pot.set_cauchy_data(rho);
      VectorXd pot = disc_pot.evaluate(gridpoints);
      VectorXd pot_interface = disc_pot.evaluate(interfacepoints);
      DiscretePotential<
          LaplaceSingleLayerPotentialGradient<LaplaceSingleLayerOperator>,
          LaplaceSingleLayerOperator>
          disc_pot_grad(ansatz_space);
      disc_pot_grad.set_cauchy_data(rho);
      Matrix<double, Dynamic, 3> pot_interface_gradient =
          disc_pot_grad.evaluate(interfacepoints);

      // do artificial interface stuff
      artificial_interface.setBoundaryDataWithInterpolation(
          pot_interface, artificial_interface.potentialGradientToNeumann(
                             pot_interface_gradient));
      VectorXd pot_fast = artificial_interface.evaluate(gridpoints);

      std::cout << (pot_fast - pot).cwiseAbs().maxCoeff() << std::endl;
      assert((pot_fast - pot).cwiseAbs().maxCoeff() < 1e-6);
    }
  }

  return 0;
}
