
#include <iostream>

#include <Eigen/Dense>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>

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
  std::function<Vector3d(Vector3d)> funGrad = [](Vector3d in) {
    return Data::HarmonicFunctionGrad(in);
  };
  std::function<double(Vector3d)> one = [](Vector3d in) { return 1.; };

  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3}) {
    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<LaplaceHypersingularOperator> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<NeumannTrace<double>, LaplaceHypersingularOperator>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(funGrad);
      disc_lf.compute();

      // Set up one
      DiscreteLinearForm<DirichletTrace<double>, LaplaceHypersingularOperator>
          one_lf(ansatz_space);
      one_lf.get_linear_form().set_function(one);
      one_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<MatrixXd, LaplaceHypersingularOperator> disc_op(
          ansatz_space);
      disc_op.compute();

      // solve system
      PartialPivLU<MatrixXd> lu;
      lu.compute(disc_op.get_discrete_operator() +
                 one_lf.get_discrete_linear_form() *
                     one_lf.get_discrete_linear_form().transpose());
      VectorXd rho = lu.solve(-disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<
          LaplaceDoubleLayerPotential<LaplaceHypersingularOperator>,
          LaplaceHypersingularOperator>
          disc_pot(ansatz_space);
      disc_pot.set_cauchy_data(rho);
      VectorXd pot = disc_pot.evaluate(gridpoints);

      // factor out constant
      pot -=
          0.5 * VectorXd::Ones(pot.rows()) * (pot.maxCoeff() + pot.minCoeff());

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
