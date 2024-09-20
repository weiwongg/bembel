
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/LinearForm>
#include <Eigen/Dense>
#include <iostream>

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

  double h = 1e-8;
  Vector3d a(2., 2., 2.);
  Vector3d hx(h, 0., 0.);
  Vector3d hy(0., h, 0.);
  Vector3d hz(0., 0., h);

  Vector3cd t;
  t(0) = 0.5 / h * (fun(a + hx) - fun(a - hx));
  t(1) = 0.5 / h * (fun(a + hy) - fun(a - hy));
  t(2) = 0.5 / h * (fun(a + hz) - fun(a - hz));
  assert((funGrad(a) - t).norm() < 1e-8);

  std::function<std::complex<double>(Vector3d)> one = [](Vector3d in) {
    return 1.;
  };

  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3}) {
    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<HelmholtzHypersingularOperator> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<NeumannTrace<std::complex<double>>,
                         HelmholtzHypersingularOperator>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(funGrad);
      disc_lf.compute();

      // Set up one
      DiscreteLinearForm<DirichletTrace<std::complex<double>>,
                         HelmholtzHypersingularOperator>
          one_lf(ansatz_space);
      one_lf.get_linear_form().set_function(one);
      one_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<MatrixXcd, HelmholtzHypersingularOperator> disc_op(
          ansatz_space);
      disc_op.get_linear_operator().set_wavenumber(wavenumber);
      disc_op.compute();

      // solve system
      PartialPivLU<MatrixXcd> lu;
      lu.compute(disc_op.get_discrete_operator());
      VectorXcd rho = lu.solve(-disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<
          HelmholtzDoubleLayerPotential<HelmholtzHypersingularOperator>,
          HelmholtzHypersingularOperator>
          disc_pot(ansatz_space);
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
