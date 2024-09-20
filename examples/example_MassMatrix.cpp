
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>
#include <Eigen/Dense>
#include <iostream>

#include "Data.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry geometry("sphere.dat");

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
    for (auto refinement_level : {0, 1, 2, 3, 4, 5, 6, 7}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<MassMatrixScalarDisc> ansatz_space(geometry, refinement_level,
                                                     polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarDisc> disc_lf(
          ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteLocalOperator<MassMatrixScalarDisc> disc_op(ansatz_space);
      disc_op.compute();

      // solve system
      ConjugateGradient<SparseMatrix<double>, Lower | Upper,
                        IdentityPreconditioner>
          cg;
      cg.compute(disc_op.get_discrete_operator());
      auto rho = cg.solve(disc_lf.get_discrete_linear_form());
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
