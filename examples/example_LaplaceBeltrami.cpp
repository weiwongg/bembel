#define _USE_MATH_DEFINES

#include <Bembel/AnsatzSpace>
#include <Bembel/DiscreteFunction>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <iostream>
#include "Bembel/src/util/Rank1Update.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;
  IO::Stopwatch sw;
  std::function<double(const Vector3d &)> fun = [](const Vector3d &in) {
    // Spherical harmonics
    return sqrt(3 / M_PI) * in(2);
  };
  std::function<double(const Vector3d &)> refsol = [](const Vector3d &in) {
    return 0.5 * sqrt(3 / M_PI) * in(2);
  };

  std::function<double(const Vector3d &)> const_fun = [](const Vector3d &in) {
    return 1.0;
  };
  Geometry geometry("sphere.dat");
  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3, 4, 5}) {
    std::cout << "Degree " << polynomial_degree << std::endl;
    std::cout << std::left << std::setw(8) << "Level" << std::left
              << std::setw(15) << "L2 norm" << std::left << std::setw(15)
              << "H1 norm" << std::endl;

    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3, 4, 5, 6}) {
      // Build ansatz space
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up and compute discrete operator
      DiscreteLocalOperator<LaplaceBeltramiScalar> disc_op(ansatz_space);
      disc_op.compute();

      DiscreteLinearForm<DirichletTrace<double>, LaplaceBeltramiScalar> disc_cf(
          ansatz_space);
      disc_cf.get_linear_form().set_function(const_fun);
      disc_cf.compute();

      DiscreteLinearForm<DirichletTrace<double>, LaplaceBeltramiScalar> disc_lf(
          ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // solve linear system of equation with matrix free method
      Eigen::Rank1Update<Eigen::SparseMatrix<double>,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>>
          A(disc_op.get_discrete_operator(),
            disc_cf.get_discrete_linear_form());
      ConjugateGradient<
        Rank1Update<SparseMatrix<double>, Matrix<double, Dynamic, 1>>,
          Lower | Upper, IdentityPreconditioner>
          cg;
      cg.compute(A);
      auto x = cg.solve(disc_lf.get_discrete_linear_form());
      // std::cout << "CG:  #iterations: " << cg.iterations()
      //        << ", estimated error: " << cg.error() << std::endl;

#if 1
      DiscreteFunction<LaplaceBeltramiScalar> approx_refsol(ansatz_space, refsol);
      // approx_refsol.plot("refsol" + std::to_string(polynomial_degree) + "_" +
      //                std::to_string(refinement_level) + ".vtp");
      DiscreteFunction<LaplaceBeltramiScalar> disc_fun(ansatz_space, x);
      //  disc_fun.plot("plot" + std::to_string(polynomial_degree) + "_" +
      //                std::to_string(refinement_level) + ".vtp");
      auto err = approx_refsol - disc_fun;
      std::cout << std::left << std::setw(8) << refinement_level << std::left
                << std::setw(15) << err.norm_l2() << std::left << std::setw(15)
                << err.norm_h1() << std::endl;
#endif
    }
  }
  // The VTKwriter sets up initial geomety information.
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
