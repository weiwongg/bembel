#define _USE_MATH_DEFINES

#include <Bembel/AnsatzSpace>
#include <Bembel/DiscreteFunction>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>
#include <Bembel/Solver>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <cmath>
#include <iostream>

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
  for (auto polynomial_degree : {1, 2, 3, 4}) {
    std::cout << "Degree " << polynomial_degree << std::endl;
    int MAX_LVL = 7;

    std::vector<Eigen::SparseMatrix<double>> Ps;

    // Iterate over refinement levels
    std::cout << std::left << std::setw(8) << "Level" << std::left
              << std::setw(8) << "Iters" << std::left << std::setw(15)
              << "L2 norm" << std::left << std::setw(15) << "H1 norm"
              << std::left << std::setw(15) << "Time/sec" << std::endl;

    for (int refinement_level = 0; refinement_level <= MAX_LVL;
         ++refinement_level) {
      // Build ansatz space
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      ansatz_space.compute_prolongation();
      Ps.push_back(ansatz_space.get_prolongation_matrix());
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

      AnsatzSpace<MassMatrixScalarCont> temp_ansatz_space(
          geometry, refinement_level, polynomial_degree);
      DiscreteLocalOperator<MassMatrixScalarCont> disc_identity_op(
          temp_ansatz_space);
      disc_identity_op.compute();
      disc_identity_op.get_discrete_operator();

      // prepare solver for the mass matrix
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      // Compute the numerical factorization
      solver.compute(disc_identity_op.get_discrete_operator());
      Eigen::VectorXd constant_one =
          solver.solve(disc_cf.get_discrete_linear_form());

      // solve linear system of equation with matrix free method
      Eigen::Matrix<double, Eigen::Dynamic, 1> x;
      auto start = std::chrono::steady_clock::now();
      int iters = MG::mmg(constant_one, disc_op.get_discrete_operator(),
                          disc_cf.get_discrete_linear_form(),
                          disc_lf.get_discrete_linear_form(), x, Ps,
                          refinement_level, 1e-10);
      auto end = std::chrono::steady_clock::now();
      // Store the time difference between start and end
      auto diff = end - start;

#if 1
      DiscreteFunction<LaplaceBeltramiScalar> approx_refsol(ansatz_space, refsol);
      // approx_refsol.plot("refsol" + std::to_string(polynomial_degree) + "_" +
      //                std::to_string(refinement_level) + ".vtp");
      DiscreteFunction<LaplaceBeltramiScalar> disc_fun(ansatz_space, x);
      //  disc_fun.plot("plot" + std::to_string(polynomial_degree) + "_" +
      //                std::to_string(refinement_level) + ".vtp");
      auto err = approx_refsol - disc_fun;
      std::cout << std::left << std::setw(8) << refinement_level << std::left
                << std::setw(8) << iters << std::left << std::setw(15)
                << err.norm_l2() << std::left << std::setw(15) << err.norm_h1()
                << std::left << std::setw(15)
                << std::chrono::duration<double>(diff).count() << std::endl;
#endif
    }
  }
  // The VTKwriter sets up initial geomety information.
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
