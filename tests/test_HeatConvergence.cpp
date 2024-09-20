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
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
  using namespace Bembel;
  using namespace Eigen;
  std::function<double(const Vector3d &)> init_u = [](const Vector3d &in) {
    return in(2);
  };

  Geometry geometry("sphere.dat");
  std::cout << "geometry loaded\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3}) {
    // Iterate over refinement levels
    std::cout << "Degree " << polynomial_degree << std::endl;

    std::cout << std::left << std::setw(8) << "Level" << std::left
              << std::setw(15) << "Error L2 norm" << std::left << std::setw(15)
              << "Time/sec" << std::endl;
    for (auto refinement_level : {0, 1, 2, 3, 4, 5}) {
      // Build ansatz space and preprare the mass and stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol(ansatz_space);
      disc_sol.compute_metric();
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_ref(ansatz_space);
      disc_sol_ref.compute_metric();

      // Heat Equation
      // u_t = r_u * \laplacian u

      // parameters
      double theta = 0.5;  // Crank-Nicolson method
      double delta_t = 0.0001;
      double diffusion_rate_u = 1.0;
      double t = 0;

      std::function<double(const Vector3d &)> real_u =
          [&t](const Vector3d &in) { return exp(-2.0 * t) * in(2); };

      // Initial solution of u at t=0
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u(ansatz_space, init_u);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u_ref(ansatz_space);

      // The left hand side of the linear systems of equations for u
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_u;

      solver_u.compute(disc_sol.get_mass_matrix() +
                       theta * delta_t * diffusion_rate_u *
                           disc_sol.get_stiffness_matrix());

      // Sampling
      double error_l2_squared = 0;

      // Solve the solutions of u at the end of the time step by step
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i <= 10000; ++i) {
        if (i % 100 == 0) {
          // save solutions of the u per 1000 iterations
          disc_sol_u_ref.set_function(real_u);
          Eigen::Matrix<double, Eigen::Dynamic, 1> u =
              disc_sol_u.get_global_fun();
          Eigen::Matrix<double, Eigen::Dynamic, 1> u_ref =
              disc_sol_u_ref.get_global_fun();
          error_l2_squared += (u - u_ref).transpose() *
                              disc_sol_ref.get_mass_matrix() * (u - u_ref);
        }
        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs_u =
            (disc_sol.get_mass_matrix() - (1 - theta) * delta_t *
                                              diffusion_rate_u *
                                              disc_sol.get_stiffness_matrix()) *
            disc_sol_u.get_global_fun();

        // update the u
        Eigen::Matrix<double, Eigen::Dynamic, 1> u = solver_u.solve(rhs_u);
        disc_sol_u.set_function(u);
        t = t + delta_t;
      }
      auto end = std::chrono::steady_clock::now();
      // Store the time difference between start and end
      auto diff = end - start;
      std::cout << std::left << std::setw(8) << refinement_level << std::left
                << std::setw(15) << sqrt(error_l2_squared / 201) << std::left
                << std::setw(15) << std::chrono::duration<double>(diff).count()
                << std::endl;
    }
  }
  return 0;
}
