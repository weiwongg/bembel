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
  std::function<double(const Eigen::Vector3d &)> heat_func =
      [](const Eigen::Vector3d &in) {
        return sin(M_PI * in(0)) * sin(M_PI * in(1)) * sin(M_PI * in(2));
      };
  Geometry geometry("bunny5.bpd", 10);
  std::cout << "geometry loaded\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {2}) {
    // Iterate over refinement levels
    std::cout << "Degree " << polynomial_degree << std::endl;

    std::cout << std::left << std::setw(8) << "Level" << std::left
              << std::setw(15) << "Error_L2" << std::left << std::setw(15)
              << "Error_H1" << std::left << std::setw(15) << "Time/sec"
              << std::endl;
    int MAX_LVL = 5;
    std::vector<Eigen::SparseMatrix<double>> Ps;
    for (int lvl = 0; lvl <= MAX_LVL - 1; ++lvl) {
      // Build ansatz space
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(geometry, lvl,
                                                      polynomial_degree);
      ansatz_space.compute_prolongation();
      Ps.push_back(ansatz_space.get_prolongation_matrix());
    }
    for (int refinement_level = 0; refinement_level < MAX_LVL;
         ++refinement_level) {
      // Build ansatz space, the mass, and the stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol(ansatz_space);
      disc_sol.compute_metric();
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space_ref(geometry, MAX_LVL,
                                                          polynomial_degree);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_ref(ansatz_space_ref);
      disc_sol_ref.compute_metric();

      // Heat Equation
      // u_t = r_u * \laplacian u

      // parameters
      double theta = 0.5;  // Crank-Nicolson method
      double delta_t = 0.001;
      double diffusion_rate_u = 1.0;
      double t = 0;

      // Initial solution of u at t=0 and heat source
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u(ansatz_space, init_u);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u_ref(ansatz_space_ref,
                                                             init_u);
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_heat(ansatz_space,
                                                            heat_func);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_heat_ref(
          ansatz_space_ref, heat_func);
      const Eigen::VectorXd fvec = delta_t * diffusion_rate_u *
                                   disc_sol.get_mass_matrix() *
                                   disc_sol_heat.get_global_fun();
      const Eigen::VectorXd fvec_ref = delta_t * diffusion_rate_u *
                                       disc_sol_ref.get_mass_matrix() *
                                       disc_sol_heat_ref.get_global_fun();

      // The left hand side of the linear systems of equations for u
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_u;

      solver_u.compute(disc_sol.get_mass_matrix() +
                       theta * delta_t * diffusion_rate_u *
                           disc_sol.get_stiffness_matrix());
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_u_ref;

      solver_u_ref.compute(disc_sol_ref.get_mass_matrix() +
                           theta * delta_t * diffusion_rate_u *
                               disc_sol_ref.get_stiffness_matrix());
      const Eigen::SparseMatrix<double> rhsMat =
          disc_sol.get_mass_matrix() - (1 - theta) * delta_t *
                                           diffusion_rate_u *
                                           disc_sol.get_stiffness_matrix();
      const Eigen::SparseMatrix<double> rhsMat_ref =
          disc_sol_ref.get_mass_matrix() -
          (1 - theta) * delta_t * diffusion_rate_u *
              disc_sol_ref.get_stiffness_matrix();
      // Sampling
      double error_l2 = 0;
      double error_h1 = 0;

      // Solve the solutions of u at the end of the time step by step
      auto start = std::chrono::steady_clock::now();
      Eigen::Matrix<double, Eigen::Dynamic, 1> u = disc_sol_u.get_global_fun();
      Eigen::Matrix<double, Eigen::Dynamic, 1> u_ref =
          disc_sol_u_ref.get_global_fun();
      for (int i = 0; i <= 2000; ++i) {
        {
          Eigen::Matrix<double, Eigen::Dynamic, 1> prolongated_u = u;
          for (int k = refinement_level; k < MAX_LVL; ++k) {
            prolongated_u = Ps[k] * prolongated_u;
          }
          error_l2 +=
              sqrt((prolongated_u - u_ref).transpose() *
                   disc_sol_ref.get_mass_matrix() * (prolongated_u - u_ref));
          error_h1 += sqrt((prolongated_u - u_ref).transpose() *
                           (disc_sol_ref.get_mass_matrix() +
                            disc_sol_ref.get_stiffness_matrix()) *
                           (prolongated_u - u_ref));
        }
        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs_u =
            rhsMat * disc_sol_u.get_global_fun() + fvec;

        // update the u
        u = solver_u.solve(rhs_u);
        disc_sol_u.set_function(u);

        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs_u_ref =
            rhsMat_ref * disc_sol_u_ref.get_global_fun() + fvec_ref;

        // update the u
        u_ref = solver_u_ref.solve(rhs_u_ref);
        disc_sol_u_ref.set_function(u_ref);
        t = t + delta_t;
      }
      disc_sol_u_ref.plot("sol", 5);
      auto end = std::chrono::steady_clock::now();
      // Store the time difference between start and end
      auto diff = end - start;
      std::cout << std::left << std::setw(8) << refinement_level << std::left
                << std::setw(15) << error_l2 / 2001 << std::left
                << std::setw(15) << error_h1 / 2001 << std::left
                << std::setw(15) << std::chrono::duration<double>(diff).count()
                << std::endl;
    }
  }
  return 0;
}
