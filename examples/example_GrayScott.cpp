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

int main() {
  using namespace Bembel;
  using namespace Eigen;
  std::function<double(const Vector3d &)> init_u = [](const Vector3d &in) {
    // small perturbation
    return 0.6581 + 0.02 * (double) rand()/RAND_MAX - 0.01;
    //return 1.0;
  };

  std::function<double(const Vector3d &)> init_v = [](const Vector3d &in) {
    // small perturbation
    return  0.2279 + 0.02 * (double) rand()/RAND_MAX - 0.01;
  };

  std::function<double(const Vector3d &)> init_cf = [](const Vector3d &in) {
    return 1.0;
  };

  Geometry geometry("sphere5.bpd");
  std::cout << "geometry loaded\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {2}) {
    // Iterate over refinement levels
    for (auto refinement_level : {4}) {
      // Build ansatz space and preprare the mass and stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      std::cout << "ansatz space computed\n";
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol(ansatz_space);
      disc_sol.compute_metric();

      // Gray Scott Model
      // u_t = r_u * \laplacian u -uv^2 + f(1-u)
      // v_y = r_v * \laplacian v +uv^2 - (f+k)v

      // parameters
      double delta_t = 0.1;
      double diffusion_rate_u = 0.01;
      // if you want to see spots on the surface, please change the diffusion_rate_v to 0.0005 or less
      double diffusion_rate_v = 0.001;
      double f = 0.1;
      double k = 0.05;

      // Initial solution of u and v at t = 0, and discrete function of the
      // constant 1
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u(ansatz_space, init_u);
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_v(ansatz_space, init_v);
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_c(ansatz_space, init_cf);
      std::cout << "data computed\n";
      // The left hand side of the linear systems of equations for u and v
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_u;
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_v;
      solver_u.compute((1 + delta_t * f) * disc_sol.get_mass_matrix() +
                       delta_t * diffusion_rate_u *
                           disc_sol.get_stiffness_matrix());
      solver_v.compute((1 + (f + k) * delta_t) * disc_sol.get_mass_matrix() +
                       delta_t * diffusion_rate_v *
                           disc_sol.get_stiffness_matrix());
      std::cout << "matrices factorized\n";
      // Prepare the discrete linear form for the right hand side of the linear
      // systems of equations
      DiscreteLinearForm<
          DirichletTrace<double, std::function<double(const SurfacePoint &p)>>,
          LaplaceBeltramiScalar>
          disc_reaction_lf(ansatz_space);
      // discrete linear form for reaction terms uv^2
      std::function<double(const SurfacePoint &p)> reaction_func =
          [&disc_sol_u, &disc_sol_v](const SurfacePoint &p) {
            return disc_sol_u.evaluate(p)(0) * disc_sol_v.evaluate(p)(0) *
                   disc_sol_v.evaluate(p)(0);
          };
      disc_reaction_lf.get_linear_form().set_function(reaction_func);
      std::cout << "sampling started" << std::endl;
      // Solve the solutions of u and v at the end of the time step by step
      for (int i = 0; i < 20000; ++i) {
        if (i % 100 == 0) {
          // save solutions of the u and the v per 10 iterations
          //disc_sol_u.plot("solutions/sol_u_" + std::to_string(i), 6);
          disc_sol_v.plot("solutions/sol_v_" + std::to_string(i), 6);
        }

        disc_reaction_lf.compute();
        // The right hand side of the linear systems of equations for u and v at
        // the time t
        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs_u =
            disc_sol.get_mass_matrix() *
                (disc_sol_c * (delta_t * f) + disc_sol_u).get_global_fun() -
            disc_reaction_lf.get_discrete_linear_form() * delta_t;

        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs_v =
            disc_sol.get_mass_matrix() * (disc_sol_v).get_global_fun() +
            disc_reaction_lf.get_discrete_linear_form() * delta_t;

        // update the u and the v
        Eigen::Matrix<double, Eigen::Dynamic, 1> u = solver_u.solve(rhs_u);
        Eigen::Matrix<double, Eigen::Dynamic, 1> v = solver_v.solve(rhs_v);
        disc_sol_u.set_function(u);
        disc_sol_v.set_function(v);
          
      }
    }
  }
  return 0;
}
