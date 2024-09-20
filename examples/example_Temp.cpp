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
  std::function<double(const Vector3d &)> u_func = [](const Vector3d &in) {
    // xyz
    return 1.0 - exp(-1.0 * (pow(in(0) - 1, 2) + pow(in(1), 2) + pow(in(2), 2)));
  };

  std::function<double(const Vector3d &)> uuu_func = [](const Vector3d &in) {
    // x^3y^3z^3
    return (1.0 - exp(-1.0 * (pow(in(0) - 1, 2) + pow(in(1), 2) + pow(in(2), 2))))*(1.0 - exp(-1.0 * (pow(in(0) - 1, 2) + pow(in(1), 2) + pow(in(2), 2))))*(1.0 - exp(-1.0 * (pow(in(0) - 1, 2) + pow(in(1), 2) + pow(in(2), 2))));
  };

  Geometry geometry("sphere.dat");
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {2}) {
    // Iterate over refinement levels
    for (auto refinement_level : {1,2,3,4,5}) {
      // Build ansatz space and preprare the mass and stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u(ansatz_space, u_func);
      disc_sol_u.compute_metric();

      std::function<double(const SurfacePoint &p)> uu_func =
          [&disc_sol_u](const SurfacePoint &p) {
            return disc_sol_u.evaluate(p)(0) * disc_sol_u.evaluate(p)(0);
          };
        
      DiscreteLinearForm<
          DirichletTrace<double, std::function<double(const SurfacePoint &p)>>,
          LaplaceBeltramiScalar>
          disc_uu_lf(ansatz_space);
      disc_uu_lf.get_linear_form().set_function(uu_func);
      disc_uu_lf.compute();
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_mass;
      solver_mass.compute(disc_sol_u.get_mass_matrix());
      Eigen::Matrix<double, Eigen::Dynamic, 1> uu =
          solver_mass.solve(disc_uu_lf.get_discrete_linear_form());
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_uu(ansatz_space, uu);

      DiscreteLinearForm<
          DirichletTrace<double, std::function<double(const SurfacePoint &p)>>,
          LaplaceBeltramiScalar>
          disc_uuu_lf(ansatz_space);
      // discrete linear form for reaction terms uv^2
      std::function<double(const SurfacePoint &p)> app_uuu_func =
          [&disc_sol_uu, &disc_sol_u](const SurfacePoint &p) {
            return disc_sol_uu.evaluate(p)(0) * disc_sol_u.evaluate(p)(0);
          };
      disc_uuu_lf.get_linear_form().set_function(app_uuu_func);
      disc_uuu_lf.compute();
      Eigen::Matrix<double, Eigen::Dynamic, 1> uuu = solver_mass.solve(disc_uuu_lf.get_discrete_linear_form());
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_uuu(ansatz_space, uuu);
      DiscreteFunction<LaplaceBeltramiScalar> ref_sol_uuu(ansatz_space, uuu_func);
        auto error = disc_sol_uuu - ref_sol_uuu;
        std::cout<<error.norm_l2()<<std::endl;
        
        std::function<double(const SurfacePoint &p)> uuu_func1 =
            [&disc_sol_u](const SurfacePoint &p) {
              return disc_sol_u.evaluate(p)(0) * disc_sol_u.evaluate(p)(0) * disc_sol_u.evaluate(p)(0);
            };
          
        DiscreteLinearForm<
            DirichletTrace<double, std::function<double(const SurfacePoint &p)>>,
            LaplaceBeltramiScalar>
            disc_uuu_lf1(ansatz_space);
        disc_uuu_lf1.get_linear_form().set_function(uuu_func1);
        disc_uuu_lf1.compute();
        Eigen::Matrix<double, Eigen::Dynamic, 1> uuu1 = solver_mass.solve(disc_uuu_lf1.get_discrete_linear_form());
        DiscreteFunction<LaplaceBeltramiScalar> disc_sol_uuu1(ansatz_space, uuu1);
        auto error1 = disc_sol_uuu1 - ref_sol_uuu;
        std::cout<<error1.norm_l2()<<std::endl;
        
      }
    }
  return 0;
}
