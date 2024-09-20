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
    return in(0) * in(0) * in(0) + in(1) * in(1) * in(1) +
           in(2) * in(2) * in(2);
  };

  Geometry geometry("sphere.dat");
  std::cout << "geometry loaded\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3}) {
    // Iterate over refinement levels
    std::cout << "Degree " << polynomial_degree << std::endl;
    int MAX_LVL = 6;
    std::vector<Eigen::SparseMatrix<double>> Ps;
    for (int lvl = 0; lvl <= MAX_LVL - 1; ++lvl) {
      // Build ansatz space
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(geometry, lvl,
                                                      polynomial_degree);
      ansatz_space.compute_prolongation();
      Ps.push_back(ansatz_space.get_prolongation_matrix());
    }
    std::cout << std::left << std::setw(8) << "Level" << std::left
              << std::setw(15) << "MSE" << std::left << std::setw(15)
              << "Time/sec" << std::endl;
    for (auto refinement_level : {0, 1, 2, 3, 5, 6}) {
      // Build ansatz space and preprare the mass and stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);
      AnsatzSpace<LaplaceBeltramiScalar> finest_ansatz_space(geometry, MAX_LVL,
                                                             polynomial_degree);

      DiscreteFunction<LaplaceBeltramiScalar> disc_u(ansatz_space, init_u);
      DiscreteFunction<LaplaceBeltramiScalar> disc_prolongated_u(
          finest_ansatz_space);

      Eigen::Matrix<double, Eigen::Dynamic, 1> prolongated_u =
          disc_u.get_global_fun();
      auto start = std::chrono::steady_clock::now();
      for (int k = refinement_level; k < MAX_LVL; ++k) {
        prolongated_u = Ps[k] * prolongated_u;
      }
      disc_prolongated_u.set_function(prolongated_u);

      double mse = 0;
      for (int i = 0; i < ansatz_space.get_number_of_patches(); ++i) {
        for (int j = 0; j < 1000; ++j) {
          Eigen::Vector2d pt;
          pt(0) = (double)rand() / (RAND_MAX);
          pt(1) = (double)rand() / (RAND_MAX);
          mse = mse + abs(disc_u.evaluateOnPatch(i, pt)[0] -
                          disc_prolongated_u.evaluateOnPatch(i, pt)[0]);
        }
      }
      mse = mse / (1000 * ansatz_space.get_number_of_patches());

      auto end = std::chrono::steady_clock::now();
      // Store the time difference between start and end
      auto diff = end - start;
      std::cout << std::left << std::setw(8) << refinement_level << std::left
                << std::setw(15) << mse << std::left << std::setw(15)
                << std::chrono::duration<double>(diff).count() << std::endl;
    }
  }
  return 0;
}
