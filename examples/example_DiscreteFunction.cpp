
#include <Bembel/AnsatzSpace>
#include <Bembel/DiscreteFunction>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>
#include <Eigen/Dense>
#include <iostream>

int main() {
  using namespace Bembel;
  using namespace Eigen;
  IO::Stopwatch sw;
  std::function<double(const Vector3d &)> fun = [](const Vector3d &in) {
    return in(0);
  };
  std::function<double(const Vector3d &)> squared_fun = [](const Vector3d &in) {
    return in(0) * in(0);
  };
  Geometry geometry("sphere.dat");
  std::cout << "\n============================================================="
               "==========\n";
  // Iterate over polynomial degree.
  for (auto polynomial_degree : {1, 2, 3, 4}) {
    // Iterate over refinement levels
    for (auto refinement_level : {0, 1, 2, 3, 4}) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\n";
      // Build ansatz space
      AnsatzSpace<MassMatrixScalarCont> ansatz_space(geometry, refinement_level,
                                                     polynomial_degree);
      // Set up and compute discrete function
      DiscreteFunction<MassMatrixScalarCont> disc_fun(ansatz_space, fun);

      std::cout << "L2-norm of the original function: " << disc_fun.norm_l2()
                << " \n";
      std::cout << "H1-norm of the original function: " << disc_fun.norm_h1()
                << " \n";

#if 1
      // Visualization of the error on the surface
      disc_fun.plot("plot" + std::to_string(polynomial_degree) + "_" +
                    std::to_string(refinement_level) + ".vtp");
#endif
    }
  }
  // The VTKwriter sets up initial geomety information.
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
