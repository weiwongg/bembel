#define _USE_MATH_DEFINES

#include <Bembel/AnsatzSpace>
#include <Bembel/DiscreteFunction>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Bembel/LocalOperator>
#include <Bembel/UQ>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <cmath>
#include <iostream>

template <typename M>
M load_csv(const std::string& path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

int main() {
  using namespace Bembel;
  using namespace Eigen;
  Geometry geometry("sphere5.bpd", 1);
  std::cout << "geometry loaded\n";
  // Iterate over polynomial degree.
  int polynomial_degree = 2;
  int refinement_level = 4;
  // Build ansatz space, the mass, and the stiffness matrices
  AnsatzSpace<LaplaceBeltramiScalar> ansatz_space_ref(
      geometry, refinement_level, polynomial_degree);

  DiscreteFunction<LaplaceBeltramiScalar> disc_sol_ref(ansatz_space_ref);
  disc_sol_ref.compute_metric();
  Eigen::MatrixXd sols =
      load_csv<Eigen::MatrixXd>("./sphere/test/0/sol1_block0.csv");

  Eigen::VectorXd exp(sols.cols());
  exp.setZero();
  Eigen::VectorXd var(sols.cols());
  var.setZero();
  int samples = sols.rows();
  for (int i = 0; i < sols.rows(); ++i) {
    exp += sols.row(i) / samples;
    var += sols.row(i).cwiseProduct(sols.row(i)) / samples;
  }
  var = var - exp.cwiseProduct(exp);
  disc_sol_ref.set_function(exp);
  disc_sol_ref.plot("sphere_exp", 5, "exp");
  Eigen::VectorXd std = var.cwiseSqrt();
  disc_sol_ref.set_function(std);
  disc_sol_ref.plot("sphere_std", 5, "std");
  /*
  Bembel::UQ::GeometryDeformer def("bunny5.bpd", 10);
  // int dim = def.set_parameterDimension(24);
  Eigen::MatrixXd Q = load_csv<Eigen::MatrixXd>("../Experiment/bunny/mc/Q.csv");
  int dim = def.get_parameterDimension();
  Eigen::VectorXd param(dim);
  assert(dim == Q.rows());
  for (int i = 0; i < 100; ++i) {
    param = Q.col(i);
    geometry =
        def.get_geometryRealization(2 * def.get_SingularValues() * param);
    AnsatzSpace<LaplaceBeltramiScalar> ansatz_space_def(
        geometry, refinement_level, polynomial_degree);
    DiscreteFunction<LaplaceBeltramiScalar> disc_sol_def(ansatz_space_def);
    disc_sol_def.set_function(exp);
    disc_sol_def.plot("def_bunny_exp_" + std::to_string(i), 5, "exp");
  }
   */
  return 0;
}
