////////////////////////////////////////////////////////////////////////////////
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
#include <SparseQuadrature/gamma2HaltonQMC.hpp>
#include <SparseQuadrature/gamma2MC.hpp>
#include <SparseQuadrature/gamma2SparseQuadrature.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  const std::string geo(argv[1]);
  const std::string geometry_name = geo + "5.bpd";
  const std::string divider(60, '-');
  const double scale = std::atof(argv[2]);
  const std::string method(argv[3]);
  const int qlevel = std::atoi(argv[4]);

  IO::Stopwatch sw;
  sw.tic();
  Geometry reference_geometry(geometry_name, scale);

  Bembel::UQ::GeometryDeformer def(geometry_name, scale);
  // int dim = def.set_parameterDimension(24);
  int dim = def.get_parameterDimension();
  Eigen::VectorXd param(dim);
  std::cout << "parameter dimension: " << dim << "\nmaximum perturbation: "
            << def.get_maxPerturbation().maxCoeff()
            << "\ncmp time: " << sw.toc() << "s" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // perform sampling
  // the following generates the matrix of quadrature points
  // Q = [xi_1, xi_2, ...] and the vector with corresponding weights
  // W = [w_1, w_2, ...]
  Eigen::MatrixXd Q;
  Eigen::VectorXd W;
  // use this for the Monte Carlo method
  // std::srand((unsigned int) time(0));
  // gamma2MC(def.get_gamma(), &Q, &W, qlevel);
  // use this for the Halton sequence
  gamma2HaltonQMC(def.get_gamma(), &Q, &W, qlevel);
  // use this for the sparse quadrature
  // gamma2SparseQuadrature(def.get_gamma(), &Q, &W, qlevel);
  const int samples = Q.cols();
  std::cout << "number of quadrature points: " << Q.cols() << std::endl;
  // check if we can integrate at least a linear exactly
  std::cout << "This -->" << (Q * W).norm() << " should be almost zero\n";
  std::cout << "sum of weights: " << W.sum() << std::endl;
  std::cout << divider << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // Bembel::UQ::OnlineCor cor(num_steps, max_k, 1e-10);
  Eigen::IOFormat CommaInitFmt(FullPrecision, 0, ", ", "\n");
  std::ofstream fileQ("../Experiment/" + geo + "/" + method + "/Q.csv");
  std::ofstream fileW("../Experiment/" + geo + "/" + method + "/W.csv");
  fileQ << Q.format(CommaInitFmt);
  fileW << W.format(CommaInitFmt);
  fileQ.close();
  fileW.close();
  return 0;
}
