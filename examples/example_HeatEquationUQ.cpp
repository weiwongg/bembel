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
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
std::function<double(const Eigen::Vector3d &)> init_u =
    [](const Eigen::Vector3d &in) { return in(2); };
std::function<double(const Eigen::Vector3d &)> heat_func =
    [](const Eigen::Vector3d &in) {
      return sin(M_PI * in(0)) * sin(M_PI * in(1)) * sin(M_PI * in(2));
    };

std::function<double(const Eigen::Vector3d &)> fun2 =
    [](const Eigen::Vector3d &point_in_space) { return point_in_space(0); };
////////////////////////////////////////////////////////////////////////////////

template <typename M>
M load_csv(const std::string &path) {
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

int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  const std::string geometry_name = "bunny5.bpd";
  const std::string divider(60, '-');
  const double scale = 10;
  const double amplifier = 1.0;
  const int polynomial_degree = 2;
  const int refinement_level = 3;
  const double delta_t = 0.01;
  const int qlevel = std::atoi(argv[1]);
  const int max_k = std::atoi(argv[2]);
  // check the correlation at 0:eval_steps:1
  const double eval_steps = 0.1;
  const int fq = (int)(eval_steps / delta_t);
  const int num_t = (int)(1.0 / eval_steps);
  const int num_steps = (int)(1.0 / delta_t);  // time steps

  // Heat Equation
  // u_t = r_u * \laplacian u + f
  // parameters
  const double theta = 0.5;  // Crank-Nicolson method
  const double diffusion_rate_u = 1.0;

  IO::Stopwatch sw;
  Geometry reference_geometry(geometry_name, scale);

  Eigen::VectorXd u0;
  Eigen::VectorXd u;
  Eigen::VectorXd f;
  // compute coefficients of init_u and heat source in the reference geometry.
  // the coefficients will keep tha same with respect to geometry deformation.
  AnsatzSpace<LaplaceBeltramiScalar> ref_ansatz_space(
      reference_geometry, refinement_level, polynomial_degree);
  DiscreteFunction<LaplaceBeltramiScalar> disc_ref_u(ref_ansatz_space, init_u);
  disc_ref_u.compute_metric();
  u0 = disc_ref_u.get_global_fun();
  u = u0;

  {
    DiscreteFunction<LaplaceBeltramiScalar> disc_ref_heat(ref_ansatz_space,
                                                          heat_func);
    f = disc_ref_heat.get_global_fun();
  }
  int num_basis = u.size();
  std::vector<Eigen::MatrixXd> DCors;
  DCors.resize(num_t);
  for (auto i = 0; i < num_t; ++i) {
    DCors[i].resize(num_basis, num_basis);
    DCors[i].setZero();
  }
  // Set up random domain deformation
  std::cout << divider << std::endl;
  sw.tic();
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
  // Eigen::MatrixXd Q = load_csv<Eigen::MatrixXd>(geo + "/" + method +
  // "/Q.csv"); Eigen::VectorXd W = load_csv<Eigen::MatrixXd>(geo + "/" + method
  // + "/W.csv");
  Eigen::MatrixXd Q;
  Eigen::VectorXd W;
  // use this for the Monte Carlo method
  // std::srand((unsigned int) time(0));
  gamma2MC(def.get_gamma(), &Q, &W, qlevel);
  // use this for the Halton sequence
  // gamma2HaltonQMC(def.get_gamma(), &Q, &W, qlevel);
  // use this for the sparse quadrature
  // gamma2SparseQuadrature(def.get_gamma(), &Q, &W, qlevel);
  const int samples = Q.cols();
  std::cout << "number of points: " << Q.cols() << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  int batch_size = 10;
  Bembel::UQ::OnlineCor cor(num_basis, num_t, max_k, 1e-15);
  Eigen::IOFormat CommaInitFmt(FullPrecision, 0, ", ", "\n");
  Eigen::MatrixXd u_joined(num_t * num_basis, batch_size);
  Eigen::VectorXd Wb(batch_size);
  // intalize progress bar
  IO::progressbar bar(samples);
  for (auto smpl_it = 0; smpl_it < samples; ++smpl_it) {
    param = amplifier * def.get_SingularValues() * Q.col(smpl_it);
    Geometry geometry = def.get_geometryRealization(param);
    //////////////////////////////////////////////////////////////////////////
    // the following block is the spatial solver
    //////////////////////////////////////////////////////////////////////////
    {
      // Build ansatz space and preprare the mass and stiffness matrices
      AnsatzSpace<LaplaceBeltramiScalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol(ansatz_space);
      // Initial solution of u at t=0
      DiscreteFunction<LaplaceBeltramiScalar> disc_sol_u(ansatz_space, u0);
      disc_sol.compute_metric();
      // The left hand side and the right hand side of the linear systems of
      // equations for u
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver_u;
      const Eigen::SparseMatrix<double> &M = disc_sol.get_mass_matrix();
      const Eigen::SparseMatrix<double> &S = disc_sol.get_stiffness_matrix();
      solver_u.compute(M + theta * delta_t * diffusion_rate_u * S);
      const Eigen::SparseMatrix<double> rhsMat =
          (M - (1 - theta) * delta_t * diffusion_rate_u * S);
      const Eigen::VectorXd fvec = delta_t * diffusion_rate_u * M * f;
      u = u0;
      Eigen::VectorXd u1(num_basis);
      Eigen::VectorXd u2(num_basis);
      double t = 0;
      // Solve the solutions of u at different time
      for (int i = 0; i < num_t; ++i) {
        for (int j = 0; j < fq; ++j) {
          u = solver_u.solve(rhsMat * u + fvec);
          t += delta_t;
        }
        u_joined.col(smpl_it % batch_size).segment(i * num_basis, num_basis) =
            u;
        DCors[i] += W(smpl_it) * u * u.transpose();
      }
      Wb(smpl_it % batch_size) = W(smpl_it);
      if (smpl_it % batch_size == batch_size - 1 || smpl_it == samples - 1) {
        if (smpl_it % batch_size != batch_size - 1) {
          int remain_num = samples - batch_size * (samples / batch_size);
          cor.update(u_joined.leftCols(remain_num), Wb.head(remain_num));

        } else {
          cor.update(u_joined, Wb);
        }
      }
    }
    // update progress bar
    bar.update();
  }

  double error = 0;

  for (auto i = 0; i < num_t; ++i) {
    error += (cor.get_cor(i, i) - DCors[i]).norm() / DCors[i].norm();
  }
  std::cout << std::endl;
  std::cout << "average relative_Frobenius_error over blocks on the diagonal: "
            << error / num_t << std::endl;

  Eigen::MatrixXd LRcor = cor.get_LRcor();
  std::ofstream file("LRcor.csv");
  file << LRcor.format(CommaInitFmt);
  file.close();
  return 0;
}
