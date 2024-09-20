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
#include <fstream>

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

template<typename M>
M load_csv (const std::string & path) {
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
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  const std::string geo(argv[1]);
  const std::string geometry_name = geo + "5.bpd";
  const std::string divider(60, '-');
  const double scale = (double) std::atof(argv[2]);
  const int amplifier_id = std::atoi(argv[3]);
  const double amplifier = amplifier_id * 0.25 + 1.0;
  const int polynomial_degree = std::atoi(argv[4]);
  const int refinement_level = std::atoi(argv[5]);
  const double delta_t = (double) std::atof(argv[6]);
  const std::string method(argv[7]);
  int block_size = std::atoi(argv[8]);
  int block_id = std::atoi(argv[9]);

  // Heat Equation
  // u_t = r_u * \laplacian u + f
  // parameters
  const double theta = 0.5; // Crank-Nicolson method
  const double diffusion_rate_u = 1.0;
  const int num_steps = (int) (1.0 / delta_t); // time steps
  // check the correlation between t1 and t2
  const double t1 = 0.5;
  const double t2 = 1.0;
  const int idx1 = (int) (t1 / delta_t - 1);
  const int idx2 = (int) (t2 / delta_t - 1);

  IO::Stopwatch sw;
  Geometry reference_geometry(geometry_name, scale);

  Eigen::VectorXd u0;
  Eigen::VectorXd u;
  Eigen::VectorXd f;
  Eigen::MatrixXd Vu;
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
  Vu.resize(u.size(), u.size());
  Vu.setZero();

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
  Eigen::MatrixXd Q = load_csv<Eigen::MatrixXd>(geo + "/" + method + "/Q.csv" );
  Eigen::VectorXd W = load_csv<Eigen::MatrixXd>(geo + "/" + method + "/W.csv" );
  assert(dim == Q.rows());
  // use this for the Monte Carlo method
  // gamma2MC(def.get_gamma(), &Q, &W, qlevel);
  // use this for the Halton sequence
  // gamma2HaltonQMC(def.get_gamma(), &Q, &W, qlevel);
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
  int blocks = samples / block_size;
    int num_basis = u.size();
    std::ofstream file1(geo + "/" + method + "/" + std::to_string(amplifier_id) + "/sol0_block" +
                        std::to_string(block_id) + ".csv");
    std::ofstream file2(geo + "/" + method + "/" + std::to_string(amplifier_id) + "/sol1_block" +
                        std::to_string(block_id) + ".csv");
    for (auto smpl_it = block_size * block_id;
         smpl_it < block_size * (block_id + 1); ++smpl_it) {
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
        std::cout << "--" << smpl_it << std::endl;
        double t = 0;
        // Solve the solutions of u at different time
        for (int i = 0; i < num_steps; ++i) {
          u = solver_u.solve(rhsMat * u + fvec);
          t += delta_t;
          if (i == idx1) {
            u1 = u;
          }
          if (i == idx2) {
            u2 = u;
          }
        }
        file1 << u1.transpose().format(CommaInitFmt) << "\n";
        file2 << u2.transpose().format(CommaInitFmt) << "\n";
      }
      std::cout << "smpl_it: " << smpl_it << std::endl;
    }
    file1.close();
    file2.close();
  return 0;
}
