// C++
#include <fstream>
#include <iostream>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

// Bembel
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/LinearForm>
#include <Bembel/UQ>
#include <Bembel/src/util/FormalSum.hpp>
#include <Bembel/src/util/GenericMatrix.hpp>

// Random domain
#include "RandomDomain/Auxilliary.h"
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

// Parallel
#include <omp.h>

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// main program
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;

  /////////////////////////////////////////////////////////////////////////////
  // set parameters
  /////////////////////////////////////////////////////////////////////////////
  InputParser input(argc, argv);
  int refinement_level_BEM = 1;
  int polynomial_degree_BEM = 7;
  int interface_lvl = 0;
  int interface_degree = 8;
  IO::Stopwatch sw;
  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 1.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
  std::string suffix;
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("david")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    geo_scale = 2.0;
    shift = Eigen::Vector3d(0, 0, -0.3);
    suffix = "10.bpd";
    interface_degree = 8;
    polynomial_degree_BEM = 5;
  } else if (!setting.compare("torus")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    geo_scale = 0.4;
    suffix = "20.bpd";
    polynomial_degree_BEM = 6;
  } else if (!setting.compare("sphere")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    suffix = "20.bpd";
  } else {
    assert(!"Setting not known");
  }
  std::complex<double> wavenumber(Bembel::Constants::BEMBEL_WAVENUM, 0.);
  std::cout << "Done." << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // RandomGeometryDeformer
  /////////////////////////////////////////////////////////////////////////////
  sw.tic();
  std::cout << "Loading geometry deformer..." << std::endl;

  Bembel::UQ::GeometryDeformer def("../geo/" + setting + suffix, geo_scale,
                                   shift);
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s\t\t" << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // define artificial interface
  /////////////////////////////////////////////////////////////////////////////
  ArtificialInterface<
      HelmholtzSingleLayerOperator,
      HelmholtzSingleLayerPotential<HelmholtzSingleLayerOperator>,
      HelmholtzDoubleLayerOperator,
      HelmholtzDoubleLayerPotential<HelmholtzDoubleLayerOperator>>
      artificial_interface(interface_geometry, interface_lvl, interface_degree);
  artificial_interface.get_single_layer_potential()
      .get_potential()
      .set_wavenumber(wavenumber);
  artificial_interface.get_double_layer_potential()
      .get_potential()
      .set_wavenumber(wavenumber);
  std::cout << "Artificial interface initialized" << std::endl;
  MatrixXd interface_points = artificial_interface.get_gridpoints();
  MatrixXd reference_points = Util::makeSphereGrid(sphere_radius, 100);
  MatrixXd gridpoints(interface_points.rows() + reference_points.rows(), 3);
  gridpoints << interface_points, reference_points;

  std::vector<VectorXcd> int_mean_dif(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> int_2mom_dif(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> int_var(polynomial_degree_BEM + 1);

  std::vector<VectorXcd> pot_mean_dif(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_2mom_dif(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_var(polynomial_degree_BEM + 1);

  for (int i = 0; i < polynomial_degree_BEM + 1; ++i) {
    pot_mean_dif[i] = MatrixXcd::Zero(reference_points.rows(), 1);
    pot_2mom_dif[i] = MatrixXcd::Zero(reference_points.rows(), 1);
    int_mean_dif[i] = MatrixXcd::Zero(2 * interface_points.rows(), 1);
    int_2mom_dif[i] = MatrixXcd::Zero(2 * interface_points.rows(), 1);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Quadrature method
  /////////////////////////////////////////////////////////////////////////////
  int maximal_quadrature_level = 6;
  int num_samples = 1 << maximal_quadrature_level;

  double weight = 1.0 / num_samples;
  srand((unsigned int)131415);
  Eigen::VectorXd param(def.get_parameterDimension());
  IO::progressbar bar(num_samples);
  Eigen::VectorXd times = Eigen::MatrixXd::Zero(polynomial_degree_BEM + 1, 1);
  Eigen::VectorXd vars = Eigen::MatrixXd::Zero(polynomial_degree_BEM + 1, 1);
  for (int i = 0; i < num_samples; ++i) {
    param = Eigen::MatrixXd::Random(def.get_parameterDimension(), 1);
    Geometry deformed_geometry =
        def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);
    VectorXcd int_x_pre = MatrixXcd::Zero(2 * interface_points.rows(), 1);
    VectorXcd pot_x_pre = MatrixXcd::Zero(reference_points.rows(), 1);
    for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
      sw.tic();
      std::vector<MatrixXcd> pot = integrand(
          deformed_geometry, gridpoints, refinement_level_BEM, deg, wavenumber);
      VectorXcd int_x(2 * interface_points.rows());
      int_x << pot[0].block(0, 0, interface_points.rows(), 1),
          artificial_interface.potentialGradientToNeumann(
              pot[1].block(0, 0, interface_points.rows(), 3));
      int_mean_dif[deg] += weight * (int_x - int_x_pre);
      int_2mom_dif[deg] += weight * (int_x - int_x_pre).cwiseAbs2();
      int_x_pre = int_x;
      VectorXcd pot_x =
          pot[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
      pot_mean_dif[deg] += weight * (pot_x - pot_x_pre);
      pot_2mom_dif[deg] += weight * (pot_x - pot_x_pre).cwiseAbs2();
      pot_x_pre = pot_x;
      times(deg) += sw.toc() / num_samples;
    }
    bar.update();
  }
  for (int deg = polynomial_degree_BEM; deg > 0; --deg) {
    times(deg) = times(deg-1) + times(deg);
  }
  ///////////////////////////////////////////////////////////////////////////
  // print out variance
  ///////////////////////////////////////////////////////////////////////////
  std::cout << std::endl;
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    std::cout << "polynomial_degree: " << deg << " "
              << (int_2mom_dif[deg] - int_mean_dif[deg].cwiseAbs2())
                     .cwiseAbs()
                     .maxCoeff()
              << std::endl;
  }
  std::cout << "========" << std::endl;
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    std::cout << "polynomial_degree: " << deg << " "
              << (pot_2mom_dif[deg] - pot_mean_dif[deg].cwiseAbs2())
                     .cwiseAbs()
                     .maxCoeff()
              << std::endl;
  }

  std::cout << "========" << std::endl;
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    std::cout << "time: " << deg << " " << std::setprecision(4) << times(deg)
              << std::endl;
  }

  std::ofstream file;
  file.open("time_var.txt");
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    file << deg << "\t" << times(deg) << "\t"
         << (int_2mom_dif[deg] - int_mean_dif[deg].cwiseAbs2())
                .cwiseAbs()
                .maxCoeff()
         << std::endl;
    vars(deg) = (int_2mom_dif[deg] - int_mean_dif[deg].cwiseAbs2())
                    .cwiseAbs()
                    .maxCoeff();
  }
  file.close();
  Eigen::VectorXd num_ref = Eigen::MatrixXd::Zero(polynomial_degree_BEM + 1, 1);
  double base = 128;
  num_ref(polynomial_degree_BEM) = base;
  for (int deg = polynomial_degree_BEM - 1; deg >= 0; --deg) {
    num_ref(deg) =
        sqrt((times(deg + 1) * vars(deg)) / (times(deg) * vars(deg + 1))) *
        num_ref(deg + 1);
  }

  Eigen::VectorXd nums = Eigen::MatrixXd::Zero(polynomial_degree_BEM, 1);
  nums(polynomial_degree_BEM - 1) = base;
  for (int deg = polynomial_degree_BEM - 2; deg >= 0; --deg) {
    nums(deg) =
        sqrt((times(deg + 1) * vars(deg)) / (times(deg) * vars(deg + 1))) *
        nums(deg + 1);
  }

  std::cout << "======Number of samples=======" << std::endl;
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    std::cout << num_ref(deg) << "\t";
  }
  std::cout << std::endl;
  for (int deg = 0; deg < polynomial_degree_BEM; ++deg) {
    std::cout << nums(deg) << "\t";
  }
  std::cout << std::endl;
  std::cout << "======Times in munite=======" << std::endl;
  double total_time_ref = 0;
  for (int deg = 0; deg < polynomial_degree_BEM + 1; ++deg) {
    std::cout << num_ref(deg) * times(deg) / 60.0 << "\t";
    total_time_ref += num_ref(deg) * times(deg) / 60.0;
  }
  std::cout << std::endl;
  std::cout << total_time_ref << std::endl;

  double total_time = 0;
  for (int deg = 0; deg < polynomial_degree_BEM; ++deg) {
    std::cout << nums(deg) * times(deg) / 60.0 << "\t";
    total_time += nums(deg) * times(deg) / 60.0;
  }
  std::cout << std::endl;
  std::cout << total_time << std::endl;

  return 0;
}
