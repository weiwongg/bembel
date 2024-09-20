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
#include <Bembel/src/util/FormalSum.hpp>
#include <Bembel/src/util/GenericMatrix.hpp>
#include <Bembel/src/util/PivotedCholesky.hpp>

// Parallel
#include <omp.h>

// other helpers
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

#define __BEMBEL_PIVOTEDCHOLESKY_TOL__ 1e-6

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
  Bembel::IO::Stopwatch sw;

  /////////////////////////////////////////////////////////////////////////////
  // set parameters
  /////////////////////////////////////////////////////////////////////////////
  InputParser input(argc, argv);
  int refinement_level_BEM = 1;
  int polynomial_degree_BEM = 5;
  int interface_lvl = 0;
  int interface_degree = 8;
  int wavenum = Bembel::Constants::BEMBEL_WAVENUM;
  std::string inputdir("./");
  std::complex<double> wavenumber(wavenum, 0.);
  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry geometry;            // geometry to deform
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 1.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("david")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
    geo_scale = 2.0;
    shift = Eigen::Vector3d(0, 0, -0.3);
    interface_degree = 8;
  } else if (!setting.compare("torus")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
    geo_scale = 0.4;
  } else if (!setting.compare("sphere")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
  } else {
    assert(!"Setting not known");
  }

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

  /////////////////////////////////////////////////////////////////////////////
  // define gridpoints for evaluation of potential
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Load evaluation points..." << std::endl;
  MatrixXd interface_points;
  MatrixXd reference_points;
  IO::bin2Mat(inputdir + std::string("reference_points.bin"),
              &reference_points);
  assert((reference_points - Util::makeSphereGrid(sphere_radius, 100)).norm() <
         1e-12);
  IO::bin2Mat(inputdir + std::string("interface_points.bin"),
              &interface_points);
  assert((interface_points - artificial_interface.get_gridpoints()).norm() <
         1e-12);
  /////////////////////////////////////////////////////////////////////////////
  // load data
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Load data..." << std::endl;
  std::vector<VectorXcd> int_mean(polynomial_degree_BEM + 1);
  VectorXcd int_mean_ref;

  std::vector<VectorXcd> pot_mean(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_mean_fast(polynomial_degree_BEM + 1);
  VectorXcd pot_mean_ref;
  sw.tic();
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    std::string identifier =
        std::to_string(refinement_level_BEM) + "_" + std::to_string(l) + "_" +
        std::to_string(interface_lvl) + "_" + std::to_string(interface_degree) +
        "_" + std::to_string(wavenum) + "_" + setting;
    IO::bin2Mat(std::string(inputdir) + identifier + "_int_mean.bin",
                &int_mean[l]);
    IO::bin2Mat(std::string(inputdir) + identifier + "_pot_mean.bin",
                &pot_mean[l]);
  }
  std::string identifier_ref = "1_6_" + std::to_string(interface_lvl) + "_" +
                               std::to_string(interface_degree) + "_" +
                               std::to_string(wavenum) + "_" + setting;
  IO::bin2Mat(std::string(inputdir) + identifier_ref + "_int_mean.bin",
              &int_mean_ref);
  IO::bin2Mat(std::string(inputdir) + identifier_ref + "_pot_mean.bin",
              &pot_mean_ref);

  // Fast evaluation via Artificial Interface
  std::cout << "Fast evaluation of mean..." << std::endl;
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    artificial_interface.setBoundaryDataWithInterpolation(
        int_mean[l].head(interface_points.rows()),
        int_mean[l].tail(interface_points.rows()));
    pot_mean_fast[l] = artificial_interface.evaluate(reference_points);
  }

  /////////////////////////////////////////////////////////////////////////////
  // show convergence of quadrature towards solution on highest level
  /////////////////////////////////////////////////////////////////////////////
  std::ofstream file_mean_int;
  std::ofstream file_mean_points;
  std::ofstream file_mean_fast_points;

  file_mean_int.open(inputdir + "mean_interface_" +
                     std::to_string(refinement_level_BEM) + "_bem.txt");
  file_mean_points.open(inputdir + "mean_points_" +
                        std::to_string(refinement_level_BEM) + "_bem.txt");
  file_mean_fast_points.open(inputdir + "mean_fast_points_" +
                             std::to_string(refinement_level_BEM) + "_bem.txt");

  std::cout << "Error..." << std::endl;
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    double int_mean_err = (int_mean[l] - int_mean_ref).cwiseAbs().maxCoeff();
    double pot_mean_err = (pot_mean[l] - pot_mean_ref).cwiseAbs().maxCoeff();
    double pot_mean_fast_err =
        (pot_mean_fast[l] - pot_mean_ref).cwiseAbs().maxCoeff();
    std::cout << "Level " << l << std::endl
              << "Err mean on points: " << pot_mean_err << " "
              << pot_mean_fast_err << std::endl;
    file_mean_int << l << "\t" << int_mean_err << std::endl;
    file_mean_points << l << "\t" << pot_mean_err << std::endl;
    file_mean_fast_points << l << "\t" << pot_mean_fast_err << std::endl;
  }
  file_mean_int.close();
  file_mean_points.close();
  file_mean_fast_points.close();
  std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
  return 0;
}
