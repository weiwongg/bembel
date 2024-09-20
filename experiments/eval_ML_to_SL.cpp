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

// FMCA

// Random domain

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

  /////////////////////////////////////////////////////////////////////////////
  // set parameters
  /////////////////////////////////////////////////////////////////////////////
  InputParser input(argc, argv);
  assert(input.cmdOptionExists("-bemlvl"));
  int refinement_level_BEM = std::stoi(input.getCmdOption("-bemlvl"));
  assert(input.cmdOptionExists("-bemdeg"));
  int polynomial_degree_BEM = std::stoi(input.getCmdOption("-bemdeg"));
  assert(input.cmdOptionExists("-interlvl"));
  int interface_lvl = std::stoi(input.getCmdOption("-interlvl"));
  assert(input.cmdOptionExists("-interdeg"));
  int interface_degree = std::stoi(input.getCmdOption("-interdeg"));
  assert(input.cmdOptionExists("-wavenum"));
  int wavenum = std::stoi(input.getCmdOption("-wavenum"));
  assert(input.cmdOptionExists("-sample"));
  std::string sample = input.getRequiredCmd("-sample");

  IO::Stopwatch sw;

  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry geometry;            // geometry to deform
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("brick")) {
    interface_geometry =
        Geometry("../geo/artificial_cuboid.bpd", 1.0, Eigen::Vector3d(0, 0, 0));
    sphere_radius = 8.;
  } else if (!setting.compare("torus")) {
    interface_geometry =
        Geometry("../geo/refined_cube.bpd", 4.0, Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
  } else if (!setting.compare("sphere")) {
    interface_geometry =
        Geometry("../geo/refined_cube.bpd", 4.0, Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
  } else {
    assert(!"Setting not known");
  }
  assert(input.cmdOptionExists("-inputdir"));
  std::string inputdir = input.getCmdOption("-inputdir");
  assert(input.cmdOptionExists("-refdir"));
  std::string refdir = input.getCmdOption("-refdir");
  std::cout << "Done." << std::endl;
  std::complex<double> wavenumber((double)(wavenum), 0.);

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
  assert((reference_points - Util::makeSphereGrid(sphere_radius, 100)).norm() < 1e-12);
  IO::bin2Mat(inputdir + std::string("interface_points.bin"),
              &interface_points);
  assert((interface_points - artificial_interface.get_gridpoints()).norm() <
         1e-12);

  /////////////////////////////////////////////////////////////////////////////
  // load data
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Load data..." << std::endl;
  std::vector<VectorXcd> int_mean(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> int_2mom(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> int_var(polynomial_degree_BEM + 1);
  std::vector<MatrixXcd> int_cor(polynomial_degree_BEM + 1);
  VectorXcd int_mean_ref;
  VectorXcd int_2mom_ref;
  MatrixXcd int_cor_ref;

  std::vector<VectorXcd> pot_mean(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_2mom(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_var(polynomial_degree_BEM + 1);
  std::vector<MatrixXcd> pot_cor(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> pot_mean_fast(polynomial_degree_BEM + 1);
  std::vector<MatrixXcd> pot_cor_fast(polynomial_degree_BEM + 1);
  VectorXcd pot_mean_ref;
  VectorXcd pot_2mom_ref;
  MatrixXcd pot_cor_ref;

  std::vector<VectorXcd> pot_vars(polynomial_degree_BEM + 1);
  std::vector<VectorXcd> int_vars(polynomial_degree_BEM + 1);
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    std::string identifier =
        std::to_string(refinement_level_BEM) + "_" + std::to_string(l) + "_" +
        std::to_string(interface_lvl) + "_" + std::to_string(interface_degree) +
        "_" + std::to_string(wavenum) + "_" + sample + "_" + setting;
    IO::bin2Mat(std::string(inputdir) + identifier + "_int_mean.bin",
                &int_mean[l]);
    IO::bin2Mat(std::string(inputdir) + identifier + "_int_2mom.bin",
                &int_2mom[l]);
    IO::bin2Mat(std::string(inputdir) + identifier + "_int_cor.bin",
                &int_cor[l]);

    IO::bin2Mat(std::string(inputdir) + identifier + "_pot_mean.bin",
                &pot_mean[l]);
    IO::bin2Mat(std::string(inputdir) + identifier + "_pot_2mom.bin",
                &pot_2mom[l]);
    IO::bin2Mat(std::string(inputdir) + identifier + "_pot_cor.bin",
                &pot_cor[l]);
    int_vars[l] = int_2mom[l] - int_mean[l].cwiseAbs2();
    pot_vars[l] = pot_2mom[l] - pot_mean[l].cwiseAbs2();
    if (l > 0) int_mean[l] += int_mean[l - 1];
    if (l > 0) int_2mom[l] += int_2mom[l - 1];
    if (l > 0) int_cor[l] += int_cor[l - 1];
    if (l > 0) pot_mean[l] += pot_mean[l - 1];
    if (l > 0) pot_2mom[l] += pot_2mom[l - 1];
    if (l > 0) pot_cor[l] += pot_cor[l - 1];
  }

  std::string identifier_ref =
      std::to_string(interface_lvl) + "_" + std::to_string(interface_degree) +
      "_" + std::to_string(wavenum) + "_reference_" + setting;
  IO::bin2Mat(refdir + identifier_ref + "_int_mean.bin", &int_mean_ref);
  IO::bin2Mat(refdir + identifier_ref + "_int_2mom.bin", &int_2mom_ref);
  IO::bin2Mat(refdir + identifier_ref + "_int_cor.bin", &int_cor_ref);

  IO::bin2Mat(refdir + identifier_ref + "_pot_mean.bin", &pot_mean_ref);
  IO::bin2Mat(refdir + identifier_ref + "_pot_2mom.bin", &pot_2mom_ref);
  IO::bin2Mat(refdir + identifier_ref + "_pot_cor.bin", &pot_cor_ref);

  /////////////////////////////////////////////////////////////////////////////
  // compute pivoted Cholesky of correlation of Cauchy data
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Pivoted Cholesky..." << std::endl;
  std::vector<MatrixXcd> int_cor_sumL(polynomial_degree_BEM + 1);
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    PivotedCholesky<std::complex<double>> pC(int_cor[l],

                                             __BEMBEL_PIVOTEDCHOLESKY_TOL__);
    int_cor_sumL[l] = pC.get_L();
  }
  PivotedCholesky<std::complex<double>> pC_ref(int_cor_ref,

                                               __BEMBEL_PIVOTEDCHOLESKY_TOL__);
  MatrixXcd int_cor_sumL_ref = pC_ref.get_L();
  // Fast evaluation via Artificial Interface
  std::cout << "Fast evaluation of mean..." << std::endl;
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {	  
    artificial_interface.setBoundaryDataWithInterpolation(
        int_mean[l].head(interface_points.rows()),
        int_mean[l].tail(interface_points.rows()));
    pot_mean_fast[l] = artificial_interface.evaluate(reference_points);
    //pot_mean_fast[l] = pot_mean[l];
  }
  // Fast evaluation of cor via Artificial Interface
  std::cout << "Fast evaluation of cor..." << std::endl;
  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    MatrixXcd pot(reference_points.rows(), int_cor_sumL[l].cols());
    /*
    for (auto i = 0; i < int_cor_sumL[l].cols(); ++i) {
      artificial_interface.setBoundaryDataWithInterpolation(
          int_cor_sumL[l].block(0, i, interface_points.rows(), 1),
          int_cor_sumL[l].block(interface_points.rows(), i,
                                interface_points.rows(), 1));
      pot.col(i) = artificial_interface.evaluate(reference_points);
    }
    pot_cor_fast[l] = pot * pot.adjoint();
    */
    pot_cor_fast[l] = pot_cor[l];
  }

  /////////////////////////////////////////////////////////////////////////////
  // show convergence of quadrature towards solution on highest level
  /////////////////////////////////////////////////////////////////////////////
  std::ofstream file_mean_int;
  std::ofstream file_cor_int;
  std::ofstream file_mean_points;
  std::ofstream file_mean_fast_points;
  std::ofstream file_cor_points;
  std::ofstream file_cor_fast_points;
  file_mean_int.open(inputdir + sample + "Toref_mean_interface_" +
                     std::to_string(polynomial_degree_BEM) + ".txt");
  file_cor_int.open(inputdir + sample + "Toref_cor_interface_" +
                    std::to_string(polynomial_degree_BEM) + ".txt");
  file_mean_points.open(inputdir + sample + "Toref_mean_points_" +
                        std::to_string(polynomial_degree_BEM) + ".txt");
  file_mean_fast_points.open(inputdir + sample + "Toref_mean_fast_points_" +
                             std::to_string(polynomial_degree_BEM) + ".txt");
  file_cor_points.open(inputdir + sample + "Toref_cor_points_" +
                       std::to_string(polynomial_degree_BEM) + ".txt");
  file_cor_fast_points.open(inputdir + sample + "Toref_cor_fast_points_" +
                            std::to_string(polynomial_degree_BEM) + ".txt");
  std::cout << "Error..." << std::endl;

  for (auto l = 0; l <= polynomial_degree_BEM; ++l) {
    double int_mean_err = (int_mean[l] - int_mean_ref).cwiseAbs().maxCoeff();
    double int_cor_err = (int_cor[l] - int_cor_ref).cwiseAbs().maxCoeff();
    double pot_mean_err = (pot_mean[l] - pot_mean_ref).cwiseAbs().maxCoeff();
    double pot_cor_err = (pot_cor[l] - pot_cor_ref).cwiseAbs().maxCoeff();
    double pot_mean_fast_err =
        (pot_mean_fast[l] - pot_mean_ref).cwiseAbs().maxCoeff();
    double pot_cor_fast_err =
        (pot_cor_fast[l] - pot_cor_ref).cwiseAbs().maxCoeff();
    double int_var = (int_vars[l]).cwiseAbs().maxCoeff();
    double pot_var = (pot_vars[l]).cwiseAbs().maxCoeff();
    std::cout << "Level " << l << std::endl
              << "Err mean on points: " << pot_mean_err << std::endl
              << "Err mean on points using interface: " << pot_mean_fast_err
              << std::endl
              << "Err cor on points: " << pot_cor_err << std::endl
              << "Err cor on points using interface: " << pot_cor_fast_err
              << std::endl
              << "Variance on points: " << pot_var << std::endl;

    file_mean_int << l << "\t" << int_mean_err << std::endl;
    file_cor_int << l << "\t" << int_cor_err << std::endl;

    file_mean_points << l << "\t" << pot_mean_err << std::endl;
    file_mean_fast_points << l << "\t" << pot_mean_fast_err << std::endl;
    file_cor_points << l << "\t" << pot_cor_err << std::endl;
    file_cor_fast_points << l << "\t" << pot_cor_fast_err << std::endl;
  }
  file_mean_int.close();
  file_cor_int.close();
  file_mean_points.close();
  file_mean_fast_points.close();
  file_cor_points.close();
  return 0;
}
