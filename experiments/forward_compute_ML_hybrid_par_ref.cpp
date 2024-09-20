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

// other helpers
#include <FMCA/src/util/HaltonSet.h>

#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

// Parallel
#include <mpi.h>
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
  std::string sample = input.getRequiredCmd("-sample");
  IO::Stopwatch sw;
  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 1.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
  std::string suffix = "20.bpd";
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("david")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
    geo_scale = 2.0;
    shift = Eigen::Vector3d(0, 0, -0.3);
    suffix = "10.bpd";
  } else if (!setting.compare("torus")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    geo_scale = .4;
  } else if (!setting.compare("sphere")) {
    interface_geometry = Geometry("../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
  } else {
    assert(!"Setting not known");
  }
  std::string outputdir("./");
  if (input.cmdOptionExists("-outputdir"))
    outputdir = input.getCmdOption("-outputdir");
  std::complex<double> wavenumber((double)(wavenum), 0.);
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

  Eigen::VectorXcd my_int_mean_dif =
      MatrixXcd::Zero(2 * interface_points.rows(), 1);
  Eigen::VectorXcd my_int_2mom_dif =
      MatrixXcd::Zero(2 * interface_points.rows(), 1);
  Eigen::MatrixXcd my_int_cor_dif =
      MatrixXcd::Zero(2 * interface_points.rows(), 2 * interface_points.rows());

  Eigen::VectorXcd my_pot_mean_dif =
      MatrixXcd::Zero(reference_points.rows(), 1);
  Eigen::VectorXcd my_pot_2mom_dif =
      MatrixXcd::Zero(reference_points.rows(), 1);
  Eigen::MatrixXcd my_pot_cor_dif =
      MatrixXcd::Zero(reference_points.rows(), reference_points.rows());

  /////////////////////////////////////////////////////////////////////////////
  // Quadrature method
  /////////////////////////////////////////////////////////////////////////////
  Eigen::VectorXi samples_vec(7);
  samples_vec << 98470,47260,5969,2137,1029,389,128;
  //samples_vec << 155500,10550,2821,1491,368,128;
  int num_samples = samples_vec(polynomial_degree_BEM);
  /////////////////////////////////////////////////////////////////////////////
  // OPEN MPI and OPENMP
  /////////////////////////////////////////////////////////////////////////////
  int my_ID, num_procs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  int my_n, my_first_i, my_last_i;
  my_n = num_samples / num_procs;
  my_first_i = my_ID * my_n;
  my_last_i = my_first_i + my_n;
  if (my_ID == num_procs - 1) my_last_i = num_samples;
  double weight = 1.0 / num_samples;
  if (!sample.compare("mc")) {
    srand((unsigned int) (2*polynomial_degree_BEM));
  } else if (!sample.compare("mc1")) {
    srand((unsigned int) (2*polynomial_degree_BEM + 1));
  }
  Eigen::VectorXd param(def.get_parameterDimension());
  auto HS = FMCA::HaltonSet<100>(def.get_parameterDimension());
  for (int i = 0; i < my_first_i; ++i) {
    HS.next();
    param = Eigen::MatrixXd::Random(def.get_parameterDimension(), 1);
  }
  IO::progressbar bar(my_last_i - my_first_i);

  int max_size = my_last_i - my_first_i;
  Eigen::MatrixXcd my_int_data =
      MatrixXcd::Zero(2*interface_points.rows(),max_size);
  Eigen::MatrixXcd my_int_data_ =
      MatrixXcd::Zero(2*interface_points.rows(),max_size);
 
  for (int i = my_first_i; i < my_last_i; ++i) {
    // sw.tic();
    if ((!sample.compare("mc")) || (!sample.compare("mc1")))
      param = Eigen::MatrixXd::Random(def.get_parameterDimension(), 1);
    else if (!sample.compare("qmc")) {
      for (auto j = 0; j < HS.HaltonVector().size(); ++j) {
        param(j) = 2.0 * HS.HaltonVector()[j] - 1.0;
      }
      HS.next();
    } else {
      assert(!"Sampling method not known");
    }
    Geometry deformed_geometry = def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);
    if (polynomial_degree_BEM == 0) {
      std::vector<MatrixXcd> pot =
          integrand(deformed_geometry, gridpoints, refinement_level_BEM,
                    polynomial_degree_BEM, wavenumber);
      VectorXcd int_x(2 * interface_points.rows());
      int_x << pot[0].block(0, 0, interface_points.rows(), 1),
          artificial_interface.potentialGradientToNeumann(
              pot[1].block(0, 0, interface_points.rows(), 3));

      my_int_mean_dif += weight * int_x;
      my_int_2mom_dif += weight * int_x.cwiseAbs2();
      //my_int_cor_dif += weight * int_x * int_x.adjoint();
      my_int_data.col(i-my_first_i) = int_x;

      VectorXcd pot_x =
          pot[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
      my_pot_mean_dif += weight * pot_x;
      my_pot_2mom_dif += weight * pot_x.cwiseAbs2();
      my_pot_cor_dif += weight * pot_x * pot_x.adjoint();

    } else {
      std::vector<MatrixXcd> pot =
          integrand(deformed_geometry, gridpoints, refinement_level_BEM,
                    polynomial_degree_BEM, wavenumber);
      std::vector<MatrixXcd> pot_ =
          integrand(deformed_geometry, gridpoints, refinement_level_BEM,
                    polynomial_degree_BEM - 1, wavenumber);

      VectorXcd int_x(2 * interface_points.rows());
      VectorXcd int_x_(2 * interface_points.rows());
      int_x << pot[0].block(0, 0, interface_points.rows(), 1),
          artificial_interface.potentialGradientToNeumann(
              pot[1].block(0, 0, interface_points.rows(), 3));
      int_x_ << pot_[0].block(0, 0, interface_points.rows(), 1),
          artificial_interface.potentialGradientToNeumann(
              pot_[1].block(0, 0, interface_points.rows(), 3));

      my_int_mean_dif += weight * (int_x - int_x_);
      my_int_2mom_dif += weight * (int_x - int_x_).cwiseAbs2();
      //my_int_cor_dif +=
      //     weight * (int_x * int_x.adjoint() - int_x_ * int_x_.adjoint());
      my_int_data.col(i-my_first_i) = int_x;
      my_int_data_.col(i-my_first_i) = int_x_;

      VectorXcd pot_x =
          pot[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
      VectorXcd pot_x_ =
          pot_[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
      my_pot_mean_dif += weight * (pot_x - pot_x_);
      my_pot_2mom_dif += weight * (pot_x - pot_x_).cwiseAbs2();
      my_pot_cor_dif +=
          weight * (pot_x * pot_x.adjoint() - pot_x_ * pot_x_.adjoint());
    }
    if (!my_ID) bar.update();
    // std::cout << i << " time " << std::setprecision(4) << sw.toc() << " s" <<
    // std::endl;
  }
  // Correlation computing
  sw.tic();
  if (polynomial_degree_BEM == 0)
    my_int_cor_dif = weight * my_int_data * my_int_data.adjoint();
  else
    my_int_cor_dif = weight * (my_int_data * my_int_data.adjoint() -
                               my_int_data_ * my_int_data_.adjoint());
  std::cout << std::endl;
  std::cout << "correlation computing time " << std::setprecision(4) << sw.toc()
            << " s" << std::endl;

  ///////////////////////////////////////////////////////////////////////////
  // reduce quantities of interest to root process
  ///////////////////////////////////////////////////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  if (!my_ID) std::cout << "CONGRATULATION ALL DONE!!!" << std::endl;
  Eigen::VectorXcd int_mean_dif =
      MatrixXcd::Zero(my_int_mean_dif.rows(), my_int_mean_dif.cols());
  Eigen::VectorXcd int_2mom_dif =
      MatrixXcd::Zero(my_int_2mom_dif.rows(), my_int_2mom_dif.cols());
  Eigen::MatrixXcd int_cor_dif =
      MatrixXcd::Zero(my_int_cor_dif.rows(), my_int_cor_dif.cols());
  Eigen::VectorXcd pot_mean_dif =
      MatrixXcd::Zero(my_pot_mean_dif.rows(), my_pot_mean_dif.cols());
  Eigen::VectorXcd pot_2mom_dif =
      MatrixXcd::Zero(my_pot_2mom_dif.rows(), my_pot_2mom_dif.cols());
  Eigen::MatrixXcd pot_cor_dif =
      MatrixXcd::Zero(my_pot_cor_dif.rows(), my_pot_cor_dif.cols());

  MPI_Reduce(my_int_mean_dif.data(), int_mean_dif.data(), int_mean_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_int_2mom_dif.data(), int_2mom_dif.data(), int_2mom_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_int_cor_dif.data(), int_cor_dif.data(), int_cor_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(my_pot_mean_dif.data(), pot_mean_dif.data(), pot_mean_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_pot_2mom_dif.data(), pot_2mom_dif.data(), pot_2mom_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_pot_cor_dif.data(), pot_cor_dif.data(), pot_cor_dif.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

  ///////////////////////////////////////////////////////////////////////////
  // write quantities of interest to files
  ///////////////////////////////////////////////////////////////////////////
  if (!my_ID) {
    std::string identifier =
        std::to_string(refinement_level_BEM) + "_" +
        std::to_string(polynomial_degree_BEM) + "_" +
        std::to_string(interface_lvl) + "_" + std::to_string(interface_degree) +
        "_" + std::to_string(wavenum) + "_" + sample + "_" + setting;
    // quantities of interest on interface
    IO::print2bin(outputdir + identifier + std::string("_int_mean.bin"),
                  int_mean_dif);
    IO::print2bin(outputdir + identifier + std::string("_int_2mom.bin"),
                  int_2mom_dif);
    IO::print2bin(outputdir + identifier + std::string("_int_cor.bin"),
                  int_cor_dif);
    IO::print2bin(outputdir + identifier + std::string("_pot_mean.bin"),
                  pot_mean_dif);
    IO::print2bin(outputdir + identifier + std::string("_pot_2mom.bin"),
                  pot_2mom_dif);
    IO::print2bin(outputdir + identifier + std::string("_pot_cor.bin"),
                  pot_cor_dif);
    IO::print2bin(outputdir + std::string("reference_points.bin"),
                  reference_points);
    IO::print2bin(outputdir + std::string("interface_points.bin"),
                  interface_points);
  }
  MPI_Finalize();
  return 0;
}

