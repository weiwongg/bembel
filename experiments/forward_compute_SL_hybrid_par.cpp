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
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

// #include "HaltonSet.hpp"
#include <FMCA/src/util/HaltonSet.h>

// Parallel
#include <mpi.h>
#include <omp.h>
// #define NUM_THREADS 1

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
  assert(input.cmdOptionExists("-seed"));
  int seed = std::stoi(input.getCmdOption("-seed"));
  IO::Stopwatch sw;
  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 1.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("brick")) {
    interface_geometry =
        Geometry("../geo/artificial_cuboid.bpd", 1.0, Eigen::Vector3d(0, 0, 0));
    sphere_radius = 8.;
  } else if (!setting.compare("torus")) {
    interface_geometry =
        Geometry("../geo/refined_cube.bpd", 4.0, Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    geo_scale = 0.5;
  } else if (!setting.compare("sphere")) {
    interface_geometry =
        Geometry("../geo/refined_cube.bpd", 4.0, Eigen::Vector3d(-0.5, -0.5, -0.5));
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
  Bembel::UQ::GeometryDeformer def("../geo/" + setting + "20.bpd", geo_scale, shift);
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

  Eigen::VectorXcd my_int_mean =
      MatrixXcd::Zero(2 * interface_points.rows(), 1);
  Eigen::VectorXcd my_int_2mom =
      MatrixXcd::Zero(2 * interface_points.rows(), 1);
  Eigen::MatrixXcd my_int_cor =
      MatrixXcd::Zero(2 * interface_points.rows(), 2 * interface_points.rows());

  Eigen::VectorXcd my_pot_mean = MatrixXcd::Zero(reference_points.rows(), 1);
  Eigen::VectorXcd my_pot_2mom = MatrixXcd::Zero(reference_points.rows(), 1);
  Eigen::MatrixXcd my_pot_cor =
      MatrixXcd::Zero(reference_points.rows(), reference_points.rows());

  /////////////////////////////////////////////////////////////////////////////
  // Quadrature method
  /////////////////////////////////////////////////////////////////////////////
  int maximal_quadrature_level = 15;
  int num_samples = 1 << maximal_quadrature_level;
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
  srand((unsigned int)(seed));
  Eigen::VectorXd param(def.get_parameterDimension());
  auto HS = FMCA::HaltonSet<131415>(def.get_parameterDimension());
  for (int i = 0; i < my_first_i; ++i) {
    HS.next();
    param = Eigen::MatrixXd::Random(def.get_parameterDimension(), 1);
  }
  IO::progressbar bar(my_last_i-my_first_i);
  for (int i = my_first_i; i < my_last_i; ++i) {
    //sw.tic();
    if (!sample.compare("mc"))
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
    std::vector<MatrixXcd> pot =
        integrand(deformed_geometry, gridpoints, refinement_level_BEM,
                  polynomial_degree_BEM, wavenumber);
    VectorXcd int_x(2 * interface_points.rows());
    int_x << pot[0].block(0, 0, interface_points.rows(), 1),
        artificial_interface.potentialGradientToNeumann(
            pot[1].block(0, 0, interface_points.rows(), 3));

    my_int_mean += weight * int_x;
    my_int_2mom += weight * int_x.cwiseAbs2();
    my_int_cor += weight * int_x * int_x.adjoint();

    VectorXcd pot_x =
        pot[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
    my_pot_mean += weight * pot_x;
    my_pot_2mom += weight * pot_x.cwiseAbs2();
    my_pot_cor += weight * pot_x * pot_x.adjoint();
    if (!my_ID) bar.update();
    //std::cout << i << " time " << std::setprecision(4) << sw.toc() << " s" << std::endl;
  }
  ///////////////////////////////////////////////////////////////////////////
  // reduce quantities of interest to root process
  ///////////////////////////////////////////////////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  if (!my_ID) std::cout << "CONGRATULATION ALL DONE!!!" << std::endl;
  Eigen::VectorXcd int_mean =
      MatrixXcd::Zero(my_int_mean.rows(), my_int_mean.cols());
  Eigen::VectorXcd int_2mom =
      MatrixXcd::Zero(my_int_2mom.rows(), my_int_2mom.cols());
  Eigen::MatrixXcd int_cor =
      MatrixXcd::Zero(my_int_cor.rows(), my_int_cor.cols());
  Eigen::VectorXcd pot_mean =
      MatrixXcd::Zero(my_pot_mean.rows(), my_pot_mean.cols());
  Eigen::VectorXcd pot_2mom =
      MatrixXcd::Zero(my_pot_2mom.rows(), my_pot_2mom.cols());
  Eigen::MatrixXcd pot_cor =
      MatrixXcd::Zero(my_pot_cor.rows(), my_pot_cor.cols());

  MPI_Reduce(my_int_mean.data(), int_mean.data(), int_mean.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_int_2mom.data(), int_2mom.data(), int_2mom.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_int_cor.data(), int_cor.data(), int_cor.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(my_pot_mean.data(), pot_mean.data(), pot_mean.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_pot_2mom.data(), pot_2mom.data(), pot_2mom.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(my_pot_cor.data(), pot_cor.data(), pot_cor.size(),
             MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

  ///////////////////////////////////////////////////////////////////////////
  // write quantities of interest to files
  ///////////////////////////////////////////////////////////////////////////
  if (!my_ID) {
    std::string identifier = std::to_string(interface_lvl) + "_" +
                             std::to_string(interface_degree) + "_" +
                             std::to_string(wavenum) + "_reference_" + setting;
    // quantities of interest on interface
    IO::print2bin(outputdir + identifier + std::string("_int_mean.bin"),
                  int_mean);
    IO::print2bin(outputdir + identifier + std::string("_int_2mom.bin"),
                  int_2mom);
    IO::print2bin(outputdir + identifier + std::string("_int_cor.bin"),
                  int_cor);
    IO::print2bin(outputdir + identifier + std::string("_pot_mean.bin"),
                  pot_mean);
    IO::print2bin(outputdir + identifier + std::string("_pot_2mom.bin"),
                  pot_2mom);
    IO::print2bin(outputdir + identifier + std::string("_pot_cor.bin"),
                  pot_cor);
    IO::print2bin(outputdir + std::string("reference_points.bin"),
                  reference_points);
    IO::print2bin(outputdir + std::string("interface_points.bin"),
                  interface_points);
  }
  MPI_Finalize();
  return 0;
}
