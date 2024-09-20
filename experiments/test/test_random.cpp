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

// FMCA
#include <FMCA/CovarianceKernel>
#include <FMCA/Samplets>

// Random domain
#include "RandomDomain/Auxilliary.h"

// other helpers
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

#include <FMCA/src/util/HaltonSet.h>

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
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("brick")) {
    interface_geometry = Geometry("../geo/artificial_cuboid.bpd");
    sphere_radius = 8.;
  } else {
    assert(!"Setting not known");
  }
  std::string outputdir("./");
  if (input.cmdOptionExists("-outputdir"))
    outputdir = input.getCmdOption("-outputdir");
  std::complex<double> wavenumber((double)(wavenum), 0.);
  std::cout << "Done." << std::endl;

  int num_samples = 1 << 4;
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
    srand((unsigned int) 0);
  } else if (!sample.compare("mc1")) {
    srand((unsigned int) 131415926);
  }
  Eigen::VectorXd param(3);
  auto HS = FMCA::HaltonSet<100>(3);
  for (int i = 0; i < my_first_i; ++i) {
    HS.next();
    param = Eigen::MatrixXd::Random(3, 1);
  }
  for (int i = my_first_i; i < my_last_i; ++i) {
    sw.tic();
    if ((!sample.compare("mc")) || (!sample.compare("mc1")))
      param = Eigen::MatrixXd::Random(3, 1);
    else if (!sample.compare("qmc")) {
      for (auto j = 0; j < HS.HaltonVector().size(); ++j) {
        param(j) = 2.0 * HS.HaltonVector()[j] - 1.0;
      }
      HS.next();
    } else {
      assert(!"Sampling method not known");
    }

    std::cout << "Rank: " << my_ID << std::endl;
    std::cout << param.transpose() <<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
