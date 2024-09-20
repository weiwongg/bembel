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
// #include <FMCA/src/util/HaltonSet.h>

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
  int polynomial_degree_BEM_max = 8;
  int interface_lvl = 0;
  int interface_degree = 8;
  int wavenum = Bembel::Constants::BEMBEL_WAVENUM;
  IO::Stopwatch sw;
  /////////////////////////////////////////////////////////////////////////////
  // load geometries
  /////////////////////////////////////////////////////////////////////////////
  Geometry geometry;            // geometry to deform
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 1.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
  std::string suffix;
  std::cout << "Loading geometries..." << std::endl;
  std::string setting = input.getRequiredCmd("-setting");
  if (!setting.compare("david")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
    geo_scale = 2.0;
    shift = Eigen::Vector3d(0, 0, -0.3);
    suffix = "10.bpd";
    interface_degree = 8;
    polynomial_degree_BEM_max = 6;
  } else if (!setting.compare("torus")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.;
    geo_scale = 0.4;
    suffix = "20.bpd";
    polynomial_degree_BEM_max = 7;
  } else if (!setting.compare("sphere")) {
    interface_geometry = Geometry("../../geo/refined_cube.bpd", 4.0,
                                  Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
    suffix = "20.bpd";
  } else {
    assert(!"Setting not known");
  }
  sw.tic();
  Bembel::UQ::GeometryDeformer def("../../geo/" + setting + suffix, geo_scale,
                                   shift);
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s" << std::endl;
  int dim = def.get_parameterDimension();
  std::cout << "UQ dimension = " << def.get_parameterDimension() << std::endl;
  srand((unsigned int)0);
  Eigen::VectorXd param = Eigen::MatrixXd::Random(dim, 1);
  geometry = def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);
  VTKSurfaceExport writer(geometry, 6);
  writer.writeToFile("geo.vtp");
  VTKSurfaceExport writer_interface(interface_geometry, 6);
  writer_interface.writeToFile("interface_geo.vtp");
  std::string outputdir("./");
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
  std::cout << "Artificial interface initialized" << std::endl;
  MatrixXd interface_points = artificial_interface.get_gridpoints();
  MatrixXd reference_points = Util::makeSphereGrid(sphere_radius, 100);
  MatrixXd gridpoints(interface_points.rows() + reference_points.rows(), 3);
  gridpoints << interface_points, reference_points;

  Eigen::IOFormat CommaInitFmt(Eigen::FullPrecision, 0, ", ", "\n");
  std::ofstream file0("./plot/reference_data.csv");
  file0 << reference_points.format(CommaInitFmt);
  file0.close();
  std::ofstream file1("./plot/interface_points.csv");
  file1 << interface_points.format(CommaInitFmt);
  file1.close();

  IO::print2bin(outputdir + std::string("reference_points.bin"),
                reference_points);
  IO::print2bin(outputdir + std::string("interface_points.bin"),
                interface_points);

  for (int polynomial_degree_BEM = 0;
       polynomial_degree_BEM <= polynomial_degree_BEM_max;
       ++polynomial_degree_BEM) {
    sw.tic();
    std::vector<MatrixXcd> pot =
        integrand(geometry, gridpoints, refinement_level_BEM,
                  polynomial_degree_BEM, wavenumber);
    std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";

    ///////////////////////////////////////////////////////////////////////
    // save waves on interface
    ///////////////////////////////////////////////////////////////////////
    VectorXcd int_x(2 * interface_points.rows());
    int_x << pot[0].block(0, 0, interface_points.rows(), 1),
        artificial_interface.potentialGradientToNeumann(
            pot[1].block(0, 0, interface_points.rows(), 3));
    VectorXcd int_mean = int_x;
    VectorXcd int_2mom = int_x.cwiseAbs2();
    MatrixXcd int_cor = int_x * int_x.adjoint();

    VectorXcd pot_x =
        pot[0].block(interface_points.rows(), 0, reference_points.rows(), 1);
    VectorXcd pot_mean = pot_x;
    VectorXcd pot_2mom = pot_x.cwiseAbs2();
    MatrixXcd pot_cor = pot_x * pot_x.adjoint();
    std::string identifier = std::to_string(refinement_level_BEM) + "_" +
                             std::to_string(polynomial_degree_BEM) + "_" +
                             std::to_string(interface_lvl) + "_" +
                             std::to_string(interface_degree) + "_" +
                             std::to_string(wavenum) + "_" + setting;
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
  }

  return 0;
}
