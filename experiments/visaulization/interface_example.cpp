#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/UQ>

// Random domain
#include "RandomDomain/Auxilliary.h"

// other helpers
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

// #include "HaltonSet.hpp"
// #include <FMCA/src/util/HaltonSet.h>

int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  InputParser input(argc, argv);

  IO::Stopwatch sw;
  sw.tic();
  Geometry interface_geometry;  // geometry for interface
  double sphere_radius;
  double geo_scale = 2.0;
  Eigen::Vector3d shift = Eigen::Vector3d(0, 0, -0.3);
  std::cout << "Loading geometries..." << std::endl;
    interface_geometry =
        Geometry("../../geo/refined_cube.bpd", 4.0, Eigen::Vector3d(-0.5, -0.5, -0.5));
    sphere_radius = 5.0;
  
  Geometry geo;
  Bembel::UQ::GeometryDeformer def("../../geo/david10.bpd", geo_scale,
                                   shift);
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s" << std::endl;

  int refinement_level_BEM = 1;
  int polynomial_degree_BEM = 4;
  int interface_degree = 8;
  double wavenumber = Bembel::Constants::BEMBEL_WAVENUM;

  /////////////////////////////////////////////////////////////////////////////
  // define artificial interface
  /////////////////////////////////////////////////////////////////////////////
  ArtificialInterface<
      HelmholtzSingleLayerOperator,
      HelmholtzSingleLayerPotential<HelmholtzSingleLayerOperator>,
      HelmholtzDoubleLayerOperator,
      HelmholtzDoubleLayerPotential<HelmholtzDoubleLayerOperator>>
      artificial_interface(interface_geometry, 0, interface_degree);
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

  // int dim = def.set_parameterDimension(24);
  int dim = def.get_parameterDimension();
  Eigen::MatrixXd Q = Eigen::MatrixXd::Random(dim, 10);
  assert(dim == Q.rows());
  // std::cout<<def.get_SingularValues()<<std::endl;
  Eigen::VectorXd param(dim);
  std::function<double(const Eigen::Vector3d &)> fun =
      [](const Eigen::Vector3d &point_in_space) { return point_in_space(0); };

  for (auto i = 0; i < 10; ++i) {
    sw.tic();
    param = Q.col(i);
    geo = def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);
    VTKSurfaceExport writer(geo, 6);
    writer.addDataSet("X-Value", fun);
    writer.writeToFile("geo_" + std::to_string(i) + ".vtp");
    std::vector<MatrixXcd> pot =
        integrand(geo, gridpoints, refinement_level_BEM, polynomial_degree_BEM,
                  wavenumber);

    ///////////////////////////////////////////////////////////////////////
    // plot waves on interface
    ///////////////////////////////////////////////////////////////////////
    AnsatzSpace<MassMatrixComplexDisc> artifical_ansatz_space(
        interface_geometry, 0, interface_degree);
    // Visualization
    FunctionEvaluator<MassMatrixComplexDisc> evaluator(artifical_ansatz_space);
    evaluator.set_function(artificial_interface.interpolate(
        pot[0].block(0, 0, interface_points.rows(), 1)));
    VTKSurfaceExport writer_(interface_geometry, 6);
    std::function<double(int, const Eigen::Vector2d &)> fun1 =
        [&](int patch_number, const Eigen::Vector2d &reference_domain_point) {
          auto retval =
              evaluator.evaluateOnPatch(patch_number, reference_domain_point);
          return double(retval(0, 0).real());
        };
    std::function<double(int, const Eigen::Vector2d &)> fun2 =
        [&](int patch_number, const Eigen::Vector2d &reference_domain_point) {
          auto retval =
              evaluator.evaluateOnPatch(patch_number, reference_domain_point);
          return double(retval(0, 0).imag());
        };
    writer_.addDataSet("u_drlt_real", fun1);
    writer_.addDataSet("u_drlt_imag", fun2);
    writer_.writeToFile("artificial_interface_" + std::to_string(i) + ".vtp");
    std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
    std::cout << "finished " + std::to_string(i) + " sample. \n" << std::endl;

    Eigen::MatrixXd scatter_data(reference_points.rows(), 5);
    scatter_data.block(0, 0, reference_points.rows(), 3) = reference_points;
    scatter_data.block(0, 3, reference_points.rows(), 1) =
        pot[0]
            .block(interface_points.rows(), 0, reference_points.rows(), 1)
            .real();
    scatter_data.block(0, 4, reference_points.rows(), 1) =
        pot[0]
            .block(interface_points.rows(), 0, reference_points.rows(), 1)
            .imag();
    std::ofstream file("reference_data" + std::to_string(i) + " .csv");
    Eigen::IOFormat CommaInitFmt(Eigen::FullPrecision, 0, ", ", "\n");
    file << scatter_data.format(CommaInitFmt);
    file.close();
  }
}
