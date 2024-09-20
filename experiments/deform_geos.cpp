#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/UQ>

#include "Util/Inputs.hpp"

class FakeDivergenceConformingOperator;
template <>
struct Bembel::LinearOperatorTraits<FakeDivergenceConformingOperator> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum { OperatorOrder = 0, Form = DifferentialForm::DivConforming };
};

int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  InputParser input(argc, argv);
    std::string setting = input.getRequiredCmd("-setting");
    double geo_scale = 1.0;
    Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0);
    std::string suffix;
    
    if (!setting.compare("david")) {
      geo_scale = 2.0;
      shift = Eigen::Vector3d(0, 0, -0.3);
      suffix = "10.bpd";
    } else if (!setting.compare("torus")) {
      geo_scale = 0.4;
      suffix = "20.bpd";
    } else if (!setting.compare("sphere")) {
      suffix = "20.bpd";
    } else {
      assert(!"Setting not known");
    }
  std::cout << "Loading geometries..." << std::endl;
  IO::Stopwatch sw;
  sw.tic();
  Geometry geo;
  Bembel::UQ::GeometryDeformer def("../geo/" + setting + suffix, geo_scale,
                                   shift);
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s" << std::endl;
  std::cout << "UQ dimension " << def.get_parameterDimension() << std::endl;

  // int dim = def.set_parameterDimension(24);
  int dim = def.get_parameterDimension();
  Eigen::MatrixXd Q = Eigen::MatrixXd::Random(dim, 10);
  assert(dim == Q.rows());
  // std::cout<<def.get_SingularValues()<<std::endl;
  Eigen::VectorXd param(dim);
  std::function<double(const Eigen::Vector3d &)> fun =
      [](const Eigen::Vector3d &point_in_space) { return point_in_space(0); };
  for (auto i = 0; i < 10; ++i) {
    param = Q.col(i);
    if (i == 0)
      geo = def.get_geometryRealization(param, 0.0);
    else
      geo = def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);
    VTKSurfaceExport writer(geo, 6);
    writer.addDataSet("X-Value", fun);
    writer.writeToFile("mc_geo_" + std::to_string(i) + ".vtp");
  }
}
