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
  Bembel::UQ::PreGeometryDeformer def("../geo/" + setting + suffix, geo_scale,
                                      shift);
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s" << std::endl;
  std::cout << "UQ dimension " << def.get_parameterDimension() << std::endl;
}
