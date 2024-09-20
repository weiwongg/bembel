#include <Bembel/IO>
#include <Bembel/LocalOperator>
#include <Bembel/UQ>

class FakeDivergenceConformingOperator;
template <>
struct Bembel::LinearOperatorTraits<FakeDivergenceConformingOperator> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum { OperatorOrder = 0, Form = DifferentialForm::DivConforming };
};

int main() {
  using namespace Bembel;
  using namespace Eigen;
  IO::Stopwatch sw;
  Geometry geo("bunny5.bpd", 10.0);
  Bembel::UQ::GeometryDeformer def("bunny5.bpd", 10.0);
  // int dim = def.set_parameterDimension(24);
  int dim = def.get_parameterDimension();
  std::cout << dim << std::endl;
  // std::cout<<def.get_SingularValues()<<std::endl;
  Eigen::VectorXd param(dim);
  std::function<double(const Eigen::Vector3d &)> fun2 =
      [](const Eigen::Vector3d &point_in_space) { return point_in_space(0); };
  {
    param.setZero();
    geo = def.get_geometryRealization(param);
    VTKSurfaceExport writer(geo, 4);
    writer.addDataSet("X-Value", fun2);
    writer.writeToFile("RefGeo.vtp");
  }
  for (auto i = 0; i < 20; ++i) {
    param = Eigen::VectorXd::Random(dim);

    for (auto j = 0; j < 5; ++j) {
      geo = def.get_geometryRealization((1 + 0.25 * j) *
                                        def.get_SingularValues() * param);
      VTKSurfaceExport writer(geo, 4);
      writer.addDataSet("X-Value", fun2);
      writer.writeToFile("geo_" + std::to_string(j) + "_" + std::to_string(i) +
                         ".vtp");
    }
  }
}
