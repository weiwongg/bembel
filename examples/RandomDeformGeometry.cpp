#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/LinearForm>
#include <Bembel/LinearOperator>
#include <Bembel/Quadrature>
#include <Bembel/UQ>
#include <FMCA/CovarianceKernel>
#include <FMCA/Samplets>
#include <iostream>

#include "sampletKernelCompressor.h"

//
// main program
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  using namespace Bembel;
  using namespace Eigen;
  /////////////////////////////////////////////////////////////////////////////
  // parameters
  /////////////////////////////////////////////////////////////////////////////
  int polynomial_degree = 2;
  int refinement_level = 2;

  /////////////////////////////////////////////////////////////////////////////
  // load geometry
  /////////////////////////////////////////////////////////////////////////////
  Geometry geometry("../geo/sphere.dat");

  /////////////////////////////////////////////////////////////////////////////
  // Build ansatz spaces
  /////////////////////////////////////////////////////////////////////////////
  AnsatzSpace<MassMatrixScalarCont> ansatz_space(geometry, refinement_level,
                                                 polynomial_degree);
  UQ::DeformationFieldInterpolator<MassMatrixScalarCont> interp(
      geometry, refinement_level, polynomial_degree);
  const Eigen::Matrix<double, 3, -1> pts = interp.get_gridpoints().transpose();
  /////////////////////////////////////////////////////////////////////////////
  // UQ parameters
  /////////////////////////////////////////////////////////////////////////////
  // parameters
  const std::string kernel_type = "EXPONENTIAL";
  const FMCA::Index npts = pts.cols();
  const FMCA::Index dim = 3;
  const FMCA::Index mpole_deg = 3;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar eta = 0.001;
  const FMCA::Scalar threshold = 1e-8;
  const FMCA::Scalar ridgep = 1e-9;
  //////////////////////////////////////////////////////////////////////////////
  // objects
  const FMCA::CovarianceKernel kernel(kernel_type, 10);
  const Moments mom(pts, mpole_deg);
  const MatrixEvaluator mat_eval(mom, kernel);
  const SampletMoments samp_mom(pts, dtilde - 1);
  const H2SampletTree hst(mom, samp_mom, 0, pts);
  FMCA::iVector indices = Eigen::Map<const FMCA::iVector>(hst.indices().data(),
                                                          hst.indices().size());

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, unsigned int> perm(
      npts);
  perm.indices() = indices;
  //////////////////////////////////////////////////////////////////////////////
  // output parameters
  std::cout << "kernel:                       " << kernel_type << std::endl;
  std::cout << "npts:                         " << npts << std::endl;
  std::cout << "dim:                          " << dim << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  std::cout << "mpole deg:                    " << mpole_deg << std::endl;
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "threshold:                    " << threshold << std::endl;
  std::cout << "ridgep:                       " << ridgep << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // construct samplet matrix
  Eigen::SparseMatrix<FMCA::Scalar> K(npts, npts);
  std::vector<Eigen::Triplet<FMCA::Scalar>> trips =
      sampletKernelCompressor(kernel, hst, mat_eval, pts, eta, threshold);
  K.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> I(npts, npts);
  I.setIdentity();
  // K += ridgep * I;
  //////////////////////////////////////////////////////////////////////////////
  // deform geometry
  UQ::RandomGeometryDeformer<MassMatrixScalarCont> tgd(
      geometry, refinement_level, polynomial_degree, K, .2);

  Eigen::MatrixXd T = hst.inverseSampletTransform(I);
  for (int i = 0; i < 10; ++i) {
    Eigen::MatrixXd param =
        Eigen::MatrixXd::Random(interp.get_sysc().rows(), 3);
    Geometry deformed_geometry = tgd.get_geometryRealization(param, T, perm);

    ///////////////////////////////////////////////////////////////////////
    // Visualization
    ///////////////////////////////////////////////////////////////////////
    VTKSurfaceExport writer(deformed_geometry, 6);
    writer.writeToFile("deformation_geo_" + std::to_string(i) + ".vtp");
    std::cout << "finished " + std::to_string(i) + " sample." << std::endl;
  }

  return 0;
}
