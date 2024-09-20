// This file is part of Bembel, the higher order C++ boundary element library.
//
// Copyright (C) 2022 see <http://www.bembel.eu>
//
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

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
#include "sampletKernelCompressor.h"

// other helpers
#include "Grids.hpp"

int main() {
  using namespace Bembel;
  using namespace Eigen;

  Bembel::IO::Stopwatch sw;

  int polynomial_degree = 2;
  int refinement_level = 2;
  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry geometry("../geo/platewith4holes.dat");
  AnsatzSpace<MassMatrixScalarCont> ansatz_space(geometry, refinement_level,
                                                 polynomial_degree);
  DiscreteLocalOperator<MassMatrixScalarCont> disc_op(ansatz_space);
  disc_op.compute();
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
  const FMCA::Index mpole_deg = 5;
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
      geometry, refinement_level, polynomial_degree, K, 0.2);

  Eigen::MatrixXd T = hst.inverseSampletTransform(I);
  MatrixXd gridpoints = Util::makePlaneGrid(4.0, 8.0, 200, 500);
  for (int i = 0; i < 10; ++i) {
    Eigen::MatrixXd param =
        Eigen::MatrixXd::Random(interp.get_sysc().rows(), 3);
    Geometry deformed_geometry = tgd.get_geometryRealization(param, T, perm);
    VTKSurfaceExport writer(deformed_geometry, 6);
    writer.writeToFile("deformation_geo_" + std::to_string(i) + ".vtp");
    // Define analytical solution using lambda function, in this case the
    // Helmholtz fundamental solution centered on 0, see Data.hpp
    std::complex<double> wavenumber(-6., 0.);
    Vector3d d(sqrt(3.0) / 3.0, sqrt(3.0) / 3.0, sqrt(3.0) / 3.0);
    const std::function<std::complex<double>(Vector3d)> fun = [wavenumber,
                                                               d](Vector3d pt) {
      return std::exp(-std::complex<double>(0, 1) * wavenumber * pt.dot(d));
    };
    const std::function<Vector3cd(Vector3d)> fun_grad = [wavenumber, d,
                                                         &fun](Vector3d pt) {
      return (-std::complex<double>(0, 1) * wavenumber * fun(pt)) * d;
    };
    sw.tic();
    int refinement_level_BEM = 2;
    int polynomial_degree_BEM = 2;
    // Build ansatz space
    AnsatzSpace<HelmholtzColtonKressOperator> ansatz_space_ck(
        deformed_geometry, refinement_level_BEM, polynomial_degree_BEM);
    AnsatzSpace<HelmholtzSingleLayerOperator> ansatz_space_single(
        deformed_geometry, refinement_level_BEM, polynomial_degree_BEM);
    AnsatzSpace<MassMatrixScalarDisc> ansatz_space_mass(
        deformed_geometry, refinement_level_BEM, polynomial_degree_BEM);

    // Set up load vector
    DiscreteLinearForm<HelmholtzColtonKressRhs, HelmholtzColtonKressOperator>
        disc_lf_ck(ansatz_space_ck);
    disc_lf_ck.get_linear_form().set_dirichlet(fun);
    disc_lf_ck.get_linear_form().set_neumann(fun_grad);
    disc_lf_ck.get_linear_form().set_wavenumber(wavenumber);
    disc_lf_ck.compute();

    // Set up and compute discrete operator
    DiscreteOperator<H2Matrix<std::complex<double>>,
                     HelmholtzColtonKressOperator>
        disc_op_ck(ansatz_space_ck);
    disc_op_ck.get_linear_operator().set_wavenumber(wavenumber);
    disc_op_ck.compute();
    DiscreteOperator<SparseMatrix<double>, MassMatrixScalarDisc> disc_op_mass(
        ansatz_space_mass);
    disc_op_mass.compute();

    SparseMatrix<std::complex<double>> mass =
        (0.5 *
         disc_op_mass.get_discrete_operator().cast<std::complex<double>>())
            .eval();
    FormalSum<SparseMatrix<std::complex<double>>,
              H2Matrix<std::complex<double>>>
        system_matrix_ck(mass, disc_op_ck.get_discrete_operator());
    // solve system
    GMRES<FormalSum<SparseMatrix<std::complex<double>>,
                    H2Matrix<std::complex<double>>>,
          IdentityPreconditioner>
        gmres_ck;
    gmres_ck.set_restart(300);
    gmres_ck.setTolerance(1e-8);
    gmres_ck.compute(system_matrix_ck);
    Eigen::VectorXcd rho =
        gmres_ck.solve(disc_lf_ck.get_discrete_linear_form());

    // evaluate potential
    DiscretePotential<
        HelmholtzSingleLayerPotential<HelmholtzSingleLayerOperator>,
        HelmholtzSingleLayerOperator>
        disc_pot_drlt(ansatz_space_single);
    disc_pot_drlt.get_potential().set_wavenumber(wavenumber);
    disc_pot_drlt.set_cauchy_data(rho);
    std::function<std::complex<double>(const Vector3d &)> u_drlt =
        [&disc_pot_drlt](const Vector3d &pt) {
          auto pot = disc_pot_drlt.evaluate(pt.transpose());
          return pot[0];
        };
    DiscretePotential<HelmholtzSingleLayerPotentialNormalDerivative<
                          HelmholtzSingleLayerOperator>,
                      HelmholtzSingleLayerOperator>
        disc_pot_neum(ansatz_space_single);
    disc_pot_neum.get_potential().set_wavenumber(wavenumber);
    disc_pot_neum.set_cauchy_data(rho);
    std::function<std::complex<double>(const Vector3d &)> u_neum =
        [&disc_pot_neum](const Vector3d &pt) {
          auto pot = disc_pot_neum.evaluate(pt.transpose());
          return pot[0];
        };

    /////////////////////////////////////////////////////////////////////////////
    // Artifical interface
    /////////////////////////////////////////////////////////////////////////////
    Geometry artifical_interface("../geo/artifical_sphere.dat");
    // Build ansatz space
    int interface_lvl = 0;
    int interface_degree = 6;
    AnsatzSpace<MassMatrixComplexDisc> artifical_ansatz_space(
        artifical_interface, interface_lvl, interface_degree);
    DiscreteLocalOperator<MassMatrixComplexDisc> artifical_disc_op(
        artifical_ansatz_space);
    artifical_disc_op.compute();
    SparseLU<SparseMatrix<std::complex<double>>, COLAMDOrdering<int>> slu;
    // Compute the ordering permutation vector from the structural pattern of
    slu.analyzePattern(artifical_disc_op.get_discrete_operator());
    // Compute the numerical factorization
    slu.factorize(artifical_disc_op.get_discrete_operator());
    // Use the factors to solve the linear system
    DiscreteLinearForm<DirichletTrace<std::complex<double>>,
                       MassMatrixComplexDisc>
        artifical_disc_lf_drlt(artifical_ansatz_space);
    artifical_disc_lf_drlt.get_linear_form().set_function(u_drlt);
    artifical_disc_lf_drlt.compute();
    Eigen::VectorXcd artifical_x_drlt =
        slu.solve(artifical_disc_lf_drlt.get_discrete_linear_form());

    DiscreteLinearForm<DirichletTrace<std::complex<double>>,
                       MassMatrixComplexDisc>
        artifical_disc_lf_neum(artifical_ansatz_space);
    artifical_disc_lf_neum.get_linear_form().set_function(u_neum);
    artifical_disc_lf_neum.compute();
    Eigen::VectorXcd artifical_x_neum =
        slu.solve(artifical_disc_lf_neum.get_discrete_linear_form());
    if (true) {
      ArtificialPotentialEval<
          HelmholtzSingleLayerOperator,
          HelmholtzSingleLayerPotential<HelmholtzSingleLayerOperator>,
          HelmholtzDoubleLayerOperator,
          HelmholtzDoubleLayerPotential<HelmholtzDoubleLayerOperator>>
          artificial_interface(artifical_interface, interface_lvl,
                               interface_degree);
      artificial_interface.setBoundaryData(artifical_x_drlt, artifical_x_neum,
                                           wavenumber);

      VectorXcd pot_fast = artificial_interface.evaluate(gridpoints);
      VectorXcd pot = disc_pot_drlt.evaluate(gridpoints);
      std::cout << (pot_fast - pot).cwiseAbs().maxCoeff() << std::endl;
      Eigen::MatrixXd info(gridpoints.rows(), 5);
      info.col(0) = gridpoints.col(0);
      info.col(1) = gridpoints.col(1);
      info.col(2) = gridpoints.col(2);
      info.col(3) = pot_fast.real();
      info.col(4) = pot_fast.imag();

      std::ofstream out;
      out.open("./scatter_" + std::to_string(i) + ".csv");
      Eigen::IOFormat CommaInitFmt(FullPrecision, 0, ", ", "\n");
      out << info.format(CommaInitFmt) << "\n";
    }

    std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
    std::cout << "finished " + std::to_string(i) + " sample. \n" << std::endl;
  }
  return 0;
}
