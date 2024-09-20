
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"

using namespace Bembel;
using namespace Eigen;
// Newton potential
std::function<double(Vector3d)> nf = [](Vector3d in) {
  return -(in(0) * in(0) + in(1) * in(1) + in(2) * in(2)) / 6.0;
};
// g
std::function<double(Vector3d)> g = [](Vector3d in) { return 1.0; };
// h
std::function<double(Vector3d)> h = [](Vector3d in) { return 4.0; };
// funGrad
std::function<Vector3d(Vector3d)> funGrad = [](Vector3d in) {
  return Eigen::Vector3d(-in(0) / 3.0, -in(1) / 3.0, -in(2) / 3.0);
};

int main(int argc, char *argv[]) {
  const int refinement_level = 2;
  const int polynomial_degree = 2;
  // Load geometries
  Geometry geo_sigma("../geo/inward_L_shape.dat");  // Sigam
  // Geometry geo_gamma("../geo/outward_sphere.dat");       // Gamma
  const int it = std::atoi(argv[1]);
  Geometry geo_gamma("../geo/sphere.dat");
  PatchVector pv = geo_sigma.get_geometry();
  pv.insert(pv.end(), geo_gamma.get_geometry().begin(),
            geo_gamma.get_geometry().end());
  Geometry geo = Geometry(pv);  // Sigma+Gamma
  // Build ansatz space
  AnsatzSpace<MassMatrixScalarCont> ansatz_space_mass_sigma(
      geo_sigma, refinement_level, polynomial_degree);
  AnsatzSpace<MassMatrixScalarCont> ansatz_space_mass_gamma(
      geo_gamma, refinement_level, polynomial_degree);
  AnsatzSpace<LaplaceSingleLayerOperator> ansatz_space_single(
      geo, refinement_level, polynomial_degree);
  AnsatzSpace<LaplaceDoubleLayerOperator> ansatz_space_double(
      geo, refinement_level, polynomial_degree);

  int n_sigma = ansatz_space_mass_sigma.get_number_of_dofs();
  int n_gamma = ansatz_space_mass_gamma.get_number_of_dofs();
  int n = n_sigma + n_gamma;  // number of bases

  // Build the left hand side vector
  DiscreteLocalOperator<MassMatrixScalarCont> mass_matrix_sigma(
      ansatz_space_mass_sigma);
  mass_matrix_sigma.compute();
  SparseMatrix<double> M_sigma = mass_matrix_sigma.get_discrete_operator();
  SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver_sigma;
  solver_sigma.analyzePattern(M_sigma);
  solver_sigma.factorize(M_sigma);
  DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarCont> sigma_lf(
      ansatz_space_mass_sigma);
  sigma_lf.get_linear_form().set_function(g);
  sigma_lf.compute();
  DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarCont> sigma_dlf(
      ansatz_space_mass_sigma);
  sigma_dlf.get_linear_form().set_function(nf);
  sigma_dlf.compute();  // Sigam

  DiscreteLocalOperator<MassMatrixScalarCont> mass_matrix_gamma(
      ansatz_space_mass_gamma);
  mass_matrix_gamma.compute();
  SparseMatrix<double> M_gamma = mass_matrix_gamma.get_discrete_operator();
  SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver_gamma;
  solver_gamma.analyzePattern(M_gamma);
  solver_gamma.factorize(M_gamma);
  DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarCont> gamma_lf(
      ansatz_space_mass_gamma);
  gamma_lf.get_linear_form().set_function(h);
  gamma_lf.compute();
  DiscreteLinearForm<NeumannTrace<double>, MassMatrixScalarCont> gamma_nlf(
      ansatz_space_mass_gamma);
  gamma_nlf.get_linear_form().set_function(funGrad);
  gamma_nlf.compute();  // Gamma

  VectorXd rhs_v(n);
  rhs_v.head(n_sigma) =
      solver_sigma.solve(sigma_lf.get_discrete_linear_form() -
                         sigma_dlf.get_discrete_linear_form());
  rhs_v.tail(n_gamma) =
      -solver_gamma.solve(gamma_lf.get_discrete_linear_form() +
                          gamma_nlf.get_discrete_linear_form());

  // Compute single and double layer operators on the entire surface
  DiscreteOperator<MatrixXd, LaplaceSingleLayerOperator> disc_op_single(
      ansatz_space_single);
  disc_op_single.compute();
  DiscreteOperator<MatrixXd, LaplaceDoubleLayerOperator> disc_op_double(
      ansatz_space_double);
  disc_op_double.compute();
  // Build lhs and rhs matrices
  MatrixXd lhs_mat = MatrixXd::Zero(n, n);
  MatrixXd rhs_mat = MatrixXd::Zero(n, n);
  lhs_mat.block(0, 0, n, n_sigma) =
      -disc_op_single.get_discrete_operator().block(0, 0, n, n_sigma);
  lhs_mat.block(0, n_sigma, n, n_gamma) =
      disc_op_double.get_discrete_operator().block(0, n_sigma, n, n_gamma);
  lhs_mat.block(n_sigma, n_sigma, n_gamma, n_gamma) =
      lhs_mat.block(n_sigma, n_sigma, n_gamma, n_gamma) + 0.5 * M_gamma;
  rhs_mat.block(0, 0, n, n_sigma) =
      -disc_op_double.get_discrete_operator().block(0, 0, n, n_sigma);
  rhs_mat.block(0, n_sigma, n, n_gamma) =
      disc_op_single.get_discrete_operator().block(0, n_sigma, n, n_gamma);
  rhs_mat.block(0, 0, n_sigma, n_sigma) =
      rhs_mat.block(0, 0, n_sigma, n_sigma) - 0.5 * M_sigma;

  // Solve Cauchy data
  VectorXd rho = lhs_mat.fullPivLu().solve(rhs_mat * rhs_v);
  DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarCont> gamma_lf_nf(
      ansatz_space_mass_gamma);
  gamma_lf_nf.get_linear_form().set_function(nf);
  gamma_lf_nf.compute();
  auto nf_v = solver_gamma.solve(gamma_lf_nf.get_discrete_linear_form());
  // Update Geometry
  Matrix<double, Dynamic, 3> normal_term;
  VectorXd mag = 0.1 * (rho.tail(n_gamma) + nf_v);
  Normal(ansatz_space_mass_gamma, mag, &normal_term);
  Matrix<double, Dynamic, 3> deform = MatrixXd::Zero(n_gamma, 3);
  deform.col(0) = solver_gamma.solve(normal_term.col(0));
  deform.col(1) = solver_gamma.solve(normal_term.col(1));
  deform.col(2) = solver_gamma.solve(normal_term.col(2));
  DeformationField<MassMatrixScalarCont> deformation(ansatz_space_mass_gamma);
  deformation.set_function(deform);
  GeometryDeformator geometry_deformator(geo_gamma, refinement_level,
                                         polynomial_degree);
  geometry_deformator.calculate_deformation(deformation.get_patch_vector());
  geo_gamma = geometry_deformator.get_deformed();
  VTKSurfaceExport writer(geo_gamma, 6);
  writer.writeToFile("deformed_gamma.vtp");

  return 0;
}
