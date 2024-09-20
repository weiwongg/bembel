
// C++
#include <cmath>
#include <iostream>

// Eigen
#include <Eigen/Dense>

// Bembel
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/LinearForm>
#include <Bembel/Quadrature>
#include <Bembel/src/util/Macros.hpp>

std::function<double(Eigen::Matrix<double, 3, 1>)> fun_mag =
    [](Eigen::Matrix<double, 3, 1> in) { return 1.0; };

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
  Geometry geometry("example.dat");

  /////////////////////////////////////////////////////////////////////////////
  // Build ansatz spaces
  /////////////////////////////////////////////////////////////////////////////
  AnsatzSpace<MassMatrixScalarCont> ansatz_space(geometry, refinement_level,
                                                 polynomial_degree);

  /////////////////////////////////////////////////////////////////////////////
  // Build deformation vector
  /////////////////////////////////////////////////////////////////////////////
  DiscreteLocalOperator<MassMatrixScalarCont> mass_matrix(ansatz_space);
  mass_matrix.compute();
  SparseMatrix<double> M = mass_matrix.get_discrete_operator();
  SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
  solver.analyzePattern(M);
  solver.factorize(M);
  DiscreteLinearForm<DirichletTrace<double>, MassMatrixScalarCont> disc_lf(
      ansatz_space);
  disc_lf.get_linear_form().set_function(fun_mag);
  disc_lf.compute();
  VectorXd mag = solver.solve(disc_lf.get_discrete_linear_form());
  Matrix<double, Dynamic, 3> normal_term;
  Normal(ansatz_space, mag, &normal_term);
  Matrix<double, Dynamic, 3> deform = MatrixXd::Zero(M.cols(), 3);
  deform.col(0) = solver.solve(normal_term.col(0));
  deform.col(1) = solver.solve(normal_term.col(1));
  deform.col(2) = solver.solve(normal_term.col(2));

  ///////////////////////////////////////////////////////////////////////
  // deformation field for geometry
  ///////////////////////////////////////////////////////////////////////
  DeformationField<MassMatrixScalarCont> deformation(ansatz_space);
  deformation.set_function(deform);

  ///////////////////////////////////////////////////////////////////////
  // Build deformed geometry
  ///////////////////////////////////////////////////////////////////////
  GeometryDeformator geometry_deformator(geometry, refinement_level,
                                         polynomial_degree);
  geometry_deformator.calculate_deformation(deformation.get_patch_vector());
  Geometry deformed_geometry = geometry_deformator.get_deformed();

  ///////////////////////////////////////////////////////////////////////
  // Visualization
  ///////////////////////////////////////////////////////////////////////
  VTKSurfaceExport writer(geometry, 6);
  FunctionEvaluator<MassMatrixScalarCont> evaluator_x(ansatz_space);
  FunctionEvaluator<MassMatrixScalarCont> evaluator_y(ansatz_space);
  FunctionEvaluator<MassMatrixScalarCont> evaluator_z(ansatz_space);
  evaluator_x.set_function(deform.col(0));
  evaluator_y.set_function(deform.col(1));
  evaluator_z.set_function(deform.col(2));
  std::function<Vector3d(int, const Vector2d &)> fun_deformation =
      [&](int patch_number, const Vector2d &reference_domain_point) {
        return Vector3d(evaluator_x.evaluateOnPatch(patch_number,
                                                    reference_domain_point)(0),
                        evaluator_y.evaluateOnPatch(patch_number,
                                                    reference_domain_point)(0),
                        evaluator_z.evaluateOnPatch(patch_number,
                                                    reference_domain_point)(0));
      };
  writer.addDataSet("Deformation", fun_deformation);
  writer.writeToFile("deformation_field.vtp");
  VTKSurfaceExport writer2(deformed_geometry, 6);
  writer2.writeToFile("deformed_geometry.vtp");
  std::cout << "finished" << std::endl;
  return 0;
}
