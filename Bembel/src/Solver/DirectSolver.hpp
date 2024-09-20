#ifndef BEMBEL_DS_H_
#define BEMBEL_DS_H_

namespace Bembel {

// Direect Solver
namespace DS {

// Built-in iterative solvers
int cg(const Eigen::VectorXd &constant_one,
                  const Eigen::SparseMatrix<double> &S,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                  double tol = 1e-6) {
  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver;
  //solver.setTolerance(tol);
  solver.compute(S);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return 0;
  }
  Eigen::VectorXd res;
  do {
    x = solver.solveWithGuess(f, x);
    x = x - L.dot(x) / L.dot(constant_one) * constant_one;
    res = f - S * x;
    iter += solver.iterations();

  } while (res.norm() > tol);
  return iter;
}

// Built-in direct solvers
int spllt(const Eigen::VectorXd &constant_one,
                  const Eigen::SparseMatrix<double> &S,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                  double tol = 1e-6) {
  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>  solver;
  //solver.setTolerance(tol);
  solver.compute(S);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return 0;
  }
  Eigen::VectorXd res;
  x = solver.solve(f);
  x = x - L.dot(x) / L.dot(constant_one) * constant_one;
  res = f - S * x;
  iter = iter + 1;
  return iter;
}

int spldlt(const Eigen::VectorXd &constant_one,
                  const Eigen::SparseMatrix<double> &S,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                  double tol = 1e-6) {
  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>  solver;
  //solver.setTolerance(tol);
  solver.compute(S);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return 0;
  }
  Eigen::VectorXd res;
  x = solver.solve(f);
  x = x - L.dot(x) / L.dot(constant_one) * constant_one;
  res = f - S * x;
  iter = iter + 1;
  return iter;
}

int splu(const Eigen::VectorXd &constant_one,
                  const Eigen::SparseMatrix<double> &S,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                  double tol = 1e-6) {
  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
  //solver.setTolerance(tol);
  solver.compute(S);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return 0;
  }
  Eigen::VectorXd res;
  x = solver.solve(f);
  x = x - L.dot(x) / L.dot(constant_one) * constant_one;
  res = f - S * x;
  iter = iter + 1;
  return iter;
}

int spqr(const Eigen::VectorXd &constant_one,
                  const Eigen::SparseMatrix<double> &S,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                  double tol = 1e-6) {
  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::SparseQR <Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
  //solver.setTolerance(tol);
  solver.compute(S);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return 0;
  }
  Eigen::VectorXd res;
  x = solver.solve(f);
  x = x - L.dot(x) / L.dot(constant_one) * constant_one;
  res = f - S * x;
  iter = iter + 1;
  return iter;
}


}  // namespace DS
}  // namespace Bembel
#endif
