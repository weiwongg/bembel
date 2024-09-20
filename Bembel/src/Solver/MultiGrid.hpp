#ifndef BEMBEL_MG_H_
#define BEMBEL_MG_H_

namespace Bembel {

namespace MG {

const int __minLvl = 1;
const int __cycleType = 2;
const int __preSmooth = 3;
const int __postSmooth = 3;

// This is only for the Laplace Beltrami with Neumann condition

/*
 *  Add Gauss Seidel smoother for use with multigrid.
 */
void GaussSeidelSmoother(const Eigen::SparseMatrix<double> &S,
                         const Eigen::VectorXd &b, Eigen::VectorXd &x,
                         int steps) {
  for (auto i = 0; i < steps; ++i)
    x = S.triangularView<Eigen::Lower>().solve(
        b - S.triangularView<Eigen::StrictlyUpper>() * x);
}

void mmgStep(const std::vector<Eigen::SparseMatrix<double>> &Ss,
             const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
             Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
             const std::vector<Eigen::SparseMatrix<double>> &Ps, int lvl) {
  if (lvl <= __minLvl) {
    Eigen::MatrixXd A = Ss[lvl];
    x = A.fullPivHouseholderQr().solve(f);
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>  solver;
    //solver.compute(Ss[lvl]);
    //x = solver.solve(f);
  } else {
    // Pre-smoothe
    GaussSeidelSmoother(Ss[lvl], f, x, __preSmooth);
    Eigen::VectorXd r = f - Ss[lvl] * x;
    // Restrict the residual
    r = Ps[lvl - 1].transpose() * r;
    Eigen::VectorXd y = Eigen::VectorXd::Zero(r.size());
    // Recursively calling mmgStep for computing the error
    for (auto i = 0; i < __cycleType; ++i) mmgStep(Ss, r, y, Ps, lvl - 1);
    // Prolongate the error
    y = Ps[lvl - 1] * y;
    // Correction
    x += y;
    // Post-smoothe
    GaussSeidelSmoother(Ss[lvl], f, x, __preSmooth);
  }
}

int mmg(const Eigen::VectorXd &constant_one,
        const Eigen::SparseMatrix<double> &S,
        const Eigen::Matrix<double, Eigen::Dynamic, 1> &L,
        const Eigen::Matrix<double, Eigen::Dynamic, 1> &f,
        Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
        const std::vector<Eigen::SparseMatrix<double>> &Ps, int lvl,
        double tol = 1e-6) {
  // stiffness matrices
  std::vector<Eigen::SparseMatrix<double>> Ss;
  // push all the stiffness matrices and regularization terms
  Ss.push_back(S);
  for (int i = lvl - 1; i >= 0; --i)
    Ss.push_back(Ps[i].transpose() * Ss.back() * Ps[i]);
  std::reverse(Ss.begin(), Ss.end());

  // Multi Grid Part
  int iter = 0;
  // initialize the x with zeros
  x.resize(f.size());
  x.setZero();
  Eigen::VectorXd res;
  do {
    // repetitively call mmgStep()
    mmgStep(Ss, f, x, Ps, lvl);
    x = x - L.dot(x) / L.dot(constant_one) * constant_one;
    res = f -  S * x;
    ++iter;
  } while (res.norm() > tol);
  return iter;
}

}  // namespace MG
}  // namespace Bembel
#endif
