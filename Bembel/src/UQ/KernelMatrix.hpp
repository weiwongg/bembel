// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_UQ_KERNELMATRIX_H_
#define BEMBEL_UQ_KERNELMATRIX_H_

namespace Bembel {

namespace UQ {

std::function<double(double)> Matern32Kernel = [](double r) {
  double d = 20 * r;
  return (1. + sqrt(3) * d) * exp(-sqrt(3) * d);
};

std::function<double(double)> Matern52Kernel = [](double r) {
  double d = 10 * r;
  return (1 + sqrt(5) * d + 5. / 3. * d * d) * std::exp(-sqrt(5) * d);
};

std::function<double(double)> Matern72Kernel = [](double r) {
  double d = r;
  return (1. + sqrt(7) * d + 14. / 5 * d * d + 7. * sqrt(7) / 15. * d * d * d) *
         exp(-sqrt(7) * d);
};

std::function<double(double)> Matern92Kernel = [](double r) {
  return (1. + 3. * r + 27. / 7. * r * r + 18. / 7. * r * r * r +
          27. / 35. * pow(r, 4.)) *
         exp(-3 * r);
};

std::function<double(double)> MaternInfKernel = [](double r) {
  double d = r;
  return std::exp(-0.5 * d * d);
};

std::function<double(double)> MaternInfKernelSmall = [](double r) {
  double d = 4 * r;
  return 1e-4 * std::exp(-0.5 * d * d);
};

/**
 * \ingroup UQ
 * \brief wraps a kernel matrix into a lightweight data structure.
 */
template <typename Derived>
struct KernelMatern {
  //////////////////////////////////////////////////////////////////////////////
  KernelMatern(const Eigen::EigenBase<Derived> &pts)
      : pts_(pts.derived()), dim_(pts.rows() * pts.cols()) {
    kernel_.resize(pts.rows(), pts.rows());
    kernel_(0, 0) = Matern32Kernel;
    kernel_(1, 1) = Matern32Kernel;
    kernel_(2, 2) = Matern32Kernel;
    kernel_(2, 0) = MaternInfKernelSmall;
    kernel_(0, 2) = MaternInfKernelSmall;
    kernel_(0, 1) = MaternInfKernelSmall;
    kernel_(1, 0) = MaternInfKernelSmall;
    kernel_(1, 2) = MaternInfKernelSmall;
    kernel_(2, 1) = MaternInfKernelSmall;
  };
  //////////////////////////////////////////////////////////////////////////////
  unsigned int get_dim() const { return dim_; }
  //////////////////////////////////////////////////////////////////////////////
  double operator()(int i, int j) const {
    const int blocki = int(i / pts_.cols());
    const int blockj = int(j / pts_.cols());
    const int loci = i - blocki * pts_.cols();
    const int locj = j - blockj * pts_.cols();
    double r = (pts_.col(loci) - pts_.col(locj)).norm() / 10;
    return kernel_(blocki, blockj) ? kernel_(blocki, blockj)(r) : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::VectorXd col(Eigen::Index j) const {
    Eigen::VectorXd retval(dim_);
    for (auto i = 0; i < dim_; ++i) retval(i) = operator()(i, j);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::VectorXd diagonal() const {
    Eigen::VectorXd retval(dim_);
    for (auto i = 0; i < dim_; ++i) retval(i) = operator()(i, i);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix<std::function<double(double)>> kernel_;
  const Derived &pts_;
  const unsigned int dim_;
};

}  // namespace UQ
}  // namespace Bembel
#endif
