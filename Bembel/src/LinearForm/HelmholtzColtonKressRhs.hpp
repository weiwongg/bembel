// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEARFORM_HELMHOLTZCOLTONKRESSRHS_H_
#define BEMBEL_LINEARFORM_HELMHOLTZCOLTONKRESSRHS_H_

namespace Bembel {

class HelmholtzColtonKressRhs;

template <>
struct LinearFormTraits<HelmholtzColtonKressRhs> {
  typedef std::complex<double> Scalar;
};

/**
 *  \ingroup LinearForm
 *  \brief This class provides an implementation of the Neumann trace operator
 * and a corresponding method to evaluate the linear form corresponding to the
 * right hand side of the system via quadrature.
 */
class HelmholtzColtonKressRhs
    : public LinearFormBase<HelmholtzColtonKressRhs, std::complex<double>> {
 public:
  HelmholtzColtonKressRhs() {}
  void set_wavenumber(const std::complex<double> wavenumber) {
    wavenumber_ = wavenumber;
  }
  void set_dirichlet(const std::function<std::complex<double>(Eigen::Vector3d)>
                         &dirichlet_function) {
    dirichlet_function_ = dirichlet_function;
  }
  void set_neumann(const std::function<Eigen::Vector3cd(Eigen::Vector3d)>
                       &neumann_function) {
    neumann_function_ = neumann_function;
  }
  template <class T>
  void evaluateIntegrand_impl(const T &super_space, const SurfacePoint &p,
                              Eigen::VectorXcd *intval) const {
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // get evaluation points on unit square
    auto s = p.segment<2>(0);

    // get quadrature weights
    auto ws = p(2);

    // get points on geometry and tangential derivatives
    auto x_f = p.segment<3>(3);
    auto x_f_dx = p.segment<3>(6);
    auto x_f_dy = p.segment<3>(9);

    // compute (unnormalized) surface normal from tangential derivatives
    auto x_f_n = x_f_dx.cross(x_f_dy);
    auto x_kappa = x_f_n.norm();
    x_f_n /= x_kappa;

    // integrand without basis functions
    // dot: adjoint in first variable
    auto integrand = (x_f_n.dot(neumann_function_(x_f)) -
                      0.5 * std::complex<double>(0., 1.) * wavenumber_ *
                          dirichlet_function_(x_f)) *
                     x_kappa * ws;

    // multiply basis functions with integrand
    super_space.addScaledBasis(intval, integrand, s);

    return;
  };

 private:
  std::complex<double> wavenumber_;
  std::function<std::complex<double>(Eigen::Vector3d)> dirichlet_function_;
  std::function<Eigen::Vector3cd(Eigen::Vector3d)> neumann_function_;
};
}  // namespace Bembel

#endif
