// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LOCALOPERATOR_IDENTITYOPERATORBASE_H_
#define BEMBEL_LOCALOPERATOR_IDENTITYOPERATORBASE_H_

namespace Bembel {

/**
 * \ingroup LocalOperator
 * \brief This class is the base for all mass matrices.
 */
template <typename Derived>
class IdentityOperatorBase : public LocalOperatorBase<Derived> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  IdentityOperatorBase() {}
  template <class T>
  void evaluateIntegrand_impl(const T &super_space, const SurfacePoint &p1,
                              const SurfacePoint &p2,
                              Eigen::MatrixXd *intval) const {
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // get evaluation points on unit square
    auto s = p1.segment<2>(0);

    // get quadrature weights
    auto ws = p1(2);

    // get points on geometry and tangential derivatives
    auto x_f = p1.segment<3>(3);
    auto x_f_dx = p1.segment<3>(6);
    auto x_f_dy = p1.segment<3>(9);

    // compute surface measures from tangential derivatives
    auto x_kappa = x_f_dx.cross(x_f_dy).norm();

    // integrand without basis functions
    auto integrand = x_kappa * ws;

    super_space.addScaledBasisInteraction(intval, integrand, s, s);

    return;
  }
};

}  // namespace Bembel
#endif
