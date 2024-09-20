// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LOCALOPERATOR_LAPLACEBELTRAMIOPERATORBASE_H_
#define BEMBEL_LOCALOPERATOR_LAPLACEBELTRAMIOPERATORBASE_H_

namespace Bembel {

/**
 * \ingroup LocalOperator
 * \brief This class is the base for Laplace Beltrami operator
 */
template <typename Derived>
class LaplaceBeltramiOperatorBase : public LocalOperatorBase<Derived> {
 public:
  LaplaceBeltramiOperatorBase() {}
  template <class T>
  void evaluateIntegrand_impl(const T &super_space, const SurfacePoint &p1,
                              const SurfacePoint &p2,
                              Eigen::MatrixXd *intval) const {
    // get basic information
    int polynomial_degree = super_space.get_polynomial_degree();
    int polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);
    const int elements_per_direction =
        (1 << super_space.get_refinement_level());
    const double h = 1. / ((double)(elements_per_direction));

    // compute surface measures from tangential derivatives
    double x_kappa = p1.segment<3>(6).cross(p1.segment<3>(9)).norm();

    // integrand without basis functions
    double integrand = x_kappa * p1(2) * 1.0 / h * 1.0 / h;

    super_space.addScaledSurfaceGradientInteraction(intval, integrand, p1, p2);

    return;
  }
};

}  // namespace Bembel
#endif
