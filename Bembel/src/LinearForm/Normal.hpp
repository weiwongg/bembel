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

#ifndef BEMBEL_SRC_NORMAL_HPP_
#define BEMBEL_SRC_NORMAL_HPP_

namespace Bembel {
/**
 *  \ingroup LinearForm
 *  \brief This class takes care of the
 * assembly of the linear form of the normal of the considering surface.
 */
template <typename Derived, typename Scalar>
void Normal(const AnsatzSpace<Derived> &ansatz_space,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &u,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 3> *normal_lf, int deg = 4) {
  GaussSquare<Constants::maximum_quadrature_degree> GS;
  auto Q = GS[deg];
  SurfacePoint qp;
  const auto &super_space = ansatz_space.get_superspace();
  const ElementTree &et = super_space.get_mesh().get_element_tree();
  const unsigned int number_of_elements = et.get_number_of_elements();
  const unsigned int polynomial_degree = super_space.get_polynomial_degree();
  const unsigned n_shape_fun =
      (polynomial_degree + 1) * (polynomial_degree + 1);
  const auto longu = (ansatz_space.get_transformation_matrix() * u).eval();
  // compute linear form
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> normal_lf_long =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 3>::Zero(
          n_shape_fun * number_of_elements, 3);
  for (auto i = 0; i < Q.w_.size(); ++i) {
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basis_evaluated_at_pt =
        super_space.basis(Q.xi_.col(i));
    for (auto element = et.cpbegin(); element != et.cpend(); ++element) {
      super_space.map2surface(*element, Q.xi_.col(i), Q.w_(i), &qp);
      // get evaluation points on unit square
      const auto &s = qp.segment<2>(0);
      // get quadrature weights
      Scalar ws = qp(2);
      const Scalar h = element->get_h();
      // get points on geometry and tangential derivatives
      const auto &x_f = qp.segment<3>(3);
      const auto &x_f_dx = qp.segment<3>(6);
      const auto &x_f_dy = qp.segment<3>(9);
      const auto &normal = x_f_dx.cross(x_f_dy);
      const Scalar u_val =
          longu.segment(n_shape_fun * element->id_, n_shape_fun).transpose() *
          basis_evaluated_at_pt;
      for (auto j = 0; j <= 2; ++j) {
        normal_lf_long.block(n_shape_fun * element->id_, j, n_shape_fun, 1) +=
            Q.w_(i) * normal(j) * u_val * basis_evaluated_at_pt;
      }
    }
  }
  normal_lf->derived() =
      ansatz_space.get_transformation_matrix().transpose() * normal_lf_long;
}

}  // namespace Bembel
#endif  // BEMBEL_SRC_NORMAL_HPP_
