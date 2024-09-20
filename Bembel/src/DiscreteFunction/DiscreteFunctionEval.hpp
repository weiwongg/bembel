// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_DISCRETEFUNCTIONEVAL_H_
#define BEMBEL_ANSATZSPACE_DISCRETEFUNCTIONEVAL_H_

namespace Bembel {
template <typename Scalar, unsigned int DF, typename LinOp>
struct DiscreteFunctionEval {};

// continuous
template <typename Scalar, typename LinOp>
struct DiscreteFunctionEval<Scalar, DifferentialForm::Continuous, LinOp> {
  Eigen::Matrix<Scalar,
                getFunctionSpaceOutputDimension<DifferentialForm::Continuous>(),
                1>
  eval(const SuperSpace<LinOp> &super_space,
       const int polynomial_degree_plus_one_squared,
       const ElementTreeNode &element, const SurfacePoint &p,
       const Eigen::Matrix<
           Scalar, Eigen::Dynamic,
           getFunctionSpaceVectorDimension<DifferentialForm::Continuous>()>
           &coeff) const {
    auto s = p.segment<2>(0);
    return coeff.transpose() * super_space.basis(s) / element.get_h();
  };
  Eigen::Matrix<Scalar, 3,
                getFunctionSpaceOutputDimension<DifferentialForm::Continuous>()>
  eval_surf_grad(
      const SuperSpace<LinOp> &super_space,
      const int polynomial_degree_plus_one_squared,
      const ElementTreeNode &element, const SurfacePoint &p,
      const Eigen::Matrix<
          Scalar, Eigen::Dynamic,
          getFunctionSpaceVectorDimension<DifferentialForm::Continuous>()>
          &coeff) const {
    auto s = p.segment<2>(0);
    auto x_f_dx = p.segment<3>(6);
    auto x_f_dy = p.segment<3>(9);
    Eigen::Matrix<double, 3, 2> s_j;
    s_j.block(0, 0, 3, 1) = x_f_dx;
    s_j.block(0, 1, 3, 1) = x_f_dy;
    Eigen::MatrixXd s_drho(2, polynomial_degree_plus_one_squared);
    s_drho.block(0, 0, 1, polynomial_degree_plus_one_squared) =
        super_space.basisDx(s).transpose();
    s_drho.block(1, 0, 1, polynomial_degree_plus_one_squared) =
        super_space.basisDy(s).transpose();
    Eigen::Matrix<double, 3, Eigen::Dynamic> s_g;
    s_g = s_j * (s_j.transpose() * s_j).inverse() * s_drho;
    return s_g * coeff / (element.get_h() * element.get_h());
  };
  Eigen::Matrix<Scalar, 3,
                getFunctionSpaceOutputDimension<DifferentialForm::Continuous>()>
  eval_surf_curl(
      const SuperSpace<LinOp> &super_space,
      const int polynomial_degree_plus_one_squared,
      const ElementTreeNode &element, const SurfacePoint &p,
      const Eigen::Matrix<
          Scalar, Eigen::Dynamic,
          getFunctionSpaceVectorDimension<DifferentialForm::Continuous>()>
          &coeff) const {
    double kappa = p.segment<3>(6).cross(p.segment<3>(9)).norm();
    Eigen::MatrixXd s_curl(3, polynomial_degree_plus_one_squared);
    s_curl =
        (1.0 / kappa) *
        (-p.segment<3>(6) * super_space.basisDy(p.segment<2>(0)).transpose() +
         p.segment<3>(9) * super_space.basisDx(p.segment<2>(0)).transpose());
    return s_curl * coeff / (element.get_h() * element.get_h());
  };
};

// discontinuous
template <typename Scalar, typename LinOp>
struct DiscreteFunctionEval<Scalar, DifferentialForm::Discontinuous, LinOp> {
  Eigen::Matrix<
      Scalar,
      getFunctionSpaceOutputDimension<DifferentialForm::Discontinuous>(), 1>
  eval(const SuperSpace<LinOp> &super_space,
       const int polynomial_degree_plus_one_squared,
       const ElementTreeNode &element, const SurfacePoint &p,
       const Eigen::Matrix<
           Scalar, Eigen::Dynamic,
           getFunctionSpaceVectorDimension<DifferentialForm::Discontinuous>()>
           &coeff) const {
    auto s = p.segment<2>(0);
    return coeff.transpose() * super_space.basis(s) / element.get_h();
  };
  Eigen::Matrix<
      Scalar, 3,
      getFunctionSpaceOutputDimension<DifferentialForm::Discontinuous>()>
  eval_surf_grad(
      const SuperSpace<LinOp> &super_space,
      const int polynomial_degree_plus_one_squared,
      const ElementTreeNode &element, const SurfacePoint &p,
      const Eigen::Matrix<
          Scalar, Eigen::Dynamic,
          getFunctionSpaceVectorDimension<DifferentialForm::Discontinuous>()>
          &coeff) const {
    auto s = p.segment<2>(0);
    auto x_f_dx = p.segment<3>(6);
    auto x_f_dy = p.segment<3>(9);
    Eigen::Matrix<double, 3, 2> s_j;
    s_j.block(0, 0, 3, 1) = x_f_dx;
    s_j.block(0, 1, 3, 1) = x_f_dy;
    Eigen::MatrixXd s_drho(2, polynomial_degree_plus_one_squared);
    s_drho.block(0, 0, 1, polynomial_degree_plus_one_squared) =
        super_space.basisDx(s).transpose();
    s_drho.block(1, 0, 1, polynomial_degree_plus_one_squared) =
        super_space.basisDy(s).transpose();
    Eigen::Matrix<double, 3, Eigen::Dynamic> s_g;
    s_g = s_j * (s_j.transpose() * s_j).inverse() * s_drho;
    return s_g * coeff / (element.get_h() * element.get_h());
  };
  Eigen::Matrix<
      Scalar, 3,
      getFunctionSpaceOutputDimension<DifferentialForm::Discontinuous>()>
  eval_surf_curl(
      const SuperSpace<LinOp> &super_space,
      const int polynomial_degree_plus_one_squared,
      const ElementTreeNode &element, const SurfacePoint &p,
      const Eigen::Matrix<
          Scalar, Eigen::Dynamic,
          getFunctionSpaceVectorDimension<DifferentialForm::Discontinuous>()>
          &coeff) const {
    double kappa = p.segment<3>(6).cross(p.segment<3>(9)).norm();
    Eigen::MatrixXd s_curl(3, polynomial_degree_plus_one_squared);
    s_curl =
        (1.0 / kappa) *
        (-p.segment<3>(6) * super_space.basisDy(p.segment<2>(0)).transpose() +
         p.segment<3>(9) * super_space.basisDx(p.segment<2>(0)).transpose());
    return s_curl * coeff / (element.get_h() * element.get_h());
  };
};
}  // namespace Bembel
#endif
