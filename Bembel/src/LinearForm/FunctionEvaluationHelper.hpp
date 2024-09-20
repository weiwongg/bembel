// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEARFORM_FUNCTIONEVALUATIONHELPER_H_
#define BEMBEL_LINEARFORM_FUNCTIONEVALUATIONHELPER_H_

namespace Bembel {

template <typename Scalar, typename Function>
struct FunctionEvaluationHelper {};

template <typename Scalar>
struct FunctionEvaluationHelper<Scalar,
                                std::function<Scalar(Eigen::Vector3d)>> {
  static Scalar evaluate(const SurfacePoint &p,
                         std::function<Scalar(Eigen::Vector3d)> fun) {
    auto x_f = p.segment<3>(3);
    return fun(x_f);
  };
};

template <typename Scalar>
struct FunctionEvaluationHelper<Scalar,
                                std::function<Scalar(const SurfacePoint &p)>> {
  static Scalar evaluate(
      const SurfacePoint &p,
      std::function<Scalar(const SurfacePoint &p)> disc_fun) {
    return disc_fun(p);
  };
};

}  // namespace Bembel
#endif
