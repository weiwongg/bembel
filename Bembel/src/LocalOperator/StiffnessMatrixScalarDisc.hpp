// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LOCALOPERATOR_STIFFNESSMATRIXSCALARDISC_H_
#define BEMBEL_LOCALOPERATOR_STIFFNESSMATRIXSCALARDISC_H_

namespace Bembel {
// forward declaration of class StiffnessMatrixScalarDisc in order to define
// traits
class StiffnessMatrixScalarDisc;

template <>
struct LinearOperatorTraits<StiffnessMatrixScalarDisc> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum {
    OperatorOrder = 0,
    Form = DifferentialForm::Discontinuous,
    NumberOfFMMComponents = 0
  };
};

class StiffnessMatrixScalarDisc
    : public LaplaceBeltramiOperatorBase<StiffnessMatrixScalarDisc> {};

}  // namespace Bembel
#endif
