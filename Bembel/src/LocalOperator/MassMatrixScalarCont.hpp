// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LOCALOPERATOR_MASSMATRIXSCALARCONT_H_
#define BEMBEL_LOCALOPERATOR_MASSMATRIXSCALARCONT_H_

namespace Bembel {
// forward declaration of class MassMatrixScalarCont in order to define
// traits
class MassMatrixScalarCont;

template <>
struct LinearOperatorTraits<MassMatrixScalarCont> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum {
    OperatorOrder = 0,
    Form = DifferentialForm::Continuous,
    NumberOfFMMComponents = 0
  };
};

class MassMatrixScalarCont : public IdentityOperatorBase<MassMatrixScalarCont> {
};

}  // namespace Bembel
#endif
