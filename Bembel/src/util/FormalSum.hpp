// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_UTIL_FORMALSUM_H_
#define BEMBEL_UTIL_FORMALSUM_H_

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace Eigen {
/// forward definition of the FormalSum Class in order to define traits
template <typename Derived, typename otherDerived>
class FormalSum;

/// inherit the traits from the Eigen::SparseMatrix class
namespace internal {
template <typename Derived, typename otherDerived>
struct traits<FormalSum<Derived, otherDerived>>
    : public internal::traits<SparseMatrix<typename Derived::Scalar>> {};
}  // namespace internal

/** \ingroup util
 *  \brief   this class realises a formal sum expression of to derived
 *           EigenBase objects in order to facilitate their multiplication
 *           by a vector: (S+v*v^T)*x=S*x+v*(v^T*x). This way, the formal sum
 *can be used in the Eigen matrixfree solvers. A is a sparse matrix and V is a
 *column vector.
 *
 *           We emphasize however, that the use of this class is highly unsafe,
 *           as it only stores const references to S and v, hence not extending
 *           the lifetime of the involved objects till the end of the scope of
 *           the formal sum. Consequently, it is very likely to create dangeling
 *           references if this is not used properly. However, since the
 *           desired functionality is highly desirable, a more sophisticated
 *           implementation will follow at some point.
 **/

template <typename Derived, typename otherDerived>
class FormalSum : public EigenBase<FormalSum<Derived, otherDerived>> {
 public:
  //////////////////////////////////////////////////////////////////////////////
  /// Eigen related things
  //////////////////////////////////////////////////////////////////////////////
  // Required typedefs, constants and so on.
  typedef typename Derived::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Index StorageIndex;
  enum {
    ColsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    IsRowMajor = false,
    Flags = NestByRefBit
  };
  // Minimum specialisation of EigenBase methods
  Index rows() const { return summand1_.rows(); }
  Index cols() const { return summand1_.cols(); }
  const Derived& get_summand1() const { return summand1_; }
  const otherDerived& get_summand2() const { return summand2_; }
  // Definition of the matrix multiplication
  template <typename Rhs>
  Product<FormalSum, Rhs, AliasFreeProduct> operator*(
      const MatrixBase<Rhs>& x) const {
    return Product<FormalSum, Rhs, AliasFreeProduct>(*this, x.derived());
  }
  //////////////////////////////////////////////////////////////////////////////
  /// constructors
  //////////////////////////////////////////////////////////////////////////////
  FormalSum() {}
  FormalSum(const EigenBase<Derived>& A, const EigenBase<otherDerived>& B)
      : summand1_(A.derived()), summand2_(B.derived()) {}
  //////////////////////////////////////////////////////////////////////////////
  /// private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  // we declare functionality which has not been implemented (yet)
  // to be private
  FormalSum(const FormalSum<Derived, otherDerived>& H);
  FormalSum(FormalSum<Derived, otherDerived>&& H);
  FormalSum& operator=(const FormalSum<Derived, otherDerived>& H);
  FormalSum& operator=(FormalSum<Derived, otherDerived>&& H);
  const Derived& summand1_;
  const otherDerived& summand2_;
};

/**
 * \brief Implementation of FormalSum * Eigen::DenseVector through a
 * specialization of internal::generic_product_impl
 */
namespace internal {
template <typename Rhs, typename Derived, typename otherDerived>
struct generic_product_impl<FormalSum<Derived, otherDerived>, Rhs, SparseShape,
                            DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          FormalSum<Derived, otherDerived>, Rhs,
          generic_product_impl<FormalSum<Derived, otherDerived>, Rhs>> {
  typedef typename Derived::Scalar Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst,
                            const FormalSum<Derived, otherDerived>& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    dst += alpha * (lhs.get_summand1() * rhs + lhs.get_summand2() * rhs);
  }
};
}  // namespace internal
////////////////////////////////////////////////////////////////////////////////
}  // namespace Eigen
#endif
