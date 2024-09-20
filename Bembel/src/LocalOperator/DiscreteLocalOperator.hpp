// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LOCALOPERATOR_DISCRETELOCALOPERATOR_H_
#define BEMBEL_LOCALOPERATOR_DISCRETELOCALOPERATOR_H_

namespace Bembel {
/**
 *  \ingroup LocalOperator
 *  \brief Helper struct which mimics the DiscreteOperatorComputer struct to
 *         assemble sparse matrices of local operators
 */
template <typename Derived>
struct DiscreteLocalOperatorComputer {
  DiscreteLocalOperatorComputer(){};
  static void compute(
      Eigen::SparseMatrix<typename LinearOperatorTraits<Derived>::Scalar>
          *disc_op,
      const Derived &lin_op, const AnsatzSpace<Derived> &ansatz_space) {
    // Extract numbers
    auto super_space = ansatz_space.get_superspace();
    auto element_tree = super_space.get_mesh().get_element_tree();
    auto number_of_elements = element_tree.get_number_of_elements();
    const auto vector_dimension =
        getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>();
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // Quadrature
    GaussSquare<Constants::maximum_quadrature_degree> GS;
    auto ffield_deg = lin_op.get_FarfieldQuadratureDegree(polynomial_degree);
    auto Q = GS[ffield_deg];

    // Triplets
    typedef Eigen::Triplet<typename LinearOperatorTraits<Derived>::Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(polynomial_degree_plus_one_squared *
                        polynomial_degree_plus_one_squared *
                        number_of_elements * vector_dimension);

    // Iterate over elements
    for (auto element = element_tree.cpbegin(); element != element_tree.cpend();
         ++element) {
      Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                    Eigen::Dynamic, Eigen::Dynamic>
          intval(vector_dimension * polynomial_degree_plus_one_squared,
                 vector_dimension * polynomial_degree_plus_one_squared);
      intval.setZero();
      // Iterate over quadrature points
      for (auto i = 0; i < Q.w_.size(); ++i) {
        SurfacePoint qp;
        super_space.map2surface(*element, Q.xi_.col(i), Q.w_(i), &qp);
        lin_op.evaluateIntegrand(super_space, qp, qp, &intval);
      }
      // Transform local element matrices to triplets
      for (auto i = 0; i < vector_dimension; ++i)
        for (auto j = 0; j < vector_dimension; ++j)
          for (auto ii = 0; ii < polynomial_degree_plus_one_squared; ++ii)
            for (auto jj = 0; jj < polynomial_degree_plus_one_squared; ++jj)
              tripletList.push_back(
                  T(polynomial_degree_plus_one_squared *
                            (j * number_of_elements + element->id_) +
                        jj,
                    polynomial_degree_plus_one_squared *
                            (i * number_of_elements + element->id_) +
                        ii,
                    intval(j * polynomial_degree_plus_one_squared + jj,
                           i * polynomial_degree_plus_one_squared + ii)));
    }
    disc_op->resize(vector_dimension * polynomial_degree_plus_one_squared *
                        number_of_elements,
                    vector_dimension * polynomial_degree_plus_one_squared *
                        number_of_elements);
    disc_op->setFromTriplets(tripletList.begin(), tripletList.end());
    auto projector = ansatz_space.get_transformation_matrix();
    disc_op[0] = projector.transpose() * (disc_op[0] * projector);
    return;
  }
};

/**
 *  \ingroup LocalOperator
 *  \brief DiscreteLocalOperator
 *         Specialization of the DiscreteOperator class for sparse matrices
 *         and an intgrator for local operators
 **/
template <typename Derived>
using DiscreteLocalOperator = DiscreteOperator<
    Eigen::SparseMatrix<typename LinearOperatorTraits<Derived>::Scalar>,
    Derived, DiscreteLocalOperatorComputer<Derived> >;

}  // namespace Bembel
#endif
