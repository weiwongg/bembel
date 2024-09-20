// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_ANSATZSPACE_DEFORMATIONFIELD_H_
#define BEMBEL_ANSATZSPACE_DEFORMATIONFIELD_H_

namespace Bembel {
/**
 *  \ingroup AnsatzSpace
 *  \brief The DeformationField provides means to evaluate coefficient vectors
 * as functions on the geometry.
 */
template <typename Derived>
class DeformationField : public FunctionEvaluatorBase<Derived, 3> {
 public:
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  using FunctionEvaluatorBase<Derived, 3>::FunctionEvaluatorBase;
  //////////////////////////////////////////////////////////////////////////////
  //    evaluators
  //////////////////////////////////////////////////////////////////////////////
  //
  //////////////////////////////////////////////////////////////////////////////
  //    setters
  //////////////////////////////////////////////////////////////////////////////
  //
  //////////////////////////////////////////////////////////////////////////////
  //    getter
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::MatrixXd> get_patch_vector() {
    // go on each patch from hierarchical structure to rowwise structure on
    // patch
    Eigen::Matrix<double, Eigen::Dynamic, 3> sortedfun(this->fun_.rows(),
                                                       this->fun_.cols());
    for (int i = 0; i < this->ansatz_space_.get_number_of_elements(); ++i)
      sortedfun.block(i * this->polynomial_degree_plus_one_squared_, 0,
                      this->polynomial_degree_plus_one_squared_, 3) =
          this->fun_.block(this->reordering_vector_[i] *
                               this->polynomial_degree_plus_one_squared_,
                           0, this->polynomial_degree_plus_one_squared_, 3);

    // build patch vector
    std::vector<Eigen::MatrixXd> patch_vector;
    int n = 1 << this->ansatz_space_.get_refinement_level();
    for (int i = 0;
         i < this->ansatz_space_.get_superspace().get_geometry().size(); ++i)
      patch_vector.push_back(
          n * sortedfun.block(
                  i * n * n * this->polynomial_degree_plus_one_squared_, 0,
                  n * n * this->polynomial_degree_plus_one_squared_, 3));

    return patch_vector;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    private member variables
  //////////////////////////////////////////////////////////////////////////////
 private:
  //
};

}  // namespace Bembel
#endif
