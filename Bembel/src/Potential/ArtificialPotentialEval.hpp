#ifndef ARTIFICIALPOTENTIALEVAL_H_
#define ARTIFICIALPOTENTIALEVAL_H_

namespace Bembel {

/**
 *  \ingroup Potential
 *  \brief Accelerate potential evaluation with an artificial interface
 */
template <class SLO, class SLP, class DLO, class DLP>
class ArtificialPotentialEval {
 public:
  // constructor
  ArtificialPotentialEval(const Geometry &geometry, const int refinement_lvl,
                          const int polynomial_degree) {
    init_ArtificialPotentialEval(geometry, refinement_lvl, polynomial_degree);
  }
  // init relevant stuff
  void init_ArtificialPotentialEval(const Geometry &geometry,
                                    const int refinement_lvl,
                                    const int polynomial_degree) {
    // assign variables
    geometry_ = geometry;
    refinement_lvl_ = refinement_lvl;
    polynomial_degree_ = polynomial_degree;

    // build ansatz ansatz spaces
    ansatz_space_slo_ =
        AnsatzSpace<SLO>(geometry_, refinement_lvl_, polynomial_degree_);
    ansatz_space_dlo_ =
        AnsatzSpace<DLO>(geometry_, refinement_lvl_, polynomial_degree_);

    // build potentials
    single_layer_potential_ = DiscretePotential<SLP, SLO>(ansatz_space_slo_);
    double_layer_potential_ = DiscretePotential<DLP, DLO>(ansatz_space_dlo_);

    return;
  }

  // set boundary data on artificial interface
  void setBoundaryData(
      const Eigen::Matrix<typename LinearOperatorTraits<DLO>::Scalar, -1, 1>
          &dirichlet,
      const Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 1>
          &neumann,
      typename LinearOperatorTraits<DLO>::Scalar wavenumber) {
    single_layer_potential_.set_cauchy_data(neumann);
    single_layer_potential_.get_potential().set_wavenumber(wavenumber);
    double_layer_potential_.set_cauchy_data(dirichlet);
    double_layer_potential_.get_potential().set_wavenumber(wavenumber);
  }

  // evaluate solution on gridpoints away from artificial interface
  Eigen::Matrix<typename PotentialTraits<SLP>::Scalar, -1, 1> evaluate(
      const Eigen::Matrix<double, -1, 3> &gridpoints) {
    return -single_layer_potential_.evaluate(gridpoints) +
           double_layer_potential_.evaluate(gridpoints);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Getters
  /////////////////////////////////////////////////////////////////////////////
  DiscretePotential<SLP, SLO> &get_single_layer_potential() {
    return single_layer_potential_;
  }
  DiscretePotential<DLP, DLO> &get_double_layer_potential() {
    return double_layer_potential_;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Private members
  /////////////////////////////////////////////////////////////////////////////
 private:
  Geometry geometry_;
  int refinement_lvl_;
  int polynomial_degree_;
  AnsatzSpace<SLO> ansatz_space_slo_;
  AnsatzSpace<DLO> ansatz_space_dlo_;
  DiscretePotential<SLP, SLO> single_layer_potential_;
  DiscretePotential<DLP, DLO> double_layer_potential_;
};

}  // namespace Bembel

#endif  // ARTIFICIALPOTENTIALEVAL_H_
