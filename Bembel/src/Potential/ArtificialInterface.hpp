#ifndef ARTIFICIALINTERFACE_H_
#define ARTIFICIALINTERFACE_H_

namespace Bembel {

/**
 *  \ingroup Potential
 *  \brief Accelerate potential evaluation with an artificial interface
 */
template <class SLO, class SLP, class DLO, class DLP>
class ArtificialInterface {
 public:
  // constructor
  ArtificialInterface(const Geometry &geometry, const int refinement_lvl,
                      const int polynomial_degree) {
    init_ArtificialInterface(geometry, refinement_lvl, polynomial_degree);
  }

  /**
   * \todo This is redundantly implemented, see also H2Multipole
   */
  // generate Chebychev interpolation points
  Eigen::VectorXd compute1DInterpolationPoints() const {
    int n = polynomial_degree_ + 1;
    // set up interpolation points in [0,nx]
    Eigen::VectorXd ret = (BEMBEL_PI / polynomial_degree_) * Eigen::VectorXd::LinSpaced(n, 0., polynomial_degree_);
    ret = 0.5 * (1 - ret.array().cos());
    return ret;
  }

  // generate evaluation points on artificial interface
  Eigen::Matrix<double, -1, 3> compute_gridpoints() const {
    auto cheb = compute1DInterpolationPoints();
    int dim_cheb = cheb.rows();
    const ElementTree &et =
        ansatz_space_slo_.get_superspace().get_mesh().get_element_tree();
    const unsigned int number_of_elements = et.get_number_of_elements();
    Eigen::MatrixXd ret =
        Eigen::MatrixXd(number_of_elements * dim_cheb * dim_cheb, 3);
    int i = 0;
    for (auto element = et.cpbegin(); element != et.cpend(); ++element) {
      for (auto j = 0; j < dim_cheb; ++j) {
        for (auto k = 0; k < dim_cheb; ++k) {
          SurfacePoint qp;
          ansatz_space_slo_.get_superspace().map2surface(
              *element, Eigen::Vector2d(cheb(j), cheb(k)), 1.0, &qp);
          // get points on geometry
          ret.row((i * dim_cheb + j) * dim_cheb + k) =
              qp.segment<3>(3).transpose();
        }
      }
      ++i;
    }
    return ret;
  }

  void compute_sys() {
    auto cheb = compute1DInterpolationPoints();
    int dim_cheb = cheb.rows();
    int polynomial_degree_plus_one = polynomial_degree_ + 1;
    int polynomial_degree_plus_one_squared =
        polynomial_degree_plus_one * polynomial_degree_plus_one;
    auto super_space = ansatz_space_slo_.get_superspace();
    const ElementTree &et = super_space.get_mesh().get_element_tree();
    const unsigned int number_of_elements = et.get_number_of_elements();
    Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, Eigen::Dynamic,
                  Eigen::Dynamic>
        element_sys(dim_cheb * dim_cheb, polynomial_degree_plus_one_squared);
    // discontinuous interpolation matrix
    for (auto j = 0; j < dim_cheb; ++j) {
      for (auto k = 0; k < dim_cheb; ++k) {
        auto buf = super_space.basis(Eigen::Vector2d(cheb(j), cheb(k)));
        element_sys.row(j * dim_cheb + k) = buf.transpose();
      }
    }
    Eigen::SparseMatrix<typename LinearOperatorTraits<SLO>::Scalar> I(
        number_of_elements, number_of_elements);
    I.setIdentity();
    sys_ = Eigen::kroneckerProduct(I, element_sys) *
           ansatz_space_slo_.get_transformation_matrix();
    // assert(LU_.rank() == geometry_.get_geometry().size() *
    // polynomial_degree_plus_one_squared &&
    //"Interpolation matrix must have full rank");
  }

  // init relevant stuff
  void init_ArtificialInterface(const Geometry &geometry,
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

    // compute gridpoints
    gridpoints_ = compute_gridpoints();

    // build potentials
    single_layer_potential_ = DiscretePotential<SLP, SLO>(ansatz_space_slo_);
    double_layer_potential_ = DiscretePotential<DLP, DLO>(ansatz_space_dlo_);

    // compute vandermonde matrix
    compute_sys();

    return;
  }

  // convert potential gradient to values of Neumann data
  Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 1>
  potentialGradientToNeumann(
      const Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 3>
          &gradient) {
    Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 1> neumann(
        gradient.rows());
    auto cheb = compute1DInterpolationPoints();
    int dim_cheb = cheb.rows();
    const ElementTree &et =
        ansatz_space_slo_.get_superspace().get_mesh().get_element_tree();
    const unsigned int number_of_elements = et.get_number_of_elements();
    Eigen::MatrixXd ret =
        Eigen::MatrixXd(3, number_of_elements * dim_cheb * dim_cheb);
    int i = 0;
    for (auto element = et.cpbegin(); element != et.cpend(); ++element) {
      for (auto j = 0; j < dim_cheb; ++j) {
        for (auto k = 0; k < dim_cheb; ++k) {
          SurfacePoint qp;
          ansatz_space_slo_.get_superspace().map2surface(
              *element, Eigen::Vector2d(cheb(j), cheb(k)), 1.0, &qp);
          auto x_f_dx = qp.segment<3>(6);
          auto x_f_dy = qp.segment<3>(9);
          // compute (unnormalized) surface normal from tangential derivatives
          auto normal = (x_f_dx.cross(x_f_dy)).normalized();
          // get points on geometry
          neumann((i * dim_cheb + j) * dim_cheb + k) =
              normal.dot(gradient.row((i * dim_cheb + j) * dim_cheb + k));
        }
      }
      ++i;
    }
    return neumann;
  }

  // set boundary data on artificial interface from point evaluations
  void setBoundaryDataWithInterpolation(
      const Eigen::Matrix<typename LinearOperatorTraits<DLO>::Scalar, -1, 1>
          &dirichlet,
      const Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 1>
          &neumann) {
    const double h = 1.0 / (1 << refinement_lvl_);
    Eigen::LeastSquaresConjugateGradient<
        Eigen::SparseMatrix<typename LinearOperatorTraits<SLO>::Scalar>>
        lscg(sys_);
    //lscg.setTolerance(1e-10);
    setBoundaryData(h * lscg.solve(dirichlet), h * lscg.solve(neumann));
  }

  // set boundary data on artificial interface
  void setBoundaryData(
      const Eigen::Matrix<typename LinearOperatorTraits<DLO>::Scalar, -1, 1>
          &dirichlet,
      const Eigen::Matrix<typename LinearOperatorTraits<SLO>::Scalar, -1, 1>
          &neumann) {
    single_layer_potential_.set_cauchy_data(neumann);
    double_layer_potential_.set_cauchy_data(dirichlet);
  }

  // evaluate solution on gridpoints away from artificial interface
  Eigen::Matrix<typename PotentialTraits<SLP>::Scalar, -1, 1> evaluate(
      const Eigen::Matrix<double, -1, 3> &gridpoints) {
    return -single_layer_potential_.evaluate(gridpoints) +
           double_layer_potential_.evaluate(gridpoints);
  }

  Eigen::Matrix<typename LinearOperatorTraits<DLO>::Scalar, -1, 1> interpolate(
      const Eigen::Matrix<typename LinearOperatorTraits<DLO>::Scalar, -1, 1>
          &data) {
    const double h = 1.0 / (1 << refinement_lvl_);
    Eigen::LeastSquaresConjugateGradient<
        Eigen::SparseMatrix<typename LinearOperatorTraits<SLO>::Scalar>>
        lscg(sys_);
    //lscg.setTolerance(1e-10);
    return h * lscg.solve(data);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Getters
  /////////////////////////////////////////////////////////////////////////////
  const Eigen::Matrix<double, -1, 3> &get_gridpoints() const {
    return gridpoints_;
  }
  Eigen::Matrix<double, -1, 3> &get_gridpoints() { return gridpoints_; }
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
  Eigen::SparseMatrix<typename LinearOperatorTraits<SLO>::Scalar> sys_;
  Eigen::Matrix<double, -1, 3> gridpoints_;
  AnsatzSpace<SLO> ansatz_space_slo_;
  AnsatzSpace<DLO> ansatz_space_dlo_;
  DiscretePotential<SLP, SLO> single_layer_potential_;
  DiscretePotential<DLP, DLO> double_layer_potential_;
};

}  // namespace Bembel

#endif  // ARTIFICIALINTERFACE_H_
