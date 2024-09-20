// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_DISCRETEFUNCTION_H_
#define BEMBEL_DISCRETEFUNCTION_H_

namespace Bembel {
/**
 *  \ingroup AnsatzSpace
 *  \brief The DiscreteFunction provides means to evaluate coefficient vectors
 * as functions on the geometry.
 */
template <typename Derived> class DiscreteFunction {
public:
  typedef typename LinearOperatorTraits<Derived>::Scalar Scalar;
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  DiscreteFunction() {}
  DiscreteFunction(const DiscreteFunction &other) = default;
  DiscreteFunction(DiscreteFunction &&other) = default;
  DiscreteFunction &operator=(DiscreteFunction other) {
    ansatz_space_ = other.ansatz_space_;
    polynomial_degree_ = other.polynomial_degree_;
    polynomial_degree_plus_one_squared_ =
        other.polynomial_degree_plus_one_squared_;
    level_ = other.level_;
    global_fun_ = other.global_fun_;
    fun_ = other.fun_;
    return *this;
  }
  DiscreteFunction(const AnsatzSpace<Derived> &ansatz_space) {
    init_DiscreteFunction(ansatz_space);
    return;
  }
  DiscreteFunction(const AnsatzSpace<Derived> &ansatz_space,
                   const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &fun) {
    init_DiscreteFunction(ansatz_space, fun);
    return;
  }
  DiscreteFunction(const AnsatzSpace<Derived> &ansatz_space,
                   std::function<Scalar(const Eigen::Vector3d &)> func) {
    init_DiscreteFunction(ansatz_space, func);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //    init_Ansatzspace
  //////////////////////////////////////////////////////////////////////////////
  void init_DiscreteFunction(const AnsatzSpace<Derived> &ansatz_space) {
    ansatz_space_ = ansatz_space;
    polynomial_degree_ = ansatz_space_.get_polynomial_degree();
    polynomial_degree_plus_one_squared_ =
        (polynomial_degree_ + 1) * (polynomial_degree_ + 1);
    level_ = ansatz_space_.get_refinement_level();
    reordering_vector_ = ansatz_space_.get_superspace()
                             .get_mesh()
                             .get_element_tree()
                             .computeReorderingVector();
    mass_initialized = false;
    stiffness_initialized = false;
    return;
  }
  void init_DiscreteFunction(
      const AnsatzSpace<Derived> &ansatz_space,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &global_fun) {
    ansatz_space_ = ansatz_space;
    polynomial_degree_ = ansatz_space_.get_polynomial_degree();
    polynomial_degree_plus_one_squared_ =
        (polynomial_degree_ + 1) * (polynomial_degree_ + 1);
    level_ = ansatz_space_.get_refinement_level();
    reordering_vector_ = ansatz_space_.get_superspace()
                             .get_mesh()
                             .get_element_tree()
                             .computeReorderingVector();
    mass_initialized = false;
    stiffness_initialized = false;
    set_function(global_fun);
    return;
  }
  void
  init_DiscreteFunction(const AnsatzSpace<Derived> &ansatz_space,
                        std::function<Scalar(const Eigen::Vector3d &)> func) {
    ansatz_space_ = ansatz_space;
    polynomial_degree_ = ansatz_space_.get_polynomial_degree();
    polynomial_degree_plus_one_squared_ =
        (polynomial_degree_ + 1) * (polynomial_degree_ + 1);
    level_ = ansatz_space_.get_refinement_level();
    reordering_vector_ = ansatz_space_.get_superspace()
                             .get_mesh()
                             .get_element_tree()
                             .computeReorderingVector();
    mass_initialized = false;
    stiffness_initialized = false;
    set_function(func);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //    operators reload
  //////////////////////////////////////////////////////////////////////////////
  DiscreteFunction operator-(DiscreteFunction &other) {
    return DiscreteFunction(ansatz_space_, global_fun_ - other.global_fun_);
  }
  DiscreteFunction operator+(DiscreteFunction &other) {
    return DiscreteFunction(ansatz_space_, global_fun_ + other.global_fun_);
  }
  DiscreteFunction operator*(double scale) {
    return DiscreteFunction(ansatz_space_, scale * global_fun_);
  }
  //////////////////////////////////////////////////////////////////////////////
  //    evaluators
  //////////////////////////////////////////////////////////////////////////////
  // evaluate function value on patch with patch ID and the coordinates on the
  // unit square.
  Eigen::Matrix<
      Scalar,
      getFunctionSpaceOutputDimension<LinearOperatorTraits<Derived>::Form>(), 1>
  evaluateOnPatch(int patch, const Eigen::Vector2d &ref_point) const {
    const int elements_per_direction =
        (1 << ansatz_space_.get_refinement_level());
    const int elements_per_patch =
        elements_per_direction * elements_per_direction;
    const double h = 1. / ((double)(elements_per_direction));
    const int x_idx_ = std::floor(ref_point(0) / h);
    const int y_idx_ = std::floor(ref_point(1) / h);
    const int x_idx = std::min(std::max(x_idx_, 0), elements_per_direction - 1);
    const int y_idx = std::min(std::max(y_idx_, 0), elements_per_direction - 1);
    const int tp_idx =
        x_idx + elements_per_direction * y_idx + patch * elements_per_patch;
    const int et_idx = reordering_vector_[tp_idx];
    const ElementTreeNode &element = *(
        ansatz_space_.get_superspace().get_mesh().get_element_tree().cpbegin() +
        et_idx);

    SurfacePoint sp;
    ansatz_space_.get_superspace().get_geometry()[patch].updateSurfacePoint(
        &sp, ref_point, 1, element.mapToReferenceElement(ref_point));
    return evaluate(element, sp);
  }
  // evaluate function value on element with element ID and the surfacepoint.
  Eigen::Matrix<
      Scalar,
      getFunctionSpaceOutputDimension<LinearOperatorTraits<Derived>::Form>(), 1>
  evaluate(const ElementTreeNode &element, const SurfacePoint &p) const {
    return eval_.eval(
        ansatz_space_.get_superspace(), polynomial_degree_plus_one_squared_,
        element, p,
        fun_.block(polynomial_degree_plus_one_squared_ * element.id_, 0,
                   polynomial_degree_plus_one_squared_,
                   getFunctionSpaceVectorDimension<
                       LinearOperatorTraits<Derived>::Form>()));
  }
  // evaluate function value on the surfacepoint.
  Eigen::Matrix<
      Scalar,
      getFunctionSpaceOutputDimension<LinearOperatorTraits<Derived>::Form>(), 1>
  evaluate(const SurfacePoint &p) const {
    int element_id = (int)p(12);
    return evaluate(*(ansatz_space_.get_superspace()
                          .get_mesh()
                          .get_element_tree()
                          .cpbegin() +
                      element_id),
                    p);
  }
  // evaluate surface gradient on element with element ID and the surfacepoint.
  Eigen::Matrix<
      Scalar, 3,
      getFunctionSpaceOutputDimension<LinearOperatorTraits<Derived>::Form>()>
  evaluateSurfaceGradient(const ElementTreeNode &element,
                          const SurfacePoint &p) const {
    return eval_.eval_surf_grad(
        ansatz_space_.get_superspace(), polynomial_degree_plus_one_squared_,
        element, p,
        fun_.block(polynomial_degree_plus_one_squared_ * element.id_, 0,
                   polynomial_degree_plus_one_squared_,
                   getFunctionSpaceVectorDimension<
                       LinearOperatorTraits<Derived>::Form>()));
  }
  // evaluate surface curl on element with element ID and the surfacepoint.
  Eigen::Matrix<
      Scalar, 3,
      getFunctionSpaceOutputDimension<LinearOperatorTraits<Derived>::Form>()>
  evaluateSurfaceCurl(const ElementTreeNode &element,
                      const SurfacePoint &p) const {
    return eval_.eval_surf_curl(
        ansatz_space_.get_superspace(), polynomial_degree_plus_one_squared_,
        element, p,
        fun_.block(polynomial_degree_plus_one_squared_ * element.id_, 0,
                   polynomial_degree_plus_one_squared_,
                   getFunctionSpaceVectorDimension<
                       LinearOperatorTraits<Derived>::Form>()));
  }
  //////////////////////////////////////////////////////////////////////////////
  //    setters
  //////////////////////////////////////////////////////////////////////////////
  void set_function(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fun) {
    global_fun_ = fun;
    const auto vec_dim =
        getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>();
    auto longfun = (ansatz_space_.get_transformation_matrix() * fun).eval();
    fun_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, vec_dim>>(
        longfun.data(), longfun.rows() / vec_dim, vec_dim);
  }
  void set_function(std::function<Scalar(const Eigen::Vector3d &)> func) {
    if (!mass_initialized)
      compute_mass();
    global_fun_ = init_.interpolation(ansatz_space_, level_, polynomial_degree_,
                                      mass_, func);

    const auto vec_dim =
        getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>();
    auto longfun =
        (ansatz_space_.get_transformation_matrix() * global_fun_).eval();
    fun_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, vec_dim>>(
        longfun.data(), longfun.rows() / vec_dim, vec_dim);
  }
  //////////////////////////////////////////////////////////////////////////////
  //    visualization
  //////////////////////////////////////////////////////////////////////////////
  void plot(const std::string &name = "plot", int resolution_level = 6, const std::string &vname = "f_val") {
    VTKSurfaceExport writer(ansatz_space_.get_geometry(), resolution_level);
    std::function<double(int, const Eigen::Vector2d &)> f_val =
        [&](int patch_number, const Eigen::Vector2d &reference_domain_point) {
          auto retval = evaluateOnPatch(patch_number, reference_domain_point);
          return double(retval(0, 0));
        };
    writer.addDataSet(vname, f_val);
    writer.writeToFile(name + ".vtp");
  }

  //////////////////////////////////////////////////////////////////////////////
  //   mass and stiffness matrices
  //////////////////////////////////////////////////////////////////////////////
  void compute_mass() {
    mass_ = init_.get_mass(ansatz_space_, level_, polynomial_degree_);
    mass_initialized = true;
  }
  void compute_stiffness() {
    stiffness_ = init_.get_stiffness(ansatz_space_, level_, polynomial_degree_);
    stiffness_initialized = true;
  }

  void compute_metric() {
    compute_mass();
    compute_stiffness();
  }

  //////////////////////////////////////////////////////////////////////////////
  //    norm
  //////////////////////////////////////////////////////////////////////////////
  double norm_l2() {
    if (!mass_initialized)
      compute_mass();
    auto norm_squared = global_fun_.transpose() * mass_ * global_fun_;
    return sqrt(norm_squared(0));
  }
  double norm_h1() {
    if (!mass_initialized)
      compute_mass();
    if (!stiffness_initialized)
      compute_stiffness();
    auto norm_squared = global_fun_.transpose() * mass_ * global_fun_ +
                        global_fun_.transpose() * stiffness_ * global_fun_;
    return sqrt(norm_squared(0));
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  //    prolongation and restriction one step
  ////////////////////////////////////////////////////////////////////////////////////////////
  DiscreteFunction prolongation() {
    AnsatzSpace<Derived> refined_ansatz_space(ansatz_space_.get_geometry(),
                                              level_ + 1, polynomial_degree_);
    DiscreteFunction<Derived> fine_df(refined_ansatz_space);

    auto fine_global_fun_ = ansatz_space_.prolongate_one_step(global_fun_);
    fine_df.set_function(fine_global_fun_);
    return fine_df;
  }

  //////////////////////////////////////////////////////////////////////////////
  //    getters
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>()>
  get_global_fun() const {
    return global_fun_;
  }
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>()>
  get_fun() const {
    return fun_;
  }
  const Eigen::SparseMatrix<Scalar> &get_mass_matrix() const { return mass_; }
  const Eigen::SparseMatrix<Scalar> &get_stiffness_matrix() const {
    return stiffness_;
  }

  //////////////////////////////////////////////////////////////////////////////
  //    private member variables
  //////////////////////////////////////////////////////////////////////////////
private:
  std::vector<int> reordering_vector_;
  AnsatzSpace<Derived> ansatz_space_;
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>()>
      global_fun_;
  Eigen::Matrix<
      Scalar, Eigen::Dynamic,
      getFunctionSpaceVectorDimension<LinearOperatorTraits<Derived>::Form>()>
      fun_;
  int polynomial_degree_;
  int polynomial_degree_plus_one_squared_;
  int level_;
  DiscreteFunctionEval<Scalar, LinearOperatorTraits<Derived>::Form, Derived>
      eval_;
  DiscreteFunctionInit<Scalar, LinearOperatorTraits<Derived>::Form, Derived>
      init_;
  Eigen::SparseMatrix<Scalar> mass_;
  Eigen::SparseMatrix<Scalar> stiffness_;
  bool mass_initialized;
  bool stiffness_initialized;
}; // namespace Bembel

} // namespace Bembel
#endif
