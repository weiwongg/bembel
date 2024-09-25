// This file is part of Bembel, the higher order C++ boundary element library.
//
// Copyright (C) 2022 see <http://www.bembel.eu>
//
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_SRC_GEOMETRY_POLYNOMIALPATCH_HPP_
#define BEMBEL_SRC_GEOMETRY_POLYNOMIALPATCH_HPP_

namespace Bembel {

/**
 * \ingroup Geometry
 * \class PolynomialPatch
 * \brief handles a single polynomial patch
 */
class PolynomialPatch {
 public:
  PolynomialPatch() {}
  PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int nx, int px,
                  int ny = -1, int py = -1, double scale = 1.) {
    init_PolynomialPatch(interpolation_points, nx, px, ny, py, scale);
  }
  void init_PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int nx,
                            int px, int ny = -1, int py = -1,
                            double scale = 1.) {
    nx_ = nx;
    px_ = px;
    if (ny == -1 || py == -1) {
      ny_ = nx_;
      py_ = px_;
    } else {
      ny_ = ny;
      py_ = py;
    }
    assert(nx_ > px_ && ny_ > py_ && "polynomial degree too high");
    assert(px_ > 0 && py_ > 0 && "polynomial degree too low");
    //  determine number of windows
    //  for polynomial degree p, we get p+1 inerpolation points
    //  per polynomial. Hence, the polynomial starting at position
    //  i has then support i,i+1,..i+p. Example:
    //  p = 2, n = 5: |0 2|1 3|2 4|
    //  p = 3, n = 5: |0 3|1 4|
    nwin_x_ = (nx_ - px_);
    nwin_y_ = (ny_ - py_);
    Eigen::Matrix<double, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    // set up interpolation points in [0,nx]
    xvec_ = Eigen::VectorXd::LinSpaced(nx_, 0, nx_ - 1);
    // set up interpolation points in [0,ny]
    yvec_ = Eigen::VectorXd::LinSpaced(ny_, 0, ny_ - 1);
    // determine coefficients of interpolating polynomials
    polynomials_.resize(nwin_x_, nwin_y_);
    for (auto i = 0; i < nwin_x_; ++i)
      for (auto j = 0; j < nwin_y_; ++j) {
        polynomials_(i, j).resize(3, (px_ + 1) * (py_ + 1));
        polynomials_(i, j).setZero();
        // copy current chunk of interpolation points
        for (auto l = 0; l <= py_; ++l)
          polynomials_(i, j).block(0, l * (px_ + 1), 3, (px_ + 1)) =
              scale *
              interpolation_points.block(0, (j + l) * nx_ + i, 3, (px_ + 1));
        // interpolate values in y-direction
        for (auto k = 0; k <= px_; ++k)
          for (auto l = 1; l <= py_; ++l)
            for (auto m = py_; m >= l; --m)
              polynomials_(i, j).col(m * (px_ + 1) + k) =
                  (polynomials_(i, j).col(m * (px_ + 1) + k) -
                   polynomials_(i, j).col((m - 1) * (px_ + 1) + k)) /
                  (yvec_(j + m) - yvec_(j + m - l));
        // interpolate values in x-direction
        for (auto k = 0; k <= py_; ++k)
          for (auto l = 1; l <= px_; ++l)
            for (auto m = px_; m >= l; --m)
              polynomials_(i, j).col(k * (px_ + 1) + m) =
                  (polynomials_(i, j).col(k * (px_ + 1) + m) -
                   polynomials_(i, j).col(k * (px_ + 1) + m - 1)) /
                  (xvec_(i + m) - xvec_(i + m - l));
      }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d eval(const Eigen::Vector2d &reference_point) const {
    // compute the right window by index shift in [0, nx] x [0, ny]
    Eigen::Matrix<double, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));

    double x = reference_point(0) * (nx_ - 1);
    double y = reference_point(1) * (ny_ - 1);
    int ind_x = int(x) >= nwin_x_ ? nwin_x_ - 1 : int(x);
    int ind_y = int(y) >= nwin_y_ ? nwin_y_ - 1 : int(y);
    const Eigen::MatrixXd &pol = polynomials_(ind_x, ind_y);
    // first evaluate along x-direction
    for (auto i = 0; i <= py_; ++i)
      eval_buffer.col(i) =
          HornersMethod(pol.block(0, i * (px_ + 1), 3, (px_ + 1)),
                        xvec_.segment(ind_x, px_ + 1), x);
    // then interpolate the result in y-direction
    return HornersMethod(eval_buffer.block(0, 0, 3, py_ + 1),
                         yvec_.segment(ind_y, py_ + 1), y);
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 2> evalJacobian(
      const Eigen::Vector2d &reference_point) const {
    Eigen::Matrix<double, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    Eigen::Matrix<double, 3, 2> retval;
    // compute the right window by index shift in [0, nx] x [0, ny]
    double x = reference_point(0) * (nx_ - 1);
    double y = reference_point(1) * (ny_ - 1);
    int ind_x = int(x) >= nwin_x_ ? nwin_x_ - 1 : int(x);
    int ind_y = int(y) >= nwin_y_ ? nwin_y_ - 1 : int(y);
    const Eigen::MatrixXd &pol = polynomials_(ind_x, ind_y);
    // first evaluate along x-direction
    for (auto i = 0; i <= py_; ++i) {
      Eigen::Matrix<double, 3, 2> ret =
          HornersMethodDerivative(pol.block(0, i * (px_ + 1), 3, (px_ + 1)),
                                  xvec_.segment(ind_x, px_ + 1), x);
      eval_buffer.col(i) = ret.col(0);
      eval_buffer.col(i + py_ + 1) = ret.col(1);
    }
    // then interpolate the result in y-direction
    retval.col(0) = HornersMethod(eval_buffer.block(0, py_ + 1, 3, py_ + 1),
                                  yvec_.segment(ind_y, py_ + 1), y) *
                    (nx_ - 1);
    retval.col(1) = HornersMethodDerivative(eval_buffer.block(0, 0, 3, py_ + 1),
                                            yvec_.segment(ind_y, py_ + 1), y)
                        .col(1) *
                    (ny_ - 1);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 1> evalNormal(
      const Eigen::Vector2d &reference_point) const {
    auto J = evalJacobian(reference_point);
    return J.col(0).cross(J.col(1));
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d eval(double x, double y) const {
    return eval(Eigen::Vector2d(x, y));
  }
  Eigen::Matrix<double, 3, 2> evalJacobian(double x, double y) const {
    return evalJacobian(Eigen::Vector2d(x, y));
  }
  Eigen::Matrix<double, 3, 1> evalNormal(double x, double y) const {
    return evalNormal(Eigen::Vector2d(x, y));
  }
  /*
   * This typedef is essential for any evaluation of a bilinear form. It
   *provides all required geometry information, stored as follows: (0)
   *x-coordinate of the evaluation point in the parameter domain [0,1]^2 of the
   *current element, i.e we map [0,1]^2->element->surface (1) y-coordinate of
   *the evaluation point in the parameter domain [0,1]^2 of the current element,
   *i.e we map [0,1]^2->element->surface (2) a quadrature weight. Can be left
   *empty if not used as part of a quadrature. (3) x-coordinate of patch eval in
   *space (4) y-coordinate of patch eval in space (5) z-coordinate of patch eval
   *in space (6) x-component of derivative in x-dir (7) y-component of
   *derivative in x-dir (8) z-component of derivative in x-dir (9) x-component
   *of derivative in y-dir (10) y-component of derivative in y-dir (11)
   *z-component of derivative in y-dir For application of the pull-back to the
   *reference domain, one requires the jacobian of any point on the surface.
   *Calling eval and evalJacobian of the Patch class introduces work that needs
   *to be done twice. The updateSurdacePoint method is specialized and should be
   *used, since it avoids redundant work.
   **/
  void updateSurfacePoint(SurfacePoint *srf_pt, const Eigen::Vector2d &ref_pt,
                          double w, const Eigen::Vector2d &xi) const {
    // compute the right window by index shift in [0, nx] x [0, ny]
    Eigen::Matrix<double, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    double x = ref_pt(0) * (nx_ - 1);
    double y = ref_pt(1) * (ny_ - 1);
    int ind_x = int(x) >= nwin_x_ ? nwin_x_ - 1 : int(x);
    int ind_y = int(y) >= nwin_y_ ? nwin_y_ - 1 : int(y);
    const Eigen::MatrixXd &pol = polynomials_(ind_x, ind_y);
    // first evaluate along x-direction
    for (auto i = 0; i <= py_; ++i) {
      auto ret =
          HornersMethodDerivative(pol.block(0, i * (px_ + 1), 3, (px_ + 1)),
                                  xvec_.segment(ind_x, px_ + 1), x);
      eval_buffer.col(i) = ret.col(0);
      eval_buffer.col(i + py_ + 1) = ret.col(1);
    }
    // evaluate along y-direction and write everything to the output
    srf_pt->segment(0, 2) = xi;
    (*srf_pt)(2) = w;
    srf_pt->segment(3, 3) = HornersMethod(eval_buffer.block(0, 0, 3, py_ + 1),
                                          yvec_.segment(ind_y, py_ + 1), y);
    srf_pt->segment(6, 3) =
        HornersMethod(eval_buffer.block(0, py_ + 1, 3, py_ + 1),
                      yvec_.segment(ind_y, py_ + 1), y) *
        (nx_ - 1);
    srf_pt->segment(9, 3) =
        HornersMethodDerivative(eval_buffer.block(0, 0, 3, py_ + 1),
                                yvec_.segment(ind_y, py_ + 1), y)
            .col(1) *
        (ny_ - 1);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
 private:
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived>
  Eigen::Vector3d HornersMethod(const Eigen::MatrixBase<Derived> &coeffs,
                                const Eigen::MatrixBase<otherDerived> &x,
                                double xi) const {
    Eigen::Vector3d retval;
    retval = coeffs.col(x.size() - 1);
    for (auto i = x.size() - 2; i >= 0; --i)
      retval = retval * (xi - x(i)) + coeffs.col(i);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived>
  Eigen::Matrix<double, 3, 2> HornersMethodDerivative(
      const Eigen::MatrixBase<Derived> &coeffs,
      const Eigen::MatrixBase<otherDerived> &x, double xi) const {
    Eigen::Matrix<double, 3, 2> retval;
    retval.col(0) = coeffs.col(x.size() - 1);
    retval.col(1).setZero();
    for (auto i = x.size() - 2; i >= 0; --i) {
      retval.col(1) = retval.col(1) * (xi - x(i)) + retval.col(0);
      retval.col(0) = retval.col(0) * (xi - x(i)) + coeffs.col(i);
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix<Eigen::MatrixXd> polynomials_;
  Eigen::VectorXd xvec_;
  Eigen::VectorXd yvec_;
  int nx_;
  int ny_;
  int px_;
  int py_;
  int nwin_x_;
  int nwin_y_;
};
}  // namespace Bembel

#endif  // BEMBEL_SRC_GEOMETRY_POLYNOMIALPATCH_HPP_
