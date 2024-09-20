// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_GEOMETRY_POLYNOMIALPATCH_H_
#define BEMBEL_GEOMETRY_POLYNOMIALPATCH_H_

#include <Eigen/Dense>
#include <memory>

namespace Bembel {

/**
 *
 *  this class takes nx * ny points in space, which correspond to the values
 *  of an nx x ny cartesian grid on [0,1]^2 and piecewise interpolates them by
 *  tensor product polynomials of degree px x py.
 *  Interpolation is done by a 2D divided difference scheme, while evaluation
 *  of the polynomials and their derivatives is performed by Horner's method.
 **/
class PolynomialPatch {
 public:
  typedef long double ldouble;
  typedef Eigen::Matrix<ldouble, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
  typedef Eigen::Matrix<ldouble, Eigen::Dynamic, 1> VectorXld;
  PolynomialPatch(){};
  PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int nx, int px,
                  int ny = -1, int py = -1, double scale = 1.,
                  Eigen::Vector3d shift = Eigen::VectorXd::Zero(3)) {
    init_PolynomialPatch(interpolation_points, nx, px, ny, py, scale, shift);
  }
  void init_PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int nx,
                            int px, int ny = -1, int py = -1,
                            long double scale = 1.,
                            Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0)) {
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
    Eigen::Matrix<ldouble, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    // set up interpolation points in [0,nx]
    xvec_ = (BEMBEL_PI / px_) * VectorXld::LinSpaced(px_ + 1, 0., px_);
    xvec_ = 0.5 * (1 - xvec_.array().cos());
    yvec_ = xvec_;
    // determine coefficients of interpolating polynomials
    polynomial_.resize(3, (px_ + 1) * (py_ + 1));
    polynomial_.setZero();
    // copy current chunk of interpolation points
    for (int l = 0; l <= py_; ++l)
      polynomial_.block(0, l * (px_ + 1), 3, (px_ + 1)) =
          scale * (interpolation_points.block(0, l * nx_, 3, (px_ + 1))
                       .cast<ldouble>() +
                   shift.replicate(1, (px_ + 1)).cast<ldouble>());
    // interpolate values in y-direction
    for (int k = 0; k <= px_; ++k)
      for (int l = 1; l <= py_; ++l)
        for (int m = py_; m >= l; --m)
          polynomial_.col(m * (px_ + 1) + k) =
              (polynomial_.col(m * (px_ + 1) + k) -
               polynomial_.col((m - 1) * (px_ + 1) + k)) /
              (yvec_(m) - yvec_(m - l));
    // interpolate values in x-direction
    for (int k = 0; k <= py_; ++k)
      for (int l = 1; l <= px_; ++l)
        for (int m = px_; m >= l; --m)
          polynomial_.col(k * (px_ + 1) + m) =
              (polynomial_.col(k * (px_ + 1) + m) -
               polynomial_.col(k * (px_ + 1) + m - 1)) /
              (xvec_(m) - xvec_(m - l));
    return;
  };
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d eval(const Eigen::Vector2d &reference_point) const {
    // compute the right window by index shift in [0, nx] x [0, ny]
    Eigen::Matrix<long double, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));

    double x = reference_point(0);
    double y = reference_point(1);
    const MatrixXld &pol = polynomial_;
    // first evaluate along x-direction
    for (auto i = 0; i <= py_; ++i)
      eval_buffer.col(i) =
          HornersMethod(pol.block(0, i * (px_ + 1), 3, (px_ + 1)), xvec_, x);
    // then interpolate the result in y-direction
    return HornersMethod(eval_buffer.block(0, 0, 3, py_ + 1), yvec_, y)
        .cast<double>();
  };
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 2> evalJacobian(
      const Eigen::Vector2d &reference_point) const {
    Eigen::Matrix<ldouble, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    Eigen::Matrix<double, 3, 2> retval;
    // compute the right window by index shift in [0, nx] x [0, ny]
    double x = reference_point(0);
    double y = reference_point(1);
    const MatrixXld &pol = polynomial_;
    // first evaluate along x-direction
    for (int i = 0; i <= py_; ++i) {
      Eigen::Matrix<ldouble, 3, 2> ret = HornersMethodDerivative(
          pol.block(0, i * (px_ + 1), 3, (px_ + 1)), xvec_, x);
      eval_buffer.col(i) = ret.col(0);
      eval_buffer.col(i + py_ + 1) = ret.col(1);
    }
    // then interpolate the result in y-direction
    retval.col(0) =
        (HornersMethod(eval_buffer.block(0, py_ + 1, 3, py_ + 1), yvec_, y))
            .cast<double>();
    retval.col(1) =
        (HornersMethodDerivative(eval_buffer.block(0, 0, 3, py_ + 1), yvec_, y)
             .col(1))
            .cast<double>();
    return retval;
  };
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 1> evalNormal(
      const Eigen::Vector2d &reference_point) const {
    auto J = evalJacobian(reference_point);
    return J.col(0).cross(J.col(1));
  };
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
    Eigen::Matrix<ldouble, 3, Eigen::Dynamic> eval_buffer(3, 2 * (py_ + 1));
    double x = ref_pt(0) * (nx_ - 1);
    double y = ref_pt(1) * (ny_ - 1);
    const MatrixXld &pol = polynomial_;
    // first evaluate along x-direction
    for (int i = 0; i <= py_; ++i) {
      Eigen::Matrix<ldouble, 3, 2> ret = HornersMethodDerivative(
          pol.block(0, i * (px_ + 1), 3, (px_ + 1)), xvec_, x);
      eval_buffer.col(i) = ret.col(0);
      eval_buffer.col(i + py_ + 1) = ret.col(1);
    }
    // evaluate along y-direction and write everything to the output
    srf_pt->segment(0, 2) = xi;
    (*srf_pt)(2) = w;
    srf_pt->segment(3, 3) =
        HornersMethod(eval_buffer.block(0, 0, 3, py_ + 1), yvec_, y)
            .cast<double>();
    srf_pt->segment(6, 3) =
        (HornersMethod(eval_buffer.block(0, py_ + 1, 3, py_ + 1), yvec_, y))
            .cast<double>();
    srf_pt->segment(9, 3) =
        (HornersMethodDerivative(eval_buffer.block(0, 0, 3, py_ + 1), yvec_, y)
             .col(1))
            .cast<double>();
    return;
  };
  //////////////////////////////////////////////////////////////////////////////
 private:
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived>
  Eigen::Matrix<ldouble, 3, 1> HornersMethod(
      const Eigen::MatrixBase<Derived> &coeffs,
      const Eigen::MatrixBase<otherDerived> &x, long double xi) const {
    Eigen::Matrix<long double, 3, 1> retval;
    retval = coeffs.col(x.size() - 1).template cast<long double>();
    for (int i = x.size() - 2; i >= 0; --i)
      retval = retval * (long double)(xi - x(i)) +
               coeffs.col(i).template cast<long double>();
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived>
  Eigen::Matrix<ldouble, 3, 2> HornersMethodDerivative(
      const Eigen::MatrixBase<Derived> &coeffs,
      const Eigen::MatrixBase<otherDerived> &x, long double xi) const {
    Eigen::Matrix<long double, 3, 2> retval;
    retval.col(0) = coeffs.col(x.size() - 1).template cast<long double>();
    retval.col(1).setZero();
    for (int i = x.size() - 2; i >= 0; --i) {
      retval.col(1) = retval.col(1) * (long double)(xi - x(i)) +
                      retval.col(0).template cast<long double>();
      retval.col(0) = retval.col(0) * (long double)(xi - x(i)) +
                      coeffs.col(i).template cast<long double>();
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  MatrixXld polynomial_;
  VectorXld xvec_;
  VectorXld yvec_;
  int nx_;
  int ny_;
  int px_;
  int py_;
};
}  // namespace Bembel
#endif
