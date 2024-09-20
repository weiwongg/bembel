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

#define BEMBEL_BC_ZERO 1e-14

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
  PolynomialPatch(){};
  PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int n,
                  double scale = 1.,
                  Eigen::Vector3d shift = Eigen::VectorXd::Zero(3)) {
    init_PolynomialPatch(interpolation_points, n, scale, shift);
  }
  void init_PolynomialPatch(const Eigen::MatrixXd &interpolation_points, int n,
                            double scale = 1.,
                            Eigen::Vector3d shift = Eigen::Vector3d(0, 0, 0)) {
    n_ = n;
    p_ = n_ - 1;
    // set up interpolation points in [0,nx]
    xvec_ = (BEMBEL_PI / p_) * Eigen::VectorXd::LinSpaced(n_, 0., p_);
    xvec_ = 0.5 * (1 - xvec_.array().cos());
    yvec_ = xvec_;
    xweights_.resize(n_);
    // xweights_ = barycentricWeights(xvec_);
    for (int i = 0; i < xweights_.size(); ++i)
      if (i % 2)
        xweights_(i) = -1;
      else
        xweights_(i) = 1;
    xweights_(0) *= 0.5;
    xweights_(xweights_.size() - 1) *= 0.5;
    yweights_ = xweights_;
    Dx_ = computeDiffMat(xvec_, xweights_);
    Dy_ = Dx_;
    // determine coefficients of interpolating polynomials
    polynomial_ = scale*(interpolation_points + shift.replicate(1,interpolation_points.cols()));
    return;
  };
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d eval(const Eigen::Vector2d &reference_point) const {
    double x = reference_point(0);
    double y = reference_point(1);
    const Eigen::MatrixXd &pol = polynomial_;
    // compute barycentric weights (they may become infinity, however, this
    // is cought later)
    const double wx = (xweights_.array() / (x - xvec_.array())).sum();
    const double wy = (yweights_.array() / (y - yvec_.array())).sum();
    double alphax = 0;
    double alphay = 0;
    Eigen::Index indx = 0;
    Eigen::Index indy = 0;
    Eigen::Matrix<double, 3, 1> retval;
    retval.setZero();
    // now perform the tensor product barycentric interpolation
    const int the_case =
        2 * (abs(wy) > 1. / BEMBEL_BC_ZERO) + (abs(wx) > 1. / BEMBEL_BC_ZERO);
    switch (the_case) {
      case 0: {
        // no problem case
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / wy;
          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / wx;
            retval += alphax * alphay * pol.col(i * (p_ + 1) + j);
          }
        }
        return retval;
      }
      case 1: {
        // x problem case
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / wy;
          retval += alphay * pol.col(i * (p_ + 1) + indx);
        }

        return retval;
      }
      case 2: {
        // y problem case
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        for (int j = 0; j <= p_; ++j) {
          alphax = (xweights_(j) / (x - xvec_(j))) / wx;
          retval += alphax * pol.col(indy * (p_ + 1) + j);
        }
        return retval;
      }
      case 3: {
        // x and y problem case
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        retval = pol.col(indy * (p_ + 1) + indx);
        return retval;
      }
    }
    return Eigen::Vector3d::Zero();
  };
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 2> evalJacobian(
      const Eigen::Vector2d &reference_point) const {
    Eigen::Matrix<double, 3, 2> retval;
    double x = reference_point(0);
    double y = reference_point(1);
    const Eigen::MatrixXd &pol = polynomial_;
    // compute barycentric weights (they may become infinity, however, this
    // is cought later)
    const double wx = (xweights_.array() / (x - xvec_.array())).sum();
    const double wy = (yweights_.array() / (y - yvec_.array())).sum();
    const double wpx =
        (-xweights_.array() / (x - xvec_.array()).square()).sum();
    const double wpy =
        (-yweights_.array() / (y - yvec_.array()).square()).sum();
    double alphax = 0;
    double alphadx = 0;
    double alphay = 0;
    double alphady = 0;
    Eigen::Index indx = 0;
    Eigen::Index indy = 0;
    Eigen::Matrix<double, 3, 1> divx;
    Eigen::Matrix<double, 3, 1> divy;
    retval.setZero();
    // unfortunately, to have a derivative for all situations, we need
    //  to consider the cases (abs(wx) ==inf) and (abs(wy) ==inf)
    //  and all their combinations
    const int the_case =
        2 * (abs(wy) > 1. / BEMBEL_BC_ZERO) + (abs(wx) > 1. / BEMBEL_BC_ZERO);
    switch (the_case) {
      case 0: {
        // the fast no problem case
        // derivative in x direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / wy;
          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / (wx * wx);
            alphadx =
                (-xweights_(j) / ((x - xvec_(j)) * (x - xvec_(j)))) / (wx * wx);
            retval.col(0) += (alphay * (alphadx * wx - alphax * wpx) *
                              pol.col(i * (p_ + 1) + j));
          }
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / (wy * wy);
          alphady =
              (-yweights_(i) / ((y - yvec_(i)) * (y - yvec_(i)))) / (wy * wy);

          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / wx;
            retval.col(1) += ((alphady * wy - alphay * wpy) * alphax *
                              pol.col(i * (p_ + 1) + j));
          }
        }
        return retval;
      }
      case 1: {
        // in x we are at a node and y is no problem
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        // derivative in x direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (abs(y - yvec_(i)) > BEMBEL_BC_ZERO)
                       ? (yweights_(i) / (y - yvec_(i))) / wy
                       : 1.;
          divx = pol.block(0, i * (p_ + 1), 3, (p_ + 1)) *
                 Dx_.row(indx).transpose();
          retval.col(0) += alphay * divx;
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / (wy * wy);
          alphady =
              (-yweights_(i) / ((y - yvec_(i)) * (y - yvec_(i)))) / (wy * wy);
          retval.col(1) +=
              (alphady * wy - alphay * wpy) * pol.col(i * (p_ + 1) + indx);
        }
        return retval;
      }
      case 2: {
        // in y we are at a node and x is no problem
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        // derivative in x direction
        for (int j = 0; j <= p_; ++j) {
          alphax = (xweights_(j) / (x - xvec_(j))) / (wx * wx);
          alphadx =
              (-xweights_(j) / ((x - xvec_(j)) * (x - xvec_(j)))) / (wx * wx);
          retval.col(0) +=
              (alphadx * wx - alphax * wpx) * pol.col(indy * (p_ + 1) + j);
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i)
          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / wx;
            retval.col(1) += alphax * Dy_(indy, i) * pol.col(i * (p_ + 1) + j);
          }
        return retval;
      }
      case 3: {
        // in y and x we are at a node
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        // derivative in x direction
        retval.col(0) = (pol.block(0, indy * (p_ + 1), 3, (p_ + 1)) *
                         Dx_.row(indx).transpose());
        // derivative in y direction
        // derivative in y direction
        for (int i = 0; i <= p_; ++i)
          retval.col(1) += Dy_(indy, i) * pol.col(i * (p_ + 1) + indx);
        return retval;
      }
    }
    return Eigen::Matrix<double, 3, 2>::Zero();
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
   *x-coordinate of the evaluation point in the parameter domain [0,1]^2
   *of the current element, i.e we map [0,1]^2->element->surface (1)
   *y-coordinate of the evaluation point in the parameter domain [0,1]^2
   *of the current element, i.e we map [0,1]^2->element->surface (2) a
   *quadrature weight. Can be left empty if not used as part of a
   *quadrature. (3) x-coordinate of patch eval in space (4) y-coordinate
   *of patch eval in space (5) z-coordinate of patch eval in space (6)
   *x-component of derivative in x-dir (7) y-component of derivative in
   *x-dir (8) z-component of derivative in x-dir (9) x-component of
   *derivative in y-dir (10) y-component of derivative in y-dir (11)
   *z-component of derivative in y-dir For application of the pull-back to
   *the reference domain, one requires the jacobian of any point on the
   *surface. Calling eval and evalJacobian of the Patch class introduces
   *work that needs to be done twice. The updateSurdacePoint method is
   *specialized and should be used, since it avoids redundant work.
   **/
  void updateSurfacePoint(Eigen::Matrix<double, 12, 1> *srf_pt,
                          const Eigen::Vector2d &ref_pt, double w,
                          const Eigen::Vector2d &xi) const {
    double x = ref_pt(0);
    double y = ref_pt(1);
    const Eigen::MatrixXd &pol = polynomial_;
    srf_pt->setZero();
    srf_pt->segment(0, 2) = xi;
    (*srf_pt)(2) = w;
    // srf_pt->segment(3, 3) = eval(ref_pt);
    // Eigen::Matrix<double, 3, 2> bla = evalJacobian(ref_pt);
    // srf_pt->segment(6, 3) = bla.col(0);
    // srf_pt->segment(9, 3) = bla.col(1);
    // compute barycentric weights (they may become infinity, however, this
    // is cought later)
    const double wx = (xweights_.array() / (x - xvec_.array())).sum();
    const double wy = (yweights_.array() / (y - yvec_.array())).sum();
    const double wpx =
        (-xweights_.array() / (x - xvec_.array()).square()).sum();
    const double wpy =
        (-yweights_.array() / (y - yvec_.array()).square()).sum();
    double alphax = 0;
    double alphadx = 0;
    double alphay = 0;
    double alphady = 0;
    Eigen::Index indx = 0;
    Eigen::Index indy = 0;
    Eigen::Matrix<double, 3, 1> divx;
    Eigen::Matrix<double, 3, 1> divy;
    // unfortunately, to have a derivative for all situations, we need
    //  to consider the cases (abs(wx) ==inf) and (abs(wy) ==inf)
    //  and all their combinations
    const int the_case =
        2 * (abs(wy) > 1. / BEMBEL_BC_ZERO) + (abs(wx) > 1. / BEMBEL_BC_ZERO);
    switch (the_case) {
      case 0: {
        // the fast no problem case
        // derivative in x direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / wy;
          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / (wx * wx);
            alphadx =
                (-xweights_(j) / ((x - xvec_(j)) * (x - xvec_(j)))) / (wx * wx);
            srf_pt->segment(6, 3) += (alphay * (alphadx * wx - alphax * wpx) *
                                      pol.col(i * (p_ + 1) + j));
            srf_pt->segment(3, 3) +=
                (alphax * wx * alphay * pol.col(i * (p_ + 1) + j));
          }
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / (wy * wy);
          alphady =
              (-yweights_(i) / ((y - yvec_(i)) * (y - yvec_(i)))) / (wy * wy);

          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / wx;
            srf_pt->segment(9, 3) += ((alphady * wy - alphay * wpy) * alphax *
                                      pol.col(i * (p_ + 1) + j));
          }
        }
        break;
      }
      case 1: {
        // in x we are at a node and y is no problem
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        // derivative in x direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (abs(y - yvec_(i)) > BEMBEL_BC_ZERO)
                       ? (yweights_(i) / (y - yvec_(i))) / wy
                       : 1.;
          divx = pol.block(0, i * (p_ + 1), 3, (p_ + 1)) *
                 Dx_.row(indx).transpose();
          srf_pt->segment(6, 3) += (alphay * divx);
          srf_pt->segment(3, 3) += (alphay * pol.col(i * (p_ + 1) + indx));
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i) {
          alphay = (yweights_(i) / (y - yvec_(i))) / (wy * wy);
          alphady =
              (-yweights_(i) / ((y - yvec_(i)) * (y - yvec_(i)))) / (wy * wy);
          srf_pt->segment(9, 3) +=
              ((alphady * wy - alphay * wpy) * pol.col(i * (p_ + 1) + indx));
        }

        break;
      }
      case 2: {
        // in y we are at a node and x is no problem
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        // derivative in x direction
        for (int j = 0; j <= p_; ++j) {
          alphax = (xweights_(j) / (x - xvec_(j))) / (wx * wx);
          alphadx =
              (-xweights_(j) / ((x - xvec_(j)) * (x - xvec_(j)))) / (wx * wx);
          srf_pt->segment(6, 3) +=
              ((alphadx * wx - alphax * wpx) * pol.col(indy * (p_ + 1) + j));
          srf_pt->segment(3, 3) += (alphax * wx * pol.col(indy * (p_ + 1) + j));
        }
        // derivative in y direction
        for (int i = 0; i <= p_; ++i)
          for (int j = 0; j <= p_; ++j) {
            alphax = (xweights_(j) / (x - xvec_(j))) / wx;
            srf_pt->segment(9, 3) +=
                (alphax * Dy_(indy, i) * pol.col(i * (p_ + 1) + j));
          }

        break;
      }
      case 3: {
        // in y and x we are at a node
        for (int j = 0; j <= p_; ++j)
          if (abs(x - xvec_(j)) < BEMBEL_BC_ZERO) {
            indx = j;
            break;
          }
        for (int i = 0; i <= p_; ++i)
          if (abs(y - yvec_(i)) < BEMBEL_BC_ZERO) {
            indy = i;
            break;
          }
        srf_pt->segment(3, 3) = (pol.col(indy * (p_ + 1) + indx));
        // derivative in x direction
        srf_pt->segment(6, 3) = (pol.block(0, indy * (p_ + 1), 3, (p_ + 1)) *
                                 Dx_.row(indx).transpose());
        // derivative in y direction
        for (int i = 0; i <= p_; ++i)
          srf_pt->segment(9, 3) +=
              (Dy_(indy, i) * pol.col(i * (p_ + 1) + indx));
        break;
      }
    }
    return;
  };
  //////////////////////////////////////////////////////////////////////////////
 private:
  Eigen::VectorXd barycentricWeights(const Eigen::VectorXd &xi) {
    Eigen::VectorXd W = xi;
    Eigen::VectorXd logW = xi;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> signum(xi.size());
    W.setZero();
    signum.setZero();
    logW.setZero();
    for (int i = 0; i < xi.size(); ++i) {
      logW(i) = (xi(i) - xi.head(i).array()).abs().log().sum() +
                (xi(i) - xi.tail(xi.size() - i - 1).array()).abs().log().sum();
      signum(i) = ((xi(i) - xi.array()) > 0).sum() % 2;
    }
    logW = logW.array() - logW.minCoeff();
    W = 1. / logW.array().exp();
    W(signum) = -W(signum);
    return W;
  }
  /**
   *  \brief computes the differentiation matrix for the barycentric
   *         interpolation. It holds [f'(xi)] = D * [f(xi)];
   *
   **/
  Eigen::MatrixXd computeDiffMat(const Eigen::VectorXd &xi,
                                 const Eigen::VectorXd &w) {
    Eigen::MatrixXd retval(xi.size(), xi.size());
    retval.array() = (Eigen::VectorXd::Ones(w.size()) * w.transpose()).array() /
                     (w * Eigen::VectorXd::Ones(w.size()).transpose()).array();
    retval.array() /= (xi * Eigen::VectorXd::Ones(xi.size()).transpose() -
                       Eigen::VectorXd::Ones(xi.size()) * xi.transpose())
                          .array();
    retval.diagonal().array() = 0;
    retval -= retval.rowwise().sum().asDiagonal();
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd polynomial_;
  Eigen::VectorXd xvec_;
  Eigen::VectorXd xweights_;
  Eigen::VectorXd yvec_;
  Eigen::VectorXd yweights_;
  Eigen::MatrixXd Dx_;
  Eigen::MatrixXd Dy_;
  int n_;
  int p_;
};
}  // namespace Bembel
#endif
