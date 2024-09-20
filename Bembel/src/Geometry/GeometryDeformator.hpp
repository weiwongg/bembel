// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_GEOMETRY_DEFORMGEOMETRY_H_
#define BEMBEL_GEOMETRY_DEFORMGEOMETRY_H_

namespace Bembel {

using namespace Eigen;

namespace GeometryDeformatorHelper {

class BezInterface {
 public:
  BezInterface(const Patch& in) {
    p = std::make_shared<Patch>();
    *p = in;
  }
  virtual std::vector<double> value() = 0;
  virtual void setup() = 0;
  virtual const bool isweight() const = 0;

  virtual ~BezInterface() = default;  // arbeit mit Pointern -> kein Memory leak
  std::shared_ptr<Patch> p;
};

class Weight : public BezInterface {
 public:
  Weight(const Patch& in) : BezInterface(in) { setup(); }
  void setup() override {
    int n = (this->p->data_.size());
    weights.resize((int)n / 4);
    for (int i = 0; i < n / 4; ++i) {
      weights[i] = (this->p->data_[4 * i + 3]);
    }
  }
  std::vector<double> value() override { return weights; }
  const bool isweight() const override { return true; }

 private:
  std::vector<double> weights;
};

class Point : public BezInterface {
 public:
  Point(const Patch& in) : BezInterface(in) { setup(); };
  void setup() override {
    int n = (this->p->data_.size());
    points.resize((int)3 * n / 4);
    for (int i = 0; i < n / 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        points[3 * i + j] = this->p->data_[4 * i + j];
      }
    }
  }
  std::vector<double> value() override { return points; }
  const bool isweight() const override { return false; }

 private:
  std::vector<double> points;
};
}  // namespace GeometryDeformatorHelper

/**
 *  \ingroup Geometry
 *  \brief The GeometryDeformator takes a NURBS geometry and deformes it by
 *adding a B-spline surface to each patch using the NURBS-addition.
 *
 * \todo This class requires refacturing. Most urgent: put the NURBS arithmetic
 *into its own class.
 *
 * \Warning This class will be refactured in the future.
 */
class GeometryDeformator {
 public:
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  GeometryDeformator() : polynomial_degree_(-1), refinement_level_(-1) {}
  GeometryDeformator(Geometry& undef, int level, int degree)
      : polynomial_degree_(degree),
        refinement_level_(level),
        original_geometry_(undef) {
    n_ = 1 << refinement_level_;
    n2_ = n_ * n_;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    Methods
  //////////////////////////////////////////////////////////////////////////////
  void calculate_deformation(const std::vector<Eigen::MatrixXd>& patch_vec) {
    // asserts
    assert(patch_vec.size() == original_geometry_.get_number_of_patches());

    // init
    int number_of_patches = original_geometry_.get_number_of_patches();
    std::vector<std::vector<double>> zahler_left;
    std::vector<std::vector<double>> zahler_right;
    std::vector<std::pair<std::vector<double>, std::pair<int, int>>> nenner;
    zahler_left.reserve(number_of_patches);
    zahler_right.reserve(number_of_patches);
    nenner.reserve(number_of_patches);
    PatchVector out(number_of_patches);

    // loop over patches
    for (int k = 0; k < number_of_patches; ++k) {
      // #### here ########################################
      Patch original_patch = RefinePatch(original_geometry_.get_geometry()[k]);
      Patch patch_deformation = reorder(patch_vec[k]);

      GeometryDeformatorHelper::Point original_patch_coefficients(
          original_patch);
      GeometryDeformatorHelper::Point deformation_coefficients(
          patch_deformation);
      GeometryDeformatorHelper::Weight original_patch_weights(original_patch);
      GeometryDeformatorHelper::Weight patch_weights(patch_deformation);
      std::vector<double> curr_data;
      zahler_left.push_back(std::get<0>(
          BezierMultiplication(original_patch_coefficients, patch_weights)));
      zahler_right.push_back(std::get<0>(BezierMultiplication(
          deformation_coefficients, original_patch_weights)));
      nenner.push_back(
          BezierMultiplication(original_patch_weights, patch_weights));
      int n = std::get<0>(nenner[k]).size();
      assert(n == std::pow(4, refinement_level_) *
                      (original_patch.polynomial_degree_x_ +
                       patch_deformation.polynomial_degree_x_ - 1) *
                      (original_patch.polynomial_degree_y_ +
                       patch_deformation.polynomial_degree_y_ - 1));
      curr_data.resize(4 * n);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 4; ++j) {
          if (j < 3) {
            curr_data[4 * i + j] =
                (zahler_left[k][3 * i + j] + zahler_right[k][3 * i + j]);
          } else {
            curr_data[4 * i + j] = std::get<0>(nenner[k])[i];
          }
        }
      }
      out[k].polynomial_degree_x_ = std::get<0>(std::get<1>(nenner[k])) + 1;
      out[k].polynomial_degree_y_ = std::get<1>(std::get<1>(nenner[k])) + 1;
      out[k].unique_knots_x_ = original_patch.unique_knots_x_;
      out[k].unique_knots_y_ = original_patch.unique_knots_y_;
      out[k].data_ = std::move(curr_data);
    }
    deformed_ = Geometry(out);
#if 0  // set to one if patchshredder should be activated automatically
    PatchVector out_final;
    std::vector<int> ordering = reorder_cluster();
    out_final.reserve(number_of_patches * std::pow(4, refinement_level_));
    for (int i = 0; i < number_of_patches; ++i) {
      PatchVector ins = Bembel::PatchShredder(out[i]);
      for (int j = 0; j < ins.size(); ++j) {
        out_final.push_back(ins[ordering[j]]);
      }
    }

    Geometry temp(out_final);
    deformed_ = temp;
#endif
  }

  Geometry get_deformed() { return deformed_; }

 private:
  Geometry deformed_;
  const Geometry original_geometry_;  /// geometry to deform
  int polynomial_degree_;             /// polynomial degree of deformation
  int refinement_level_;              /// refinement level of deformation field
  int n_;
  int n2_;

  Patch reorder(const Eigen::MatrixXd&
                    points) {  // Deformation field in Patch umwandeln, damit
                               // die indizes passen fuer NURBS addition
    int points_per_elem = (polynomial_degree_ + 1) * (polynomial_degree_ + 1);
    int elem_per_dim = 1 << refinement_level_;
    int n = points.rows();
    double h = std::pow(2, -refinement_level_);
    int Bezier_per_level =
        elem_per_dim *
        (polynomial_degree_ + 1);  // number of elements in one dimension times
                                   // Basis functions per element per dimension
    Patch res;
    std::vector<double> res_data;
    res_data.resize(Bezier_per_level * Bezier_per_level * 4);
    assert(n == Bezier_per_level * Bezier_per_level);
    std::vector<double> unique_knots;
    double to_insert = 0;
    while (to_insert <= 1) {
      unique_knots.push_back(to_insert);
      to_insert += h;
    }
    int col = elem_per_dim * (polynomial_degree_ + 1);
    for (int k = 0; k < n; ++k) {
      div_t m = div(k, col);  // quotient gibt an in welcher spalte
      div_t curr_y =
          div(m.rem,
              polynomial_degree_ +
                  1);  // quotient gibt an in welchem element horizontal
                       // (erster Matrixindizes). rem gibt vertikal wert an
      div_t curr_x =
          div(m.quot,
              polynomial_degree_ + 1);  // vertikal. rem gibt horizontal wert an
      int elem = (curr_y.quot) * elem_per_dim + curr_x.quot;
      int index = elem * points_per_elem +
                  (curr_y.rem) * (polynomial_degree_ + 1) +
                  curr_x.rem;  // k -> index in patchorder
      for (int j = 0; j < 4; ++j) {
        if (j < 3) {
          res_data[4 * k + j] = points(index, j);
        } else {
          res_data[4 * k + j] = 1;  // konstante Gewichte
        }
      }
    }
    res.unique_knots_x_ = unique_knots;
    res.unique_knots_y_ = unique_knots;
    res.polynomial_degree_x_ = polynomial_degree_ + 1;
    res.polynomial_degree_y_ = polynomial_degree_ + 1;
    res.data_ = res_data;

    return res;
  }

  std::pair<std::vector<double>, std::pair<int, int>> BezierMultiplication(
      GeometryDeformatorHelper::BezInterface& left,
      GeometryDeformatorHelper::BezInterface&
          right) {  // returned vector an Controlpoints und
                    // polynomgrad. Knotenvektor ist bekannt
    std::pair<int, int> left_deg{(left.p->polynomial_degree_x_ - 1),
                                 (left.p->polynomial_degree_y_ - 1)};
    std::pair<int, int> right_deg{(right.p->polynomial_degree_x_ - 1),
                                  (right.p->polynomial_degree_y_ - 1)};
    std::pair<int, int> poly_out{
        std::get<0>(left_deg) + std::get<0>(right_deg),
        std::get<1>(left_deg) + std::get<1>(right_deg)};
    std::vector<double> left_val = left.value();
    std::vector<double> right_val = right.value();
    int chips_perdim = std::pow(2, refinement_level_);
    int Bezier_x = chips_perdim * (std::get<0>(poly_out) + 1);
    int Bezier_y = chips_perdim * (std::get<1>(poly_out) + 1);
    int numy_l = chips_perdim * (std::get<1>(left_deg) + 1);
    int numy_r = chips_perdim * (std::get<1>(right_deg) + 1);
    const bool acase = left.isweight() && right.isweight();
    std::vector<double> out_vec;
    if (acase) {
      out_vec.resize(Bezier_x * Bezier_y);
    } else {
      out_vec.resize(3 * Bezier_x * Bezier_y);
    }
    for (int i = 0; i < Bezier_x; ++i) {  // i=spalte=x, j= reihe = y
      for (int j = 0; j < Bezier_y; ++j) {
        int index = i * Bezier_y + j;
        div_t bezier_index_x =
            div(i, (std::get<0>(poly_out) + 1));  // (\kappa , r)
        div_t bezier_index_y = div(j, (std::get<1>(poly_out) + 1));
        std::pair<int, int> bezier_cut{bezier_index_x.quot,
                                       bezier_index_y.quot};
        int local_x = bezier_index_x.rem;
        int local_y = bezier_index_y.rem;
        int inner_x =
            local_x < std::get<0>(right_deg)
                ? std::min(std::get<0>(left_deg), local_x)
                : std::min(std::get<0>(poly_out) - local_x,
                           std::get<0>(right_deg));  // Berechnung von a^*_{r_j}
                                                     // wie im pseudocode
        int inner_y = local_y < std::get<1>(right_deg)
                          ? std::min(std::get<1>(left_deg), local_y)
                          : std::min(std::get<1>(poly_out) - local_y,
                                     std::get<1>(right_deg));
        for (int ix = 0; ix <= inner_x; ++ix) {
          std::pair<int, int> cols{
              std::max(0, local_x - std::get<0>(right_deg)) + ix,
              std::min(local_x, std::get<0>(right_deg)) -
                  ix};  // summe ist immer local_x. Entspricht Funktion
                        // I_{r_j}(n) im pseudocode
          assert(0 <= std::get<0>(cols) &&
                 std::get<0>(cols) <= std::get<0>(left_deg));
          assert(0 <= std::get<1>(cols) &&
                 std::get<1>(cols) <= std::get<0>(right_deg));
          assert(std::get<0>(cols) + std::get<1>(cols) == local_x);
          for (int iy = 0; iy <= inner_y; ++iy) {
            std::pair<int, int> rows{
                std::max(0, local_y - std::get<1>(right_deg)) + iy,
                std::min(local_y, std::get<1>(right_deg)) - iy};
            assert(0 <= std::get<0>(rows) &&
                   std::get<0>(rows) <= std::get<1>(left_deg));
            assert(0 <= std::get<1>(rows) &&
                   std::get<1>(rows) <= std::get<1>(right_deg));
            assert(std::get<0>(rows) + std::get<1>(rows) == local_y);
            double prod =
                binomcoeff(std::get<0>(left_deg), std::get<0>(cols)) *
                binomcoeff(std::get<1>(left_deg), std::get<0>(rows)) *
                binomcoeff(std::get<0>(right_deg), std::get<1>(cols)) *
                binomcoeff(std::get<1>(right_deg), std::get<1>(rows));
            int global_in_l =
                (std::get<0>(cols) +
                 (std::get<0>(bezier_cut) * (std::get<0>(left_deg) + 1))) *
                    numy_l +
                std::get<0>(rows) +
                (std::get<1>(bezier_cut) *
                 (std::get<1>(left_deg) + 1));  // globaler koeffizient.
            int global_in_r =
                (std::get<1>(cols) +
                 (std::get<0>(bezier_cut) * (std::get<0>(right_deg) + 1))) *
                    numy_r +
                std::get<1>(rows) +
                (std::get<1>(bezier_cut) * (std::get<1>(right_deg) + 1));
            if (acase) {
              out_vec[index] +=
                  left_val[global_in_l] * right_val[global_in_r] * prod;
            } else {
              if (!left.isweight()) {
                for (int k = 0; k < 3; ++k) {
                  out_vec[3 * index + k] += left_val[3 * global_in_l + k] *
                                            right_val[global_in_r] * prod;
                }
              } else {
                for (int k = 0; k < 3; ++k) {
                  out_vec[3 * index + k] += left_val[global_in_l] *
                                            right_val[3 * global_in_r + k] *
                                            prod;
                }
              }
            }
          }
        }
        if (acase) {
          out_vec[index] /= (binomcoeff(std::get<0>(poly_out), local_x) *
                             binomcoeff(std::get<1>(poly_out), local_y));
        } else {
          for (int k = 0; k < 3; ++k) {
            out_vec[3 * index + k] /=
                (binomcoeff(std::get<0>(poly_out), local_x) *
                 binomcoeff(std::get<1>(poly_out), local_y));
          }
        }
      }
    }
    return {out_vec, poly_out};
  }

  // um die  NURBS-addition zu benutzen müssen
  // sowohl patch als auch Deformation field
  // einen gleichen Knotenvektor haben.
  // unique_knots_x_=unique_knots_y_={0,1} => p+1 ControlPoints in den beiden
  // Dimensionskoordinaten. Wir benutzen KnotInsertion Algorithm
  Patch RefinePatch(const Patch& in) {
    double h = 1. / n_;
    int poly_x = in.polynomial_degree_x_ - 1;
    int poly_y = in.polynomial_degree_y_ - 1;
    int newdata_size = 4 * n2_ * (poly_x + 1) * (poly_y + 1);
    if (newdata_size == in.data_.size()) {
      // if (refinement_level_ == 0) {
      return in;
    }
    std::vector<double> newdata(newdata_size);
    std::vector<double> newdata_old;
    int num_y = n_ * (poly_y + 1);
    for (int s = 0; s < in.data_.size() / 4; ++s) {
      div_t divres = div(s, (poly_y + 1));
      int row = divres.rem;
      int col = divres.quot;
      int start = 4 * (col * num_y + row);
      for (int xyzw = 0; xyzw < 4; ++xyzw) {
        newdata[start + xyzw] = in.data_[4 * s + xyzw];
      }
    }
    double to_insert = h;
    int count = 1;
    int vec_count = 0;
    std::vector<double> unique_knots;
    unique_knots.reserve(n_ + 1);
    unique_knots.push_back(0);
    while (count < n_) {
      unique_knots.push_back(to_insert);
      int loc_in_knotvec_x = count * poly_x + count - 1;  // count*(p+1)-1
      int loc_in_knotvec_y = count * poly_y + count - 1;
      for (int z = 1; z <= poly_x + 1; ++z) {  // p+1 mal einfügen
        ++vec_count;
        newdata_old = newdata;
        newdata.resize(newdata_size);
        for (int a_t = 0; a_t < count * (poly_x + 1) + z; ++a_t) {
          int a = a_t * num_y;
          for (int j = 0; j < count * (poly_y + 1); ++j) {
            int start = 4 * (a + j);
            if (a_t < loc_in_knotvec_x - poly_x + z) {
              for (int xyzw = 0; xyzw < 4; ++xyzw) {
                newdata[start + xyzw] = newdata_old[start + xyzw];
              }
            } else {
              int prev_index = start - (4 * num_y);
              if (loc_in_knotvec_x - poly_x + z <= a_t &&
                  a_t <= loc_in_knotvec_x) {
                double alpha = h / (1 - (to_insert - h));
                for (int xyzw = 0; xyzw < 4; ++xyzw) {
                  newdata[start + xyzw] =
                      alpha * newdata_old[start + xyzw] +
                      (1 - alpha) * newdata_old[prev_index + xyzw];
                }
              } else {
                for (int xyzw = 0; xyzw < 4; ++xyzw) {
                  newdata[start + xyzw] = newdata_old[prev_index + xyzw];
                }
              }
            }
          }
        }
      }
      for (int z = 1; z <= poly_y + 1; ++z) {  // p+1 mal einfügen in y richtung
        ++vec_count;
        //
        newdata_old = newdata;
        newdata.resize(newdata_size);
        for (int a = 0; a < count * (poly_y + 1) + z; ++a) {
          for (int j_t = 0; j_t < (count + 1) * (poly_x + 1); ++j_t) {
            int start = 4 * (a + j_t * num_y);
            if (a < loc_in_knotvec_y - poly_y + z) {
              for (int xyzw = 0; xyzw < 4; ++xyzw) {
                newdata[start + xyzw] = newdata_old[start + xyzw];
              }
            } else {
              int prev_index = start - 4;
              if (loc_in_knotvec_y - poly_y + z <= a && a <= loc_in_knotvec_y) {
                double alpha =
                    h /
                    (1 - (to_insert - h));  // since we always insert p+1 times
                                            // this is the resulting structure
                for (int xyzw = 0; xyzw < 4; ++xyzw) {
                  newdata[start + xyzw] =
                      alpha * newdata_old[start + xyzw] +
                      (1 - alpha) * newdata_old[prev_index + xyzw];
                }
              } else {
                for (int xyzw = 0; xyzw < 4; ++xyzw) {
                  newdata[start + xyzw] = newdata_old[prev_index + xyzw];
                }
              }
            }
          }
        }
      }
      count += 1;
      to_insert += h;
    }
    unique_knots.push_back(1);
    return Patch(newdata, in.polynomial_degree_x_, in.polynomial_degree_y_,
                 unique_knots, unique_knots);
  }

  int factorial(const int n) {
    assert(n >= 0);
    int prod = 1;
    for (int i = 1; i < n + 1; ++i) prod *= i;
    return prod;
  }

  double binomcoeff(int n, int k) {
    assert(k <= n);
    return (factorial(n)) / (factorial(k) * factorial(n - k));
  }

  std::vector<int> reorder_cluster() {
    std::vector<int> out(std::pow(4, refinement_level_));
    for (int i = 0; i < std::pow(4, refinement_level_); ++i) {
      div_t m = div(i, std::pow(2, refinement_level_));
      out[i] = m.rem * std::pow(2, refinement_level_) +
               (std::pow(2, refinement_level_) - 1 - m.quot);
    }

    return out;
  }
};

}  // namespace Bembel
#endif
