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

#include <Bembel/Geometry>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/LinearForm>
#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main() {
  for (int l = 1; l <= 50; ++l) {
    Bembel::Geometry geometry("../../geo/sphere" + std::to_string(l) + ".bpd");
    Bembel::PatchVector pv = geometry.get_geometry();
    int n = 2;
    Eigen::MatrixXd refs = (Eigen::MatrixXd::Random(n + 1, n + 1) +
                            Eigen::MatrixXd::Constant(n + 1, n + 1, 1.)) /
                           2.0;
    int patch_id = 0;
    double err_coord = 0;
    double err_normal = 0;
    for (Bembel::PatchVector::iterator iter = pv.begin(); iter < pv.end();
         iter++) {
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          Eigen::Vector3d pt =
              iter->eval(Eigen::Vector2d(refs(i, j), refs(i, j)));
          Eigen::Vector3d normal =
              iter->evalNormal(Eigen::Vector2d(refs(i,j), refs(i,j)));
          err_coord += abs(pt.norm() - 1);
          err_normal += (normal / normal.norm() - pt).norm();
        }
      }
      ++patch_id;
    }
    std::cout << "error of coordinate: " << err_coord / (patch_id * n * n)
              << " " << "error of norm: " << err_normal / (patch_id * n * n)
              << std::endl;
  }
  return 0;
}
