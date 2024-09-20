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
#include <string>

int main(int argc, char* argv[]) {
  Bembel::Geometry geometry("../geo/david.dat");
  Bembel::PatchVector pv = geometry.get_geometry();
  std::cout << "The geometry has " << geometry.get_number_of_patches()
            << " patches." << std::endl;
  for (int lvl = 1; lvl <= 20; ++lvl) {
    std::cout << "export level: " << lvl << std::endl;
    int num_patches = geometry.get_number_of_patches();
    int n = lvl + 1;
    /* write geometry data to file */
    FILE* file =
        std::fopen(("../geo/david" + std::to_string(lvl) + ".bpd").c_str(), "w");
    std::fprintf(file, "%d\n%d\n", n, num_patches);
    Eigen::VectorXd refs =
        (BEMBEL_PI / lvl) * Eigen::VectorXd::LinSpaced(lvl + 1, 0., lvl);
    refs = 0.5 * (1 - refs.array().cos());
    std::cout << refs.transpose() << std::endl;
    int patch_id = 0;
    for (Bembel::PatchVector::iterator iter = pv.begin(); iter < pv.end();
         iter++) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          Eigen::Vector3d pt = iter->eval(Eigen::Vector2d(refs(j), refs(i)));
          std::fprintf(file, "%d %d %d %34.15f %34.15f %34.15f\n", patch_id, i,
                       j, pt(0), pt(1), pt(2));
        }
      }
      ++patch_id;
    }
    fclose(file);
  }
  return 0;
}
