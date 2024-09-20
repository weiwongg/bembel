// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_IO_READPOINTSASCII_H_
#define BEMBEL_IO_READPOINTSASCII_H_

#include <Eigen/Dense>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace Bembel {
namespace IO {

void readPointsAscii(std::vector<Eigen::MatrixXd> *P, int *p, int *m,
                     const std::string &fname) {
  int n = 0;  /// n*n panels per patch on level m
  std::FILE *file = std::fopen(fname.c_str(), "r");
  assert(file != nullptr && "readPointsAscii: could not read data file");
  /// read number of levels and number of patches
  fscanf(file, "%d\n%d\n", m, p);
  P->resize(*p);
  n = *m;
  std::cout << "#patches= " << *p << " #points= " << *m << std::endl;
  for (auto it = P->begin(); it != P->end(); ++it)
    it->resize(3, (n) * (n));
  /// read points patchwise in rowwise, lexicographical order
  for (auto i = 0; i < *p; ++i) {
    int ptch, s, t;
    double x, y, z;
    for (auto k = 0; k < (n) * (n); ++k) {
      fscanf(file, "%d %d %d %lg %lg %lg\n", &ptch, &t, &s, &x, &y, &z);
      assert(ptch == i && "readPointsAscii: wrong file format");
      assert(s == k % (n) && "readPointsAscii: wrong file format");
      assert(t == k / (n) && "readPointsAscii: wrong file format");
      // we have that k = s * (n + 1) + t % (n+1)
      // corresponding to the kartesian coordinates (s,t) in [0,n]^2
      (*P)[i].col(k) << x, y, z;
    }
  }
  fclose(file);
  return;
}
}  // namespace IO
}  // namespace Bembel
#endif
