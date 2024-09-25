// This file is part of Bembel, the higher order C++ boundary element library.
//
// Copyright (C) 2024 see <http://www.bembel.eu>
//
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_SRC_GEOMETRY_GEOMETRYPOLY_HPP_
#define BEMBEL_SRC_GEOMETRY_GEOMETRYPOLY_HPP_

namespace Bembel {

/**
 * \ingroup Geometry
 * \brief loads  geometry from Poly file. Note that the direction
 *of the normals must be consistent.
 *
 * \param file_name path/filename pointing to the geometry file
 * \return std::vector of PolynomialPatch describing geometry
 */

void readPointsAscii(std::vector<Eigen::MatrixXd> *P, int *p, int *m,
                     const std::string &fname) {
  int n = 0;  /// n*n panels per patch on level m
  std::FILE *file = std::fopen(fname.c_str(), "r");
  assert(file != nullptr && "readPointsAscii: could not read data file");
  /// read number of levels and number of patches
  fscanf(file, "%d\n%d\n", m, p);
  P->resize(*p);
  n = 1 << (*m);
  std::cout << "#patches= " << *p << " #levels= " << *m << std::endl;
  for (auto it = P->begin(); it != P->end(); ++it)
    it->resize(3, (n + 1) * (n + 1));
  /// read points patchwise in rowwise, lexicographical order
  for (auto i = 0; i < *p; ++i) {
    int ptch, s, t;
    double x, y, z;
    for (auto k = 0; k < (n + 1) * (n + 1); ++k) {
      fscanf(file, "%d %d %d %lg %lg %lg\n", &ptch, &t, &s, &x, &y, &z);
      assert(ptch == i && "readPointsAscii: wrong file format");
      assert(s == k % (n + 1) && "readPointsAscii: wrong file format");
      assert(t == k / (n + 1) && "readPointsAscii: wrong file format");
      // we have that k = s * (n + 1) + t % (n+1)
      // corresponding to the kartesian coordinates (s,t) in [0,n]^2
      (*P)[i].col(k) << x, y, z;
    }
  }
  fclose(file);
  return;
}

std::vector<PolynomialPatch> LoadGeometryFilePoly(
    const std::string &file_name) {
  std::vector<PolynomialPatch> retval;
  std::vector<Eigen::MatrixXd> PE;
  int p = 0;
  int m = 0;
  readPointsAscii(&PE, &p, &m, file_name);
  int i = 0;
  for (auto i = 0; i < p; ++i)
    retval.push_back(PolynomialPatch(PE[i], (1 << m) + 1, 3, -1, -1));
  return retval;
}

}  // namespace Bembel
#endif  // BEMBEL_SRC_GEOMETRY_GEOMETRYPOLY_HPP_
