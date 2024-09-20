// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_GEOMETRY_INITPATCHVECTOR_H_
#define BEMBEL_GEOMETRY_INITPATCHVECTOR_H_

namespace Bembel {
/**
 *  \ingroup Geometry
 *  \brief helper function to get an abstraction of init_PatchVector in
 *         Geometry
 **/
template <typename PatchType>
std::vector<PatchType> init_PatchVector(const std::string &file_name,
                                        double scale = 1.) {
  return PatchVector();
}

template <typename PatchType>
std::vector<PatchType>
init_PatchVectorFromPoints(const std::vector<Eigen::MatrixXd> &PE,
                           double scale = 1.) {
  return PatchVector();
}

template <>
std::vector<NURBSPatch>
init_PatchVector<NURBSPatch>(const std::string &file_name, double scale) {
  return Bembel::PatchShredder(Bembel::LoadGeometryFile<NURBSPatch>(file_name));
}

template <>
std::vector<PolynomialPatch>
init_PatchVector<PolynomialPatch>(const std::string &file_name, double scale) {
  std::vector<PolynomialPatch> retval;
  std::vector<Eigen::MatrixXd> PE;
  int p = 0;
  int m = 0;
  Bembel::IO::readPointsAscii(&PE, &p, &m, file_name);
  std::cout << "scale = " << scale << std::endl;
  int i = 0;
  for (auto i = 0; i < p; ++i)
    retval.push_back(PolynomialPatch(PE[i], (1 << m) + 1,
                                     BEMBEL_SURFACE_INTERPOLATION_DEGREE_, -1,
                                     -1, scale));
  return retval;
}

template <>
std::vector<PolynomialPatch> init_PatchVectorFromPoints<PolynomialPatch>(
    const std::vector<Eigen::MatrixXd> &PE, double scale) {
  std::vector<PolynomialPatch> retval;
  int p = PE.size();
  for (auto i = 0; i < p; ++i)
    retval.push_back(PolynomialPatch(PE[i], sqrt(PE[i].cols()),
                                     BEMBEL_SURFACE_INTERPOLATION_DEGREE_, -1,
                                     -1, scale));
  return retval;
}

} // namespace Bembel

#endif
