// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef __BEMBEL_GRIDS_
#define __BEMBEL_GRIDS_
#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace Bembel {
namespace Util {

inline Eigen::Matrix<double, Eigen::Dynamic, 3> makeTensorProductGrid(
    Eigen::VectorXd X, Eigen::VectorXd Y, Eigen::VectorXd Z) {
  const int maxX = std::max(X.rows(), X.cols());
  const int maxY = std::max(Y.rows(), Y.cols());
  const int maxZ = std::max(Z.rows(), Z.cols());
  Eigen::Matrix<double, Eigen::Dynamic, 3> out(maxX * maxY * maxZ, 3);

  for (int iz = 0; iz < maxZ; iz++)
    for (int iy = 0; iy < maxY; iy++)
      for (int ix = 0; ix < maxX; ix++) {
        out.row(ix + iy * maxX + iz * maxY * maxX) =
            Eigen::Vector3d(X(ix), Y(iy), Z(iz));
      }
  return out;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 3> makeSphereGrid(
    const double r, const int nSample,
    const Eigen::Vector3d center = Eigen::Vector3d(0, 0, 0)) {
  double goldenRatio = (1 + sqrt(5.0)) / 2.0;
  Eigen::Matrix<double, Eigen::Dynamic, 3> out(nSample, 3);
  for (int i = 0; i < nSample; ++i) {
    double phi = acos(1 - 2.0 * (i + 0.5) / nSample);
    double theta = 2 * BEMBEL_PI * (i + 0.5) / goldenRatio;
    out.row(i) = (Eigen::Vector3d(r * cos(theta) * sin(phi),
                                  r * sin(theta) * sin(phi), r * cos(phi)) +
                  center);
  }
  return out;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 3> makePlaneGrid(
    const double r0, const double r1, const int n0, const int n1,
    const Eigen::Vector3d center = Eigen::Vector3d(0, 0, 0)) {
  Eigen::Matrix<double, Eigen::Dynamic, 3> out(n0 * n1, 3);
  const double h0 = (r1 - r0) / n0;
  const double h1 = 2 * 3.1415926 / n1;
  for (int i = 0; i < n0; i++) {
    for (int j = 0; j < n1; j++) {
      double r = r0 + h0 * i;
      out.row(j + i * n1) =
          (Eigen::Vector3d(r * cos(h1 * j), r * sin(h1 * j), 0) + center);
    }
  }

  return out;
}

}  // namespace Util
}  // namespace Bembel
#endif
