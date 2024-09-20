// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_GEOMETRY_PATCHVECTOR_H_
#define BEMBEL_GEOMETRY_PATCHVECTOR_H_

namespace Bembel {
/**
 *  \ingroup Geometry
 *  \brief typedef for PatchVector
 *
 *  This is the default geometry format. If custom parametric mappings were to
 *  be implemented, one would need to change Bembel::Patch here to the new
 *format, and the new format must provide an eval(), evalNorma(), evalJacobian()
 *and updateSurfacePoint() method.
 **/
class NURBSPatch;
class PolynomialPatch;

typedef std::vector<PolynomialPatch> PatchVector;
}  // namespace Bembel
#endif
