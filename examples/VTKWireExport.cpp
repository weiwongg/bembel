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

#include <Bembel/IO>

class FakeDivergenceConformingOperator;
template <>
struct Bembel::LinearOperatorTraits<FakeDivergenceConformingOperator> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum { OperatorOrder = 0, Form = DifferentialForm::DivConforming };
};

int main() {
  using namespace Bembel;
  Geometry geo("../geo/cube.bpd");

  // The refinement level for the visualization is independent of that of the
  // simulation since one might consider to visualize a coarse discretisation on
  // a smooth geometry.
  const int refinement_level = 7;

  // The VTKwriter sets up initial geomety information.
  VTKWireframeExport writer(geo, refinement_level, 3, 3);
    writer.writeToFile("write.vtp");
  return 0;
}
