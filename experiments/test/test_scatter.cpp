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

// C++
#include <fstream>
#include <iostream>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

// Bembel
#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/IO>
#include <Bembel/Identity>
#include <Bembel/LinearForm>
#include <Bembel/UQ>
#include <Bembel/src/util/FormalSum.hpp>
#include <Bembel/src/util/GenericMatrix.hpp>

// Random domain
#include "RandomDomain/Auxilliary.h"

// other helpers
#include "Util/Grids.hpp"
#include "Util/Inputs.hpp"

// Parallel
#include <omp.h>

/**
 *  \ingroup Examples
 *  \defgroup LaplaceSingleLayerFull Laplace Single Layer Full
 *  \brief This example computes the solution of a Laplace problem.
 *
 *  Consider the following Dirichlet problem.
 *  Let \f$\Omega\f$ be a compact domain with boundary \f$\Gamma\f$.
 *  Given a function \f$g\f$ on \f$\Gamma\f$, find a function \f$u\f$ on
 *  \f$\Omega\f$ such that,
 *
 *  \f{aligned}{
 *    \Delta u &= 0\quad\mathrm{in}\, \Omega \\
 *    u &= f\quad\mathrm{on}\, \Gamma.
 *  \f}
 *
 *  The solution can be computed with the representation formulae
 *
 *  \f{aligned}{
 *    u(x) = \tilde V(\omega)(x)\quad\mathrm{in}\,\Omega,
 *  \f}
 *
 *  where
 *
 *  \f{aligned}{
 *    \tilde V(\mu)(x) = \int_\Gamma \frac{\mu(y)}{4\pi\|x - y\|}\,\mathrm{d}y
 *  \f}
 *
 *  is the Laplace single layer potential with some density \f$\mu\f$.
 *  Utilizing the trace operator \f$\gamma\f$, the single layer operator is
 *  defined by \f$V = \gamma\circ\tilde V\f$. When discretized in a conformal
 *  finite-dimensional function space \f$\mathbb{S}\f$, the variational
 *  formulation is as follows: Find \f$\omega\in\mathbb{S}\f$, such that
 *
 *  \f{aligned}{
 *    \int_\Gamma \mu \int_\Gamma V(\omega)(x)\,\mathrm{d}x = \int_\Gamma \mu
 *  f\,\mathrm{d}x\quad\forall\mu\in \mathbb{S} \\ \f}
 *
 *holds, which is realized in this example.
 */

int main() {
  using namespace Bembel;
  using namespace Eigen;
  Bembel::IO::Stopwatch sw;

  int polynomial_degree_max = 4;
  int refinement_level = 1;

  double sphere_radius = 5.0;

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  // Geometry geometry("../geo/sphere8.bpd");
  Geometry geometry;
  sw.tic();
  Bembel::UQ::GeometryDeformer def("../../geo/david10.bpd", 2.0,
                                   Eigen::Vector3d(0, 0, -0.3));
  std::cout << "build geometry deformer in " << std::setprecision(4) << sw.toc()
            << "s" << std::endl;
  int dim = def.get_parameterDimension();
  std::cout << "UQ dimension = " << def.get_parameterDimension() << std::endl;
  srand((unsigned int)0);

  //Eigen::VectorXd max_perb =
  //    def.get_maxPerturbation(Bembel::Constants::ROUGH_SCALE);
  //std::cout << "max perturbation: " << max_perb.maxCoeff() << std::endl;
  for (int i = 0; i < 10; ++i) {
    Eigen::VectorXd param = Eigen::MatrixXd::Random(dim, 1);
    std::cout << "Here is mat.sum():       " << param.sum() << std::endl;
    geometry =
        def.get_geometryRealization(param, Bembel::Constants::ROUGH_SCALE);

    VTKSurfaceExport writer(geometry, 6);
    writer.writeToFile("geo.vtp");
    std::complex<double> wavenumber(Bembel::Constants::BEMBEL_WAVENUM, 0.);

    // Define evaluation points for potential field, a tensor product grid of
    // 7*7*7 points in [-.1,.1]^3
    MatrixXd gridpoints = Util::makeSphereGrid(sphere_radius, 100);
    VectorXcd pot_truth;
    {
      std::vector<MatrixXcd> pot =
          integrand(geometry, gridpoints, refinement_level,
                    polynomial_degree_max + 1, wavenumber);
      pot_truth = pot[0];
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    // Iterate over polynomial degree.
    for (int polynomial_degree = 0;
         polynomial_degree < polynomial_degree_max + 1; ++polynomial_degree) {
      // Iterate over refinement levels
      sw.tic();

      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      std::vector<MatrixXcd> pot =
          integrand(geometry, gridpoints, refinement_level, polynomial_degree,
                    wavenumber);

      // compute reference, print time, and compute error
      VectorXcd pot_ref = pot[0];
      double error = (pot_ref - pot_truth).cwiseAbs().maxCoeff();
      std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
      std::cout << error << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
  }
  return 0;
}
