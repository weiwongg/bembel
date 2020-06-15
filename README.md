<!-- This file is part of Bembel, the higher order C++ boundary element library.
It was written as part of a cooperation of J. Dölz, H. Harbrecht, S. Kurz, 
M. Multerer, S. Schöps, and F. Wolf at Technische Universitaet Darmstadt, 
Universitaet Basel, and Universita della Svizzera italiana, Lugano. This 
source code is subject to the GNU General Public License version 3 and 
provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further 
information. -->
# Bembel
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/temf/bembel)
[![DOI](https://zenodo.org/badge/173278911.svg)](https://zenodo.org/badge/latestdoi/173278911)
[![CircleCI](https://circleci.com/gh/temf/bembel/tree/master.svg?style=shield)](https://circleci.com/gh/temf/bembel/tree/master)
## Table of contents
1. [Introduction](#introduction)
2. [What is a Bembel?](#whatis)
3. [Features](#features)
4. [How to Run our Code](#example)
5. [Structure of the Repository](#structure)
6. [Documentation](#doc)
7. [Known Bugs and Upcoming Features](#bugs)
8. [Publications, Preprints, and how to cite](#publications)
9. [Contributers](#contributors)
10. [About the People](#people)
11. [Funding](#funding)

## 1. Introduction <a name="introduction"></a>

Bembel is the 
Boundary Element Method Based Engineering Library 
written in C++ to solve boundary value problems governed by the Laplace,
Helmholtz or electric wave equation within the isogeometric framework [3,4,5,6]. 
It was developed as part of a cooperation between the TU Darmstadt and the 
University of Basel, coordinated by [H. Harbrecht](#HeHa), [S. Kurz](#SK) 
and [S. Schöps](#SSc). The code is based on the Laplace BEM of [J. Dölz](#JD),
 [H. Harbrecht](#HeHa), and [M. Multerer](#MM), [2,7] as well as the spline 
and geometry framework of [F. Wolf](#FW). The code was extended by 
[J. Dölz](#JD) and [F. Wolf](#FW) in early 2018 to cover electromagnetic 
applications [5,6]. 
A short introduction to the code has been written [3]. 
If you utilise our code as part of a publication, we would appreciate it if you cite it. 

## 2. What is a Bembel?<a name="whatis"></a>

A traditional German ceramic, as depicted in our logo. 
Quoting [Wikipedia](https://en.wikipedia.org/wiki/Apfelwein):

> *Most establishments also serve Apfelwein by the Bembel (a specific Apfelwein
> jug), much like how beer can be purchased by the pitcher in many countries. The
> paunchy Bembel (made from salt-glazed stoneware) usually has a basic grey colour
> with blue-painted detailing.*

## 3. Features <a name="features"></a>

Current key features include

* Header-only implementation,
* Already implemented Laplace, Helmholtz and Maxwell single layer operator,
easily expendable to other operators,
* Arbitrary parametric mappings for the geometry representation, by default
realized as NURBS-mappings from files generated by the
[NURBS package](https://octave.sourceforge.io/nurbs/),
* Arbitrary-order B-Spline functions as Ansatz spaces, as in
the framework of isogeometric analysis for electromagnetics [1,5],
* An embedded interpolation-based fast multipole method for compression [2,5],
equivalent to the H2 matrix format,
* openMP parallelized matrix assembly,
* Full compatibility with the [Eigen](http://eigen.tuxfamily.org/) linear algebra library. 
If you are not familiar with Eigen3, we emphasize that you can use it similarly
to Matlab or Octave, 
see [here](http://eigen.tuxfamily.org/dox/AsciiQuickReference.txt).

## 4. How to Run our Code <a name="example"></a>

You need to install the Eigen3 library, see [Eigen's Documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html) for help. 
We do not rely on any other external libraries, except for the standard template
library. Thus, having installed Eigen, Bembel should run out of the box. If you want to use
Bembel as part of your application, simply add the `Bembel/` directory to your includes.

If you want to run the provided examples and tests, you can utilize the procided  `CMakeLists.txt`.
We recommend setting up a `build/` directory 
in the root directory of the Bembel, to switch to it and then to call `cmake ..` and `make debug` 
or `make release`, respectively. 

The CMake-File checks for installations of Eigen via the 
corresponding `Eigen3Config.cmake`. On Unix you may 
run `apt install libeigen3-dev`, or on Mac with 
Homebrew `brew install eigen`, and everything should work. 
Alternatively, you can delete the line
`find_package (Eigen3 3.3 REQUIRED NO_MODULE)`
and all lines of the type
`target_link_librarires(... Eigen3::Eigen)`
from `CMakeLists.txt`, and give a path to the eigen headers as an include directory manually.

Then, the examples and tests should compile without any issues. You may run all tests by calling
`make test` from the `build/` directory after a successful compilation.

The geometry files required to run the examples can be found in the `geo/` folder.

## 5. Structure of the Repository <a name="structure"></a>

The general structure of the repository looks as follows.

* The root directory contains some helpful shell scripts.
* `assets/` only contains things relevant for GitHub pages.
* `geo/` contains geometry files in the format of the octave 
[NURBS package](https://octave.sourceforge.io/nurbs/). They can be utilized for computations. Note that 
**the normal vector must be outward directed** on all patches!
  * `geo/octave_example/` includes `.m` files that showcase how geometries can be
  constructed using the [NURBS package](https://octave.sourceforge.io/nurbs/) 
  of octave.
* `Bembel/` contains the library. In `Bembel/`, module files have been created analogously to the 
design of the Eigen library. If Bembel is used, only these module files should be included. These modules are the following:
    * `AnsatzSpace` contains the files whose routines manage the discrete space on the surface of the geometry. Specifically, this is realised through the four classes `Superspace`, `Projector`, `Glue`, and `AnsatzSpace`. Therein, the `Superspace` manages local polynomial bases on every element. Through a transformation matrix generated by the template class `Projector` which depends on the `LinearOperator` and its defined traits, the `Superspace` can be related to B-Spline bases on every patch. To build conforming spaces (in the case of `DifferentialForm::DivergenceConforming` through continuity of the normal component across patch interfaces, and in the case of `DifferentialForm::Continuous` through global C0-continuity), the template class `Glue` assembles another transformation matrix to identify degrees of freedom across edges. Then, a coefficient vector in the `Superspace` can be related to one of the smooth B-Spline basis, as explained in [4,5].
    * `ClusterTree` implements an element structure for refinement. Currently, only uniform refinement has been implemented. A `ClusterTree` object itself is comprised of a `Geometry` object and an `ElementTree`. The `ElementTree` provides routines for the extraction of topological information, refinement, as well as iterators to effortlessly loop over elements.
    * `DuffyTrick` provides quadrature routines for (nearly) singular integrals. Therein, quadrature routines for 4 cases are implemented: Separated elements, elements sharing a corner, elements sharing an edge, and identical elements. 
    * `DummyOperator` provides a `LinearOperator` for testing.
    * `Geometry`  implements the handling of the geometry.
    * `H2Matrix` handles the compression of the large, densely populated matrices. The algorithm was introduced in [2] and was generalized to the Maxwell case in [5].
    * `Helmholtz` provides the necessary specializations to solve Helmholtz problems.
    * `IO` provides input-output functionality, including routines for VTK file export, timing, and writing log files.
    * `Laplace`  provides the necessary specializations to solve Laplace problems.
    * `LinearForm` implements trace operators, i.e., routines to generate the right hand side of the linear systems. Currently, only a Dirichlet trace and a rotated tangential trace are provided.
    * `LinearOperator` provides a framework to implement linear operators that can be used to solve PDEs. Therein, the kernel function, a routine for the evaluation, and certain traits (like the scalar type, the operator order, and the `DifferentialForm`, i.e., weather the basis must be discontinuous, vector-valued and divergence conforming, or globally continuous) required for discretization must be provided.
    * `Maxwell` provides the necessary specializations to solve Maxwell problems.
    * `Potential` provides routines for the evaluation of the `Potential`, i.e., the solution to the PDE via the solution of (non-singular) integrals based on the `LinearOperator`s. 
    * `Quadrature`
    * `Spline` provided basic routines related to spline function and local polynomials.
    
The `Bembel/src/` directory leads to folders corresponding to these modules, which include the actual implementation.
The folder `examples/` contains short and *well documented* examples for Bembels functionality and `tests/` contains the tests for Bembel.

## 6. Documentation <a name="doc"></a>

A good place to start are the examples in the `examples/` folder, together with publication [3]. Apart from that, a [Doxygen documentation](https://temf.github.io/bembel/Doxy_out/html/index.html) is available.

## 7. Known Bugs and Upcoming Features <a name="bugs"></a>

For a list of known bugs and upcoming features, please have a look at 
the issue tracker on github.

## 8. Publications, Preprints, and how to cite <a name="publications"></a>

The following articles and preprints influenced the development of Bembel. We appreciate a citation of [3] if you use it in one of your articles.

[1] A. Buffa, J. Dölz, S. Kurz, S. Schöps, R. Vázques, and F. Wolf. 
*Multipatch approximation of the de Rham sequence and its traces in isogeometric analysis.* In: Numer. Math., 144, 201-236, 2020. [To the paper](https://link.springer.com/article/10.1007/s00211-019-01079-x). [To the preprint](https://arxiv.org/abs/1806.01062).

[2] J. Dölz, H. Harbrecht, and M. Peters. *An interpolation-based fast multipole
method for higher-order boundary elements on parametric surfaces*. In: Int. J. Numer. Meth. Eng., 108(13):1705-1728, 2016.
[To the paper](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.5274).

[3] J. Dölz, H. Harbrecht, S. Kurz, M. Multerer, S. Schöps, and F. Wolf. *Bembel: The Fast Isogeometric Boundary Element C++ Library for Laplace, Helmholtz, and Electric Wave Equation*. In: SoftwareX, 11, 10476. [To the paper](https://doi.org/10.1016/j.softx.2020.100476).

[4] J. Dölz, H. Harbrecht, S. Kurz, S. Schöps, and F. Wolf. *A fast isogeometric
BEM for the three dimensional Laplace- and Helmholtz problems*. In: Comput. Methods Appl. Mech. Engrg., 330:83-101, 2018. 
[To the paper](https://www.sciencedirect.com/science/article/pii/S0045782517306916). 
[To the preprint](https://arxiv.org/abs/1708.09162).

[5] J. Dölz, S. Kurz, S. Schöps, and F. Wolf. *Isogeometric Boundary Elements in 
Electromagnetism: Rigorous Analysis, Fast Methods, and Examples*. In: SIAM J. Sci. Comput., 41(5), B983-B1010, 2019. [To the paper](https://epubs.siam.org/doi/abs/10.1137/18M1227251). [To the preprint](https://arxiv.org/abs/1807.03097).

[6] J. Dölz, S. Kurz, S. Schöps, and F. Wolf. *A Numerical Comparison of an Isogeometric and a Parametric Higher Order Raviart–Thomas Approach to the Electric Field Integral Equation*. In: IEEE Trans. Antenn. Prop., 68(1),
593–597, 2020. [To the paper](https://ieeexplore.ieee.org/abstract/document/8809890). [To the preprint](https://arxiv.org/abs/1807.03628).

[7] H. Harbrecht and M. Peters. *Comparison of fast boundary element methods on
parametric surfaces*. In: Comput. Methods Appl. Mech. Engrg., 261-262:39-55, 2013. 
[To the paper](https://www.sciencedirect.com/science/article/pii/S0045782513000819).

## 9. Contributors <a name="contributors"></a>

Contributors include (alphabetically): 
*   D. Andric, 
*   J. Corno, 
*   [J. Dölz](#JD), 
*   [H. Harbrecht](#HeHa), 
*   [M. Multerer](#MM), 
*   [F. Wolf](#FW).

## 10. About the People <a name="people"></a>

* [Jürgen Dölz](https://ins.uni-bonn.de/staff/doelz) 
<a name="JD"></a> currently holds an assistant professorship at the [University of Bonn](https://www.uni-bonn.de/). He may also be found 
[on GitHub](https://github.com/jdoelz).

* [Helmut Harbrecht](https://cm.dmi.unibas.ch/) 
<a name="HeHa"></a> currently holds a professorship at the 
[Departement Mathematik und Informatik](https://dmi.unibas.ch/de/home/) 
at the University of Basel.

* [Stefan Kurz](https://www.temf.tu-darmstadt.de/temf/mitarbeiter/mitarbeiterdetails_57408.en.jsp)
<a name="SK"></a> currently holds a professorship at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) 
at TU Darmstadt and is Vice President for Hybrid Modelling at Robert Bosch GmbH.

* [Michael Multerer](https://www.ics.usi.ch/index.php/people-detail-page/297-prof-michael-multerer) 
<a name="MM"></a> currently holds a professorship 
at the Institute of Computational Science at the Università della Svizzera italiana in Lugano.
He may also be found [on GitHub](https://github.com/muchip).

* [Sebastian Schöps](https://www.cem.tu-darmstadt.de/cem/group/ref_group_details_27328.en.jsp
)<a name="SSc"></a> currently holds a professorship at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) 
at TU Darmstadt. He may also be found [on GitHub](https://github.com/schoeps).

* [Felix Wolf](https://www.cem.tu-darmstadt.de/cem/group/ref_group_details_57665.en.jsp) 
<a name="FW"></a>is currently a postdoc at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) at TU Darmstadt. He may also be found 
[on GitHub](https://github.com/flx-wlf).


## 11. Funding <a name="funding"></a>

Work on Bembel was conducted with the following financial support (alphabetically):
*  DFG Grant KU1553/4-1,
*  DFG Grant SCHO1562/3-1,
*  The Graduate School of Computational Engineering at TU Darmstadt and the Excellence Initiative of the German Federal and State Governments and the Graduate School of Computational Engineering at TU Darmstadt,
*  SNSF Grant 137669,
*  SNSF Grant 156101,
*  SNSF Grant 174987. 
