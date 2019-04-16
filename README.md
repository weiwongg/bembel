<!-- This file is part of Bembel, the higher order C++ boundary element library.
It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz, 
M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt, 
Universitaet Basel, and Universita della Svizzera italiana, Lugano. This 
source code is subject to the GNU General Public License version 3 and 
provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further 
information. -->
# Bembel

## Table of contents
1. [Introduction](#introduction)
2. [What is a Bembel?](#whatis)
3. [Features](#features)
4. [How to Run our Code](#example)
5. [Known Bugs and Upcoming Features](#bugs)
6. [Publications & Preprints](#publications)
7. [Contributers](#contributors)
8. [About the People](#people)

## 1. Introduction <a name="introduction"></a>

Bembel is the 
**B**oundary **E**lement **M**ethod **B**ased **E**ngineering **L**ibrary 
written in C and C++ to solve boundary value problems governed by the Laplace,
Helmholtz or electric wave equation within the isogeometric framework [3,4,5]. 
It was developed as part of a cooperation between the TU Darmstadt and the 
University of Basel, coordinated by [H. Harbrecht](#HeHa), [S. Kurz](#SK) 
and [S. Schöps](#SSc). The code is based on the Laplace BEM of [J. Dölz](#JD),
 [H. Harbrecht](#HeHa), and [M. Multerer](#MM), [2,6] as well as the spline 
and geometry framework of [F. Wolf](#FW). The code was extended by 
[J. Dölz](#JD) and [F. Wolf](#FW) in early 2018 to cover electromagnetic 
applications [4,5].

This code is currently at version 0.1 and a work in progress. 
It has to be polished and it has to be properly documented.

## 2. What is a Bembel?<a name="whatis"></a>

A traditional German ceramic, as depicted in our logo. 
Quoting [Wikipedia](https://en.wikipedia.org/wiki/Apfelwein):

> *Most establishments also serve Apfelwein by the Bembel (a specific Apfelwein
jug), much like how beer can be purchased by the pitcher in many countries. The
paunchy Bembel (made from salt-glazed stoneware) usually has a basic grey colour
with blue-painted detailing.*

## 3. Features <a name="features"></a>

Current key features include

* Approaches based on the Laplace, Helmholtz and Maxwell single layer operator, 
where the Maxwell approach is also known under the name *Method of Moments*,
* Arbitrary parametric mappings for the geometry representation, by default
realized as NURBS-mappings from files generated by the
[NURBS package](https://octave.sourceforge.io/nurbs/),
* Higher-order (currently up to 14) B-Spline functions as Ansatz spaces, as in
the framework of isogeometric analysis for electromagnetics [1,4],
* An embedded interpolation-based fast multipole method for compression [2,4],
equivalent to the H2 matrix format,
* openMP parallelized matrix assembly,
* Compatibility of the compressed matrix with the 
[Eigen](http://eigen.tuxfamily.org/) linear algebra library.

Planned features include octave wrappers and the full Calderon projector for 
scalar problems.

## 4. How to Run our Code <a name="example"></a>

Note that all mentioned scripts are written for bash, and thus only engineered
to work on Linux / MacOS installations.

We ship a copy of [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
as a subrepository, since its unsupported modules are not available in many 
local installations of Eigen. The submodule is automatically pulled when calling
the `compile.sh` script for the first time.
If you have received this copy of bembel as an archive and not as a 
git-repository, you may download eigen via the `get_eigen.sh`. 
Note that this will **break its functionality as a submodule** if you cloned the
repository.

If you are not familiar with Eigen3, we emphasize that you can use it similarly
to Matlab or Octave, 
see [here](http://eigen.tuxfamily.org/dox/AsciiQuickReference.txt).

We do not rely on any other external libraries, except for the standard template
library. Thus, Bembel should compile out of the box. Under Linux, you may simply
run the `compile.sh` shell script, which creates a `build/` directory and takes
care of compilation. Afterwards, we recommend running `run_tests.sh` which will
call test routines, as well as examples for the computation of a Laplace,
Helmholtz, and Maxwell problem. Afterwards, if you have a workin LaTeX
installation, you can call `run_latex.sh`, generating `.pdf`'s with nice
convergence plots. However, depending on your system, this may take a while.

The general structure of the repository looks as follows.

* The root directory contains some helpful shell scripts.
* `assets/` only contains things relevant for GitHub pages.
* `geo/` contains geometry files in the format of the octave 
[NURBS package](https://octave.sourceforge.io/nurbs/). 
They can be utilized for computations. Note that 
**the normal vector must be outward directed** on all patches!
  * `geo/octave_example/` includes `.m` files that showcase how geometries can be
  construct using the [NURBS package](https://octave.sourceforge.io/nurbs/) 
  of octave.
* `LaTeX/` contains some `.tex` files utilized by the `test.sh`-script
to visualize the examples.
* `src/` contains the source files:
  * `src/bemlibC` contains the C and C++ routines. Most of these have not yet
  been converted to Eigen and modern C++.
  * `src/include/` contains the high level `.hpp`-files, which implement the
  interface to utilize our code; as well as a copy of eigen3, and some header
  files in `includeC/` and `spline/` for the low level routines.
  * `src/examples/` contains the source code for the executables.
  * `src/test/` contains the code of some automated tests called by `run_tests.sh`.

When using our code, we recommend starting with understanding the code in
`src/examples/`, followed by learning the capabilities of the high-level classes
in `*.hpp` files, which we also consider to be the persistent API for further
releases.
You may also execute Doxygen. Many functions are documented, and the callgraphs
visualize the overall structure of the code.

You may also want to take a look at our more detailed
[Documentation](./assets/DOC.md).

## 5. Known Bugs and Upcoming Features <a name="bugs"></a>

For a list of known bugs and upcoming features, please have a look at 
the issue tracker on github.

## 6. Publications & Preprints <a name="publications"></a>

[1] A. Buffa, J. Dölz, S. Kurz, S. Schöps, R. Vázques, and F. Wolf. 
*Multipatch Approximation of the de Rham Sequence and its Traces in Isogeometric 
Analysis*. 2018. Submitted. [To the preprint](https://arxiv.org/abs/1806.01062).

[2] J. Dölz, H. Harbrecht, and M. Peters. *An interpolation-based fast multipole
method for higher-order boundary elements on parametric surfaces*. Int. J. Numer. Meth. Eng., 108(13):1705-1728, 2016.
[To the paper](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.5274).

[3] J. Dölz, H. Harbrecht, S. Kurz, S. Schöps, and F. Wolf. *A fast isogeometric
BEM for the three dimensional Laplace- and Helmholtz problems*. Comput. Methods Appl. Mech. Engrg., 330:83-101, 2018. 
[To the paper](https://www.sciencedirect.com/science/article/pii/S0045782517306916). 
[To the preprint](https://arxiv.org/abs/1708.09162).

[4] J. Dölz, S. Kurz, S. Schöps, and F. Wolf. *Isogeometric Boundary Elements in 
Electromagnetism: Rigorous Analysis, Fast Methods, and Examples*. 2018.
Submitted. [To the preprint](https://arxiv.org/abs/1807.03097).

[5] J. Dölz, S. Kurz, S. Schöps, and F. Wolf. *A Numerical Comparison of an
Isogeometric and a Classical Higher-Order Approach to the Electric Field 
Integral Equation*. 2018. Submitted. [To the preprint](https://arxiv.org/abs/1807.03628).

[6] H. Harbrecht and M. Peters. *Comparison of fast boundary element methods on
parametric surfaces*. Comput. Methods Appl. Mech. Engrg., 261-262:39-55, 2013. 
[To the paper](https://www.sciencedirect.com/science/article/pii/S0045782513000819).

## 7. Contributors <a name="contributors"></a>

**Current maintainers** are [J. Dölz](#JD), [M. Multerer](#MM), [F. Wolf](#FW).

**Other contributors** include D. Andric (geometry import), 
J. Corno (modified NURBS toolbox script),
[H. Harbrecht](#HeHa) (parts of the legacy C-codebase,
particularly the quadrature routines).

## 8. About the People <a name="people"></a>

* [Jürgen Dölz](https://www.mathematik.tu-darmstadt.de/fb/personal/details/juergen_doelz.de.jsp) 
<a name="JD"></a> currently holds a postdoc position at the 
[Department of Mathematics](https://www.mathematik.tu-darmstadt.de/fb/index.de.jsp)
at TU Darmstadt. While contributing to Bembel, he was supported by SNSF Grants 156101 
and 174987, as well as the Graduate School of Computational Engineering at TU Darmstadt
and the Excellence Initiative of the German Federal and State Governments and
the Graduate School of Computational Engineering at TU Darmstadt.

* [Helmut Harbrecht](https://cm.dmi.unibas.ch/) 
<a name="HeHa"></a> currently holds a professorship at the 
[Departement Mathematik und Informatik](https://dmi.unibas.ch/de/home/) 
at the University of Basel.

* [Stefan Kurz](https://www.temf.tu-darmstadt.de/temf/mitarbeiter/mitarbeiterdetails_57408.en.jsp)
<a name="SK"></a> currently holds a professorship at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) 
at TU Darmstadt and is a 
[research expert at Bosch](https://www.bosch.com/research/know-how/research-experts/prof-dr-stefan-kurz/).

* [Michael Multerer](https://www.ics.usi.ch/index.php/people-detail-page/297-prof-michael-multerer) 
<a name="MM"></a> currently holds a professorship 
at the Institute of Computational Science at the Università della Svizzera italiana in Lugano. 
While contributing to Bembel, he was supported by SNSF Grant 137669 until 2014. 
He may also be found [on GitHub](https://github.com/muchip).

* [Sebastian Schöps](https://www.cem.tu-darmstadt.de/cem/group/ref_group_details_27328.en.jsp
)<a name="SSc"></a> currently holds a professorship at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) 
at TU Darmstadt. He may also be found [on GitHub](https://github.com/schoeps).

* [Felix Wolf](https://www.cem.tu-darmstadt.de/cem/group/ref_group_details_57665.en.jsp) 
<a name="FW"></a>is currently a PhD student at the 
[Institute TEMF](https://www.temf.tu-darmstadt.de/temf/index.en.jsp) at TU Darmstadt. 
While contributing to Bembel, he was supported by DFG Grants SCHO1562/3-1
and KU1553/4-1, as well as the Graduate School of Computational Engineering 
at TU Darmstadt and the Excellence Initiative of the German Federal and
State Governments and the Graduate School of Computational Engineering 
at TU Darmstadt. He may also be found 
[on GitHub](https://github.com/coffeedrinkingpenguin).
