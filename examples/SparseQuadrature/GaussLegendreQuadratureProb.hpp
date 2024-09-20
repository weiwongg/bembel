#ifndef __GAUSSLEGENDREQUADRATUREPROB__CLASS__
#define __GAUSSLEGENDREQUADRATUREPROB__CLASS__

#include "Eigen/Dense"

#include "UnivariateQuadrature.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

class GaussLegendreQuadratureProb : public UnivariateQuadrature {
 public:
  void initQuadrature(int maxLvl);
  void resizeQuadrature(int maxLvl);
  void testQuadrature(int maxLvl);

 protected:
  struct lvl2Deg {
    lvl2Deg(void) {
      //std::cout << "Using linear increasing Gauss Legendre quadrature with "
      //             "normalized weights\n";
    }
    int operator()(int lvl) const { return std::ceil(0.5 * lvl); };
  };
  lvl2Deg _lvl2deg;
};

#endif
