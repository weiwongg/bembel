#include "GaussLegendreQuadratureProb.hpp"

// LAPACK prototypes
extern "C" {
extern void dsteqr_(char* compz, int* n, double* d, double* e, double* Z,
                    int* ldz, double* work, int* info);
}

//!
//!    \brief computes the Gauss Legendre quadrature on the interval [-1,1]
//!    \param[out] Q quadrature struct containing points and weights
//!                  for Gauss Legendre quadrature from degree 0 to deg
//!    \param[in] maxdeg  maximum degree of the Gauss Legendre quadrature
//!               quadrature rules from 0..maxdeg are created
//!
////////////////////////////////////////////////////////////////////////////////
void GaussLegendreQuadratureProb::initQuadrature(int maxLvl) {
  // variable for polynomial degree
  int deg = 0;
  // do something only if current _maxdeg is too small
  if (_maxLvl < maxLvl) {
    int oldLvl = 0;
    oldLvl = _maxLvl;
    _maxLvl = maxLvl;
    // stl guarantees that previous content remains. relying here on that
    _Q.resize(_maxLvl + 1);
    // Variables for LaPack
    int n = 0;
    int info = 0;
    int lwork = 0;
    Eigen::VectorXd subd;
    Eigen::VectorXd diag;
    Eigen::VectorXd work;
    Eigen::VectorXd V;

    // maximum degree
    deg = _lvl2deg(_maxLvl);
    // number of quadrature points
    n = deg + 1;
    lwork = 2 * n;  // >= (max(1,2*N-2)) (least size)
    subd.resize(n);
    diag.resize(n);
    V.resize(n * n);
    work.resize(lwork);

    // compute quadrature points and weights up to degree deg
    for (int i = oldLvl + 1; i <= _maxLvl; ++i) {
      deg = _lvl2deg(i);
      n = deg + 1;
      _Q[i].xi.resize(n);
      _Q[i].w.resize(n);
      subd.setZero();
      diag.setZero();
      V.setZero();
      work.setZero();

      // three term recurrence for Legendre polynomials
      for (int j = 1; j < n; ++j) subd(j - 1) = j / (sqrt(4. * j * j - 1));

      dsteqr_((char*)"I", &n, diag.data(), subd.data(), V.data(), &n,
              work.data(), &info);

      if (info) {
        std::cout << "\nError: init_Legendre_Quadrature failed at deg info= "
                  << i << std::endl;
        exit(info);
      }
      // extract eigenvalues and weights for the quadrature use normalized
      // weights here...
      _Q[i].xi = diag.head(n);
      for (int j = 0; j < n; ++j) {
        _Q[i].w(j) = V(n * j) * V(n * j);
      }
      if (n % 2) _Q[i].xi(n / 2) = 0.;
    }
  }
}

//!
//!    \brief checks if quadrature rules up to degree maxdeg exist
//!           if not, initQuadrature is called
//!    \param[in] maxdeg  maximum degree of the Clenshaw Curtis quadrature
//!               quadrature rules from 0..maxdeg are created
//!
////////////////////////////////////////////////////////////////////////////////
void GaussLegendreQuadratureProb::resizeQuadrature(int maxLvl) {
  if (_maxLvl < maxLvl) GaussLegendreQuadratureProb::initQuadrature(maxLvl);
}

//!
//!    \brief testing routine for the Gauss Legendre Quadrature
//!    \param[in] maxdeg quadratures from deg=0..maxdeg are instantiated
//!               and the quadrature with degree maxdeg is printed to the
//!               standard output
//!
////////////////////////////////////////////////////////////////////////////////
void GaussLegendreQuadratureProb::testQuadrature(int maxLvl) {
  double sumw = 0;
  GaussLegendreQuadratureProb::initQuadrature(maxLvl);
  for (int i = 0; i <= maxLvl; ++i) {
    sumw = 0;
    for (int j = 0; j < _Q[i].xi.size(); ++j) {
      std::cout << std::setprecision(6) << "xi = " << std::setw(9)
                << _Q[i].xi(j) << "\t"
                << "w= " << std::setw(9) << _Q[i].w(j) << std::endl;
      sumw += _Q[i].w(j);
    }
    std::cout << "\nsum of weights: " << sumw << std::endl << std::endl;
  }
}
