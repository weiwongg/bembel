#ifndef __HALTONSET__CLASS__
#define __HALTONSET__CLASS__

#include <iostream>
#include <Eigen/Dense>

class HaltonSet {
 public:
  /** \brief first non-void constructor. specifies the dimension. the number of
  *          skipped entries (warm up) and the precision of the point
  *          comparison
  *          are set to default values
  */
  HaltonSet(int M);
  /** \brief second non-void constructor. specifies the dimension and the number
  *          of skipped entries (warm up). the precision of the point comparison
  *          is set to default value.
  */
  HaltonSet(int M, int skip);
  /** \brief computes the next entry of the Halton sequence starting from the
  *          current state of _bAdic by adding 1 and compting the radical
  *          inverse of the related index by the Horner scheme. we have chosen
  *          long double precision here since it allows for a more robust
  *          computation also for larger indices.
  */
  void next(void);
  /** \brief resets the Halton sequence to the index _skip
  */
  void reset(void);
  /** \brief returns the current state of _HaltonVector as a const reference
  */
  const Eigen::VectorXd &get_HaltonVector(void) const;

 private:
  /** \brief initializes the HaltonVector with respect to the warm up defined
  *          in _skip
  */
  void init_HaltonVector(void);
  static const int _prim[1000];
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _bAdic;
  Eigen::VectorXi _maxDigit;
  int _skip;
  int _M;
  Eigen::VectorXd _HaltonVector;
};

#endif
