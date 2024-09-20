#ifndef __TDINDEXSET__CLASS__
#define __TDINDEXSET__CLASS__

// include for C++ sort function
#include <algorithm>

// Eigen Library
#include <Eigen/Dense>

// include base class
#include "SparseIndexSet.hpp"

#include "CONSTANTS.hpp"


class TDindexSet : public SparseIndexSet {
 public:
  // override base class methods
  void computeIndexSet(long long int q, const Eigen::VectorXd &w);

  const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &get_sortW(void) const;

 protected:
  // override base class methods
  void combiIndexSet(long long int maxBit, long long int *k, double q, Eigen::Matrix<long long int, Eigen::Dynamic, 1> &currInd);
  long long int combiWeights(double q, long long int maxBit, long long int cw, long long int lvl);

  // new methods and variables related to weight vector
  void init_sortW(void);
  void set_w(const Eigen::VectorXd &w);

  Eigen::VectorXd _w;
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> _sortW;
  double _sumW;
  // comparison functors for C++ sort routine
  template <typename T>
  struct myCompareInc {
    const T &m;
    myCompareInc(T &p) : m(p){};

    bool operator()(const long long int &i, const long long int &j) { return (m(i) < m(j)); }
  };
};
#endif

#if 0
template <typename T>
struct myCompareDec {
    const T &m;
    myCompareDec(T &p) : m(p){};
    
    bool operator()(const long long int &i, const long long int &j) { return (m(i) > m(j)); }
};
#endif
