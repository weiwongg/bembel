#ifndef __SPARSEQUADRATURE__CLASS__
#define __SPARSEQUADRATURE__CLASS__

#include <algorithm>

#include <Eigen/Dense>

#include "CONSTANTS.hpp"

#include "SparseIndexSet.hpp"
#include "TensorProductQuadrature.hpp"
#include "UnivariateQuadrature.hpp"

class SparseQuadrature {
 public:
  SparseQuadrature(void){};
  SparseQuadrature(const SparseIndexSet &spInd, UnivariateQuadrature &Q);
  void computeSparseQuadrature(const SparseIndexSet &spInd,
                               UnivariateQuadrature &Q);
  void purgeSparseQuadrature(void);
  const Eigen::MatrixXd &get_qPoints(void) const;
  const Eigen::VectorXd &get_qWeights(void) const;

 protected:
  Eigen::MatrixXd _qPoints;
  Eigen::VectorXd _qWeights;

  struct lexiCompareInd {
    const double *array;
    long long int N, M;
    lexiCompareInd(const Eigen::MatrixXd &A)
        : array(A.data()), N((long long int)A.rows()), M((long long int)A.cols()){};

    bool operator()(const long long int &i, const long long int &j) {
      assert(i < M && j < M && "Geisreiche assert-message");
      return std::lexicographical_compare(array + i * N, array + i * N + N,
                                          array + j * N, array + j * N + N);
    }
  };

  bool isEqual(const Eigen::VectorXd &vec1, const Eigen::VectorXd &vec2) {
    if (vec1.size() != vec2.size())
      return false;
    else if ((vec1 - vec2).lpNorm<Eigen::Infinity>() < __PRECISION__)
      return true;
    else
      return false;
  }
};

#endif
