#include "TDindexSet.hpp"

/**    \brief copies the vector gamma to the class pendant and sorts it via
*             sortGamma
*      \param[in] gamma anisotropies in each dimension
*
*/
void TDindexSet::set_w(const Eigen::VectorXd &w) {
  _w = w;
  TDindexSet::init_sortW();
  // reorder _w according to sortW;
  for (long long int i = 0; i < w.size(); ++i) _w(i) = w(_sortW(i));
}

/**    \brief initializes the sparse anisotropic index set
*      \param[in] q is the maximum allowed value of the weighted sum of
*                 gamma times index
*
*/
void TDindexSet::computeIndexSet(long long int q, const Eigen::VectorXd &w) {
  long long int k = 0;
  Eigen::Matrix<long long int, Eigen::Dynamic, 1> currInd(w.size());

  // set max level for the sparse index set
  _q = q;
  // set dimension of the sparse grid
  _dim = (long long int)w.size();
  // set weight vector
  TDindexSet::set_w(w);

  _alpha.resize(_dim, __MEMCHUNKSIZE__);
  _cw.resize(__MEMCHUNKSIZE__);
  _alpha.setZero();
  _cw.setZero();
  _myOnes = Eigen::Matrix<long long int, Eigen::Dynamic, 1>::Ones(_dim);

  currInd.setZero();
  k = 0;
  // test if 0 is in Xw, else index set empty due to downward closedness
  if (0 <= _q) {
    _sumW = _w.sum();
    if (_sumW > _q) _cw(0) = TDindexSet::combiWeights(_q, 0, 1, 1);
    if (_cw(0)) ++k;
    TDindexSet::combiIndexSet(0, &k, _q, currInd);
  }

  _alpha.conservativeResize(_dim, k);
  _cw.conservativeResize(k);
}

/**    \brief sorts the values in _w with inreasing magnitude
*      \param[out] sortW permutation vector
*
*/
void TDindexSet::init_sortW(void) {
  _sortW = Eigen::Array<long long int, Eigen::Dynamic, 1>::LinSpaced((long long int)_w.size(), 0, (long long int)_w.size() - 1);

  std::sort(&(_sortW(0)), &(_sortW(0)) + _sortW.size(),
            TDindexSet::myCompareInc<Eigen::VectorXd>(_w));
}

/**    \brief computes indices in weighted sparse grid space (recursive)
*
*/
void TDindexSet::combiIndexSet(long long int maxBit, long long int *k, double q,
                               Eigen::Matrix<long long int, Eigen::Dynamic, 1> &currInd) {
  long long int cw = 0;
  double scap = 0;
  for (long long int i = maxBit; i < _dim; ++i) {
    ++currInd(i);
    q -= _w(i);
    if (q >= 0) {
      scap = _w.dot(currInd.cast<double>());
      if (scap > _q - _sumW)
        cw = TDindexSet::combiWeights(_q - scap, 0, 1, 1);
      else
        cw = 0;
      if (cw) {
        if (_alpha.cols() <= *k) {
          _alpha.conservativeResize(_dim, _alpha.cols() + __MEMCHUNKSIZE__);
          _cw.conservativeResize(_cw.size() + __MEMCHUNKSIZE__);
        }
        _alpha.col(*k) = currInd;
        _cw(*k) = cw;
        ++(*k);
      }
      TDindexSet::combiIndexSet(i, k, q, currInd);
      q += _w(i);
      --currInd(i);
    } else {  // this is the major difference to the base class, we may break
              // here due to the increasingly ordered weights
      q += _w(i);
      --currInd(i);
      break;
    }
  }
}

/**    \brief computes weights for the tensor product quadrature (recursive)
*
*/
long long int TDindexSet::combiWeights(double q, long long int maxBit, long long int cw, long long int lvl) {
  for (long long int i = maxBit; i < _dim; ++i) {
    q -= _w(i);
    if (q >= 0) {
      if (lvl % 2)
        --cw;
      else
        ++cw;
      cw = TDindexSet::combiWeights(q, i + 1, cw, lvl + 1);
      q += _w(i);
    } else {  // this is the major difference to the base class, we may break
              // here due to the increasingly ordered weights
      q += _w(i);
      break;
    }
  }

  return cw;
}

/**   \brief make an educated guess, what this function does...
*
*/
const Eigen::Matrix<long long int, Eigen::Dynamic, 1> &TDindexSet::get_sortW(void) const { return _sortW; }
