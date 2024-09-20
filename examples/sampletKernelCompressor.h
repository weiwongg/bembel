// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_L1PAPER_SAMPLETKERNELCOMPRESSOR_
#define FMCA_L1PAPER_SAMPLETKERNELCOMPRESSOR_

#include <FMCA/src/util/Macros.h>
#include <FMCA/src/util/Tictoc.h>

#include <Eigen/Sparse>
#include <FMCA/CovarianceKernel>
#include <FMCA/Samplets>

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

template <typename Triplet>
void sortTripletsInPlace(std::vector<Triplet> &trips) {
  struct customLess {
    bool operator()(const Triplet &a, const Triplet &b) const {
      if (a.row() == b.row())
        return a.col() < b.col();
      else
        return a.row() < b.row();
    }
  };
  std::sort(trips.begin(), trips.end(), customLess());
  return;
}

std::vector<Eigen::Triplet<FMCA::Scalar>> sampletKernelCompressor(
    const FMCA::CovarianceKernel &kernel, const H2SampletTree &hst,
    const MatrixEvaluator &mat_eval, const FMCA::Matrix &P, FMCA::Scalar eta,
    FMCA::Scalar threshold) {
  const FMCA::Index npts = hst.indices().size();
  std::vector<Eigen::Triplet<FMCA::Scalar>> trips;
  FMCA::Tictoc T;
  T.tic();
  {
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
    Scomp.init(hst, eta, threshold);
    T.toc("planner:                     ");
    T.tic();
    Scomp.compress(mat_eval);
    T.toc("compressor:                  ");
    T.tic();
    Scomp.triplets();
    trips = Scomp.release_triplets();
    sortTripletsInPlace(trips);
    T.toc("triplets:                    ");
  }
  std::cout << "anz:                          "
            << std::round(trips.size() / FMCA::Scalar(npts)) << std::endl;
  std::cout << "size of a triplet:            "
            << sizeof(Eigen::Triplet<FMCA::Scalar>) << "Byte(s)" << std::endl;
  std::cout << "storage of matrix:            "
            << trips.size() / 1e9 * sizeof(Eigen::Triplet<FMCA::Scalar>) << "GB"
            << std::endl;
  // compute compression error
  FMCA::Vector x(npts), y1(npts), y2(npts);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (FMCA::Index i = 0; i < 20; ++i) {
    FMCA::Index index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    FMCA::Vector col = kernel.eval(P, P.col(hst.indices()[index]));
    y1 = col;
    for (FMCA::Index j = 0; j < y1.size(); ++j) y1(j) = col(hst.indices()[j]);
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto &i : trips) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  err = sqrt(err / nrm);
  std::cout << "compression error:            " << err << std::endl;
  return trips;
}

#endif

#if 0
trips.reserve(2 * trips.size() - npts);
const FMCA::Index old_size = trips.size();
for (FMCA::Index i = 0; i < old_size; ++i)
  if (trips[i].row() != trips[i].col())
    trips.push_back(Eigen::Triplet<FMCA::Scalar>(trips[i].col(), trips[i].row(),
                                                 trips[i].value()));
#endif
