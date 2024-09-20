#ifndef BEMBEL_UQ_ONLINECOR_H_
#define BEMBEL_UQ_ONLINECOR_H_

namespace Bembel {
namespace UQ {

/**
 * \ingroup UQ
 * \brief computes svd decomposition of a stream of data*/

class OnlineCor {
 public:
  //////////////////////////////////////////////////////////////////////////////
  // void constructor
  OnlineCor(int num_basis, int m, int max_k, double tol = 1e-15) {
    num_sample_ = 0;
    m_ = m, max_k_ = max_k;  // m_: the number of blocks, max_k_: the maximally
                             // allowed low rank.
    tol_ = tol;
    // The Us of the SVD decompositions of blocks of the input data.
    Umatrices_.resize(m_);
    // The Simgas of the SVD decompositions of blocks of the input data.
    SigmaMatrices_.resize(m_);
    Exps_.resize(m_);
    for (auto i = 0; i < m_; ++i) {
      Exps_[i].resize(num_basis);
      Exps_[i].setZero();
    }
    /*
     when m = 3, CorMatrices is
          0    1   2
     0| C_00 C_01 C_02
     1| C_10 C_11 C_12
     2| C_20 C_21 C_22
     , where each element C_ij is a low dimentional corrlation matrix between
     the fat short V_i and V_j.
     


     Remark: The diagonal elements are identity matrices because V_i is
     orthonormal. C_ij = C_ji^T. Therefore, for the efficiency of the memory, we
     only rstore the upper triangular part.
     */
    CorMatrices_.resize(m_ * m_);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// getters
  //////////////////////////////////////////////////////////////////////////////
  const std::vector<Eigen::MatrixXd> &get_U(void) const { return Umatrices_; }

  const std::vector<Eigen::MatrixXd> &get_Sigma(void) const {
    return SigmaMatrices_;
  }

  int get_max_k(void) const { return max_k_; }

  double get_tol(void) const { return tol_; }

  Eigen::MatrixXd get_cor(int i, int j) {
    return Umatrices_[i] * SigmaMatrices_[i] * CorMatrices_[i * m_ + j] *
           SigmaMatrices_[j].transpose() * Umatrices_[j].transpose();
  }

  Eigen::MatrixXd get_LRcorblock(int i, int j) {
    Eigen::MatrixXd temp;

    temp = SigmaMatrices_[i] * CorMatrices_[i * m_ + j] *
           SigmaMatrices_[j].transpose();

    return temp;
  }

  Eigen::MatrixXd get_LRcor() {
    Eigen::MatrixXd LRcor(m_ * max_k_, m_ * max_k_);
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < m_; ++j) {
        LRcor.block(i * max_k_, j * max_k_, max_k_, max_k_) =
            SigmaMatrices_[i] * CorMatrices_[i * m_ + j] *
            SigmaMatrices_[j].transpose();
      }
    }
    LRcor = LRcor.cwiseQuotient(LRcor.diagonal().cwiseSqrt() *
                                LRcor.diagonal().cwiseSqrt().transpose());
    return LRcor;
  }

  //////////////////////////////////////////////////////////////////////////////
  /*
   *    \brief update the SVD decomposition after adding the new samples D
   */
  void update(Eigen::MatrixXd D, Eigen::VectorXd W) {
    IO::Stopwatch sw;
    // initialization of the residual matrix res
    // Eigen::MatrixXd res = D;
    int num = D.cols();
    int dim = D.rows() / m_;
    std::vector<Eigen::MatrixXd> Vprimes(m_);
    for (int i = 0; i < m_; ++i) {
      Eigen::MatrixXd res = D.block(i * dim, 0, dim, num);
      if (num_sample_ != 0) {
        Exps_[i] = Exps_[i] + res * W;
        int k = Umatrices_[i].cols();
        // initialization of the coefficient matrix c
        Eigen::MatrixXd c = Eigen::MatrixXd::Zero(k, num);
        // orthogonalize new sample agains existing U twice
        for (int count = 0; count < 2; ++count) {
          c = c + Umatrices_[i].transpose() * res;
          res = D.block(i * dim, 0, dim, num) - Umatrices_[i] * c;
        }
        if (res.norm() > tol_) {
          // QR decomposition of the residual
          Eigen::MatrixXd Q;
          Eigen::MatrixXd R;
          my_qr(res, &Q, &R);
          // padding the Umatrix_ and the SigmaMatrix_
          Umatrices_[i].conservativeResize(dim, k + Q.cols());
          Umatrices_[i].topRightCorner(dim, Q.cols()) = Q;
          SigmaMatrices_[i].conservativeResize(k + R.rows(), k + num);
          SigmaMatrices_[i].topRightCorner(k, num) = c;
          SigmaMatrices_[i].bottomLeftCorner(R.rows(), k) =
              Eigen::MatrixXd::Zero(R.rows(), k);
          SigmaMatrices_[i].bottomRightCorner(R.rows(), num) = R;
        } else {
          SigmaMatrices_[i].conservativeResize(k, k + num);
          SigmaMatrices_[i].topRightCorner(k, num) = c;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            SigmaMatrices_[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
        SigmaMatrices_[i] = svd.singularValues().asDiagonal();
        Umatrices_[i] = Umatrices_[i] * svd.matrixU();
        Vprimes[i] = svd.matrixV();
      } else {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_init(
            res, Eigen::ComputeThinU | Eigen::ComputeThinV);
        SigmaMatrices_[i] = svd_init.singularValues().asDiagonal();
        Umatrices_[i] = svd_init.matrixU();
        Vprimes[i] = svd_init.matrixV();
        Exps_[i] = Exps_[i] + res * W;
      }
      // re-orthogonalizing Umatrices_
      int tmp = Umatrices_[i].cols();
      sw.tic();
      Eigen::MatrixXd UQ;
      Eigen::MatrixXd UR;
      my_qr(Umatrices_[i], &UQ, &UR);
      Umatrices_[i] = UQ;
      SigmaMatrices_[i] = UR * SigmaMatrices_[i];
      Eigen::JacobiSVD<Eigen::MatrixXd> svd2(
          SigmaMatrices_[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
      SigmaMatrices_[i] = svd2.singularValues().asDiagonal();
      Umatrices_[i] = Umatrices_[i] * svd2.matrixU();
      Vprimes[i] = Vprimes[i] * svd2.matrixV();
      if (SigmaMatrices_[i].cols() > max_k_) {
        Umatrices_[i].conservativeResize(Umatrices_[i].rows(), max_k_);
        SigmaMatrices_[i].conservativeResize(max_k_, max_k_);
      }
    }

    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < m_; ++j) {
        int idx = i * m_ + j;
        int k1 = CorMatrices_[idx].rows();
        int k2 = CorMatrices_[idx].cols();
        if (num_sample_ != 0) {
          CorMatrices_[idx].conservativeResize(k1 + num, k2 + num);
          CorMatrices_[idx].topRightCorner(k1, num) =
              Eigen::MatrixXd::Zero(k1, num);
          CorMatrices_[idx].bottomLeftCorner(num, k2) =
              Eigen::MatrixXd::Zero(num, k2);
          CorMatrices_[idx].bottomRightCorner(num, num) = W.asDiagonal();
          CorMatrices_[idx] =
              Vprimes[i].transpose() * CorMatrices_[idx] * Vprimes[j];
        } else {
          CorMatrices_[idx] =
              Vprimes[i].transpose() * W.asDiagonal() * Vprimes[j];
        }
        if (CorMatrices_[idx].rows() > max_k_)
          CorMatrices_[idx].conservativeResize(max_k_,
                                               CorMatrices_[idx].cols());
        if (CorMatrices_[idx].cols() > max_k_)
          CorMatrices_[idx].conservativeResize(CorMatrices_[idx].rows(),
                                               max_k_);
        // std::cout<<"===="<<std::endl;
      }
    }
    num_sample_ += num;
  }
  void my_qr(const Eigen::MatrixXd &A, Eigen::MatrixXd *Q, Eigen::MatrixXd *R) {
    int dim = A.rows();
    int num = A.cols();
    int k = 1;
    Q->conservativeResize(dim, 1);
    Q->block(0, 0, dim, 1) = A.col(0) / A.col(0).norm();
    R->conservativeResize(1, 1);
    (*R)(0, 0) = A.col(0).norm();
    for (int i = 1; i < num; ++i) {
      // initialization of the coefficient matrix c and the residual res
      Eigen::MatrixXd c = Eigen::VectorXd::Zero(k);
      Eigen::MatrixXd res = A.col(i);
      // orthogonalize new sample agains existing Q twice
      for (int j = 0; j < 2; ++j) {
        c = c + Q->transpose() * res;
        res = A.col(i) - (*Q) * c;
      }
      R->conservativeResize(k, i + 1);
      R->block(0, i, k, 1) = c;
      if (res.norm() > 1e-10) {
        Q->conservativeResize(dim, k + 1);
        Q->block(0, k, dim, 1) = res / res.norm();
        R->conservativeResize(k + 1, i + 1);
        R->block(k, 0, 1, i + 1) = Eigen::MatrixXd::Zero(1, i + 1);
        (*R)(k, i) = res.norm();
        k += 1;
      }
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  double tol_;
  int num_sample_;
  int m_;
  int max_k_;
  std::vector<Eigen::MatrixXd> Umatrices_;
  std::vector<Eigen::MatrixXd> SigmaMatrices_;
  std::vector<Eigen::VectorXd> Exps_;
  std::vector<Eigen::MatrixXd> CorMatrices_;
};

}  // namespace UQ
}  // namespace Bembel

#endif
