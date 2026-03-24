#include <algorithm>
#include <chrono> // Required for duration literals
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

#include <Eigen/Dense>
#include <boost/math/special_functions/erf.hpp>

#define EIGEN_DONT_PARALLELIZE

#define EPSILON std::sqrt(std::numeric_limits<double>::epsilon())
#define INF std::numeric_limits<double>::infinity()
#define NAN std::numeric_limits<double>::quiet_NaN()

namespace py = pybind11;
using MAT_DN =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MAT_DN_MAP = Eigen::Map<MAT_DN>;

using VEC_DN = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using VEC_DN_MAP = Eigen::Map<VEC_DN>;

// Declare the functions from QUIC.cpp without defining them
extern "C" {
void QUIC(char mode, uint32_t &p, const double *S, double *Lambda0,
          uint32_t &pathLen, const double *path, double &tol, int32_t &msg,
          uint32_t &maxIter, double *X, double *W, double *opt, double *cputime,
          uint32_t *iter, double *dGap, double *neg_logdetX_trSX);

void QUICR(char **modePtr, uint32_t &p, const double *S, double *Lambda0,
           uint32_t &pathLen, const double *path, double &tol, int32_t &msg,
           uint32_t &maxIter, double *X, double *W, double *opt,
           double *cputime, uint32_t *iter, double *dGap,
           double *neg_logdetX_trSX);
};

// double erfinv_approx(double x) {
//     double w, p;
//     double sign;
//     if (x >= 0) {
//         sign = 1.0;
//     }
//     else {
//         sign = -1.0;
//         x = abs(x);
//     }
//     w = -log((1.0 - x) * (1.0 + x));
//     if (w < 5.0) {
//         w = w - 2.5;
//         p = 2.81022636e-08;
//         p = 3.43273939e-07 + p * w;
//         p = -3.5233877e-06 + p * w;
//         p = -4.39150654e-06 + p * w;
//         p = 0.00021858087 + p * w;
//         p = -0.00125372503 + p * w;
//         p = -0.00417768164 + p * w;
//         p = 0.246640727 + p * w;
//         p = 1.50140941 + p * w;
//     }
//     else {
//         w = sqrt(w) - 3.000000;
//         p = -0.000200214257;
//         p = 0.000100950558 + p * w;
//         p = 0.00134934322 + p * w;
//         p = -0.00367342844 + p * w;
//         p = 0.00573950773 + p * w;
//         p = -0.0076224613 + p * w;
//         p = 0.00943887047 + p * w;
//         p = 1.00167406 + p * w;
//         p = 2.83297682 + p * w;
//     }
//     return sign * p * x;
// }

double erfinv(double x) { return boost::math::erf_inv(x); }

double trace(const MAT_DN &A) { return A.diagonal().sum(); }

double trace(const MAT_DN &A, const MAT_DN &B) {
  return (A.array() * B.array()).sum();
}

bool is_symmetric(const MAT_DN &A) { return A.isApprox(A.transpose()); }

void threshold(MAT_DN_MAP &X, double tol) {
  const size_t p = X.rows();
  assert(X.cols() == p && "Matrix must be square");

  // 1) Precompute invd[i] = 1 / sqrt(X_ii)
  VEC_DN invd = X.diagonal().array().sqrt().cwiseInverse();

  // 2) Column-by-column in-place pass (serial)
  for (size_t j = 0; j < p; ++j) {
    // Obtain a modifiable Array view of column j
    auto col_j = X.col(j).array();

    // 2a) Build normalized magnitudes:
    //     scaled[i] = |X_ij| * invd[j] * invd[i]
    Eigen::ArrayXd scaled = col_j.abs() * invd[j];
    scaled *= invd.array(); // ← use invd.array() for element-wise

    // 2b) Protect the diagonal entry from zeroing
    scaled[j] = tol + 1.0;

    // 2c) Build a 0/1 mask: 1 => drop below tol, 0 => keep
    Eigen::ArrayXd mask = (scaled < tol).template cast<double>();

    // 2d) Zero-out both X_ij and X_ji symmetrically
    col_j *= (1.0 - mask);
    X.row(j).array() *= (1.0 - mask).transpose();
  }
}

std::tuple<MAT_DN, VEC_DN> compute_S(MAT_DN_MAP &Y,
                                     const std::string &mode = "corr",
                                     const bool bias = true) {

  if (mode != "cov" && mode != "corr") {
    throw std::invalid_argument("Invalid mode: use 'cov' or 'corr'");
  }

  size_t p = Y.rows();
  size_t n = Y.cols();
  double N = bias ? double(n) : double(n - 1);

  VEC_DN mean = Y.rowwise().mean();
  MAT_DN Y_ctr = Y.colwise() - mean;
  MAT_DN S = (Y_ctr * Y_ctr.transpose()) / N;
  VEC_DN std = (S.diagonal().array() + EPSILON).sqrt();

  // Compute covariance matrix
  if (mode == "corr") {
    VEC_DN std_inv = (1.0 / std.array()).matrix();
    S = std_inv.asDiagonal() * S * std_inv.asDiagonal();
  }

  return std::make_tuple(S, std);
}

std::tuple<py::array_t<double>, // X_array,
           py::array_t<double>  // W_array,
           >
AQUIC(py::array_t<double> &Y_array, double k, double gamma, double tol,
      size_t max_iter, double L_ii = 1e-6, int verbose = 1) {

  // Extract buffer information and check dimension consistency
  auto Y_buf = Y_array.request();

  // Check dimensions first
  if (Y_buf.ndim != 2)
    throw std::runtime_error("Error: Y must be a 2D array (p x n)");
  if (Y_buf.shape[0] <= 2 || Y_buf.shape[1] <= 2)
    throw std::runtime_error("Error: p and n must be greater than 2");
  if (L_ii < 0.0)
    throw std::runtime_error("Error: L_ii must be greater than 0.0");

  size_t p = Y_buf.shape[0];
  size_t n = Y_buf.shape[1];

  // Map matrices to Eigen
  MAT_DN_MAP Y(static_cast<double *>(Y_buf.ptr), p, n);

  py::array_t<double> X_array({p, p});
  auto buf_X = X_array.request();
  MAT_DN_MAP X(static_cast<double *>(buf_X.ptr), p, p);

  py::array_t<double> W_array({p, p});
  auto buf_W = W_array.request();
  MAT_DN_MAP W(static_cast<double *>(buf_W.ptr), p, p);

  // Compute covariance matrix S and standard deviation std
  MAT_DN S;
  VEC_DN std;
  std::tie(S, std) = compute_S(Y, "corr", true);

  // Scaling Matrics
  VEC_DN scale_cov_cor =
      (1.0 / std.array()).matrix(); // scale_cov_cor.asDiagonal()   * C    *
                                    // scale_cov_cor.asDiagonal()   = Cor
  VEC_DN &scale_cor_cov = std;      // scale_cor_cov.asDiagonal()   * Cor  *
                                    // scale_cor_cov.asDiagonal()   = C
  VEC_DN &scale_icor_icov =
      scale_cov_cor; // scale_icor_icov.asDiagonal() * iCor *
                     // scale_icor_icov.asDiagonal() = iCov

  if (verbose > 0) {
    std::cout << "\n#######################################"
              << "\n## AQUIC Solver Configuration"
              << "\n#######################################"
              << "\n Y_array shape   : (" << p << ", " << n << ")"
              << "\n k               : " << k
              << "\n gamma           : " << gamma
              << "\n Tolerance       : " << tol
              << "\n L_ii            : " << L_ii
              << "\n Max Iterations  : " << max_iter
              << "\n Verbosity Level : " << verbose
              << "\n#######################################\n"
              << std::flush;
  }

  // Variance Matrix:V = S^2 + np.outer(diag(S), diag(S))
  VEC_DN diagS = S.diagonal();
  MAT_DN V = S.array().square().matrix() + diagS * diagS.transpose();
  MAT_DN L_base = V.array().sqrt(); // sqrt(V) for element-wise sqrt

  MAT_DN L(p, p);
  X.setIdentity();
  W.setIdentity();

  std::function<std::tuple<double, double, size_t, double, double>(double)>
      glasso_call = [&](double _k) {
        double factor = erfinv(1. - 2. * gamma) / std::sqrt(2. * _k);
        L = factor * L_base;
        L.diagonal().setConstant(L_ii); // Keep diagonal fixed !!

        char _mode = 'D';
        uint32_t _p = p;
        double _opt[1] = {0}; // -log|Θ| + tr(ΘS) + ||Λ ⊙ Θ||_1
        double _cputime[1] = {0};
        uint32_t _iter[1] = {0};
        double _dGap[1] = {0};
        double _neg_logdetX_trSX[1] = {0};
        uint32_t _path_len = 1;
        const double _path[1] = {1.0};
        double _tol = tol;
        uint32_t _max_iter = max_iter;
        int32_t _verbose = std::max(0, verbose - 1);

        // srand(1);
        QUIC(_mode, _p, S.data(), L.data(), _path_len, _path, _tol, _verbose,
             _max_iter, X.data(), W.data(), _opt, _cputime, _iter, _dGap,
             _neg_logdetX_trSX);

        // Catch if something went wrong.
        if (_iter[0] < 0 || std::isnan(_neg_logdetX_trSX[0]))
          std::make_tuple(INF, INF, 0, 0, 0);

        // drop small values
        double EPSILON_loc = tol; // std::sqrt(tol);
        threshold(X, EPSILON_loc);
        threshold(W, EPSILON_loc);

        double X_nnzpr = double((X.array() != 0.0).count()) / double(p);
        double W_nnzpr = double((W.array() != 0.0).count()) / double(p);

        return std::make_tuple(_neg_logdetX_trSX[0], _opt[0], size_t(_iter[0]),
                               X_nnzpr, W_nnzpr);
      };

  auto result = glasso_call(k);

  // Scale back output
  X = scale_icor_icov.asDiagonal() * X * scale_icor_icov.asDiagonal();
  W = scale_cor_cov.asDiagonal() * W * scale_cor_cov.asDiagonal();

  return std::make_tuple(X_array, W_array);
}

std::tuple<MAT_DN, MAT_DN, VEC_DN> compute_QS(MAT_DN_MAP &Y,
                                              const std::string &mode = "corr",
                                              const bool bias = true) {

  if (mode != "cov" && mode != "corr") {
    throw std::invalid_argument("Invalid mode: use 'cov' or 'corr'");
  }

  const size_t p = Y.rows();
  const size_t n = Y.cols();
  const double N = bias ? double(n) : double(n - 1);

  // Center
  VEC_DN mean = Y.rowwise().mean();
  MAT_DN Y_ctr = Y.colwise() - mean;

  // Row-wise std from covariance of centered data
  MAT_DN S0 = (Y_ctr * Y_ctr.transpose()) / N; // covariance of centered data
  VEC_DN std = (S0.diagonal().array() + EPSILON).sqrt(); // avoid div-by-0
  VEC_DN inv_std = (1.0 / std.array()).matrix();

  // Form Z: centered, and if corr-mode, standardized to unit variance
  MAT_DN Z = Y_ctr;
  if (mode == "corr") {
    Z = inv_std.asDiagonal() * Z; // row-wise scaling => Var(Z_i) ~ 1
  }

  // S computed from Z (so in corr-mode diag(S) ~ 1 by construction)
  MAT_DN S = (Z * Z.transpose()) / N;

  // Q = E[Z_i^2 Z_j^2] estimated as second moment of squared entries
  MAT_DN Zsq = Z.array().square().matrix(); // elementwise square
  MAT_DN Q = (Zsq * Zsq.transpose()) / N;

  return std::make_tuple(Q, S, std);
}

std::tuple<py::array_t<double>, // X_array,
           py::array_t<double>  // W_array,
           >
QUIC_base(py::array_t<double> &Y_array, double L_ij, double tol,
          size_t max_iter, double L_ii = 1e-6, int verbose = 1) {

  // Extract buffer information and check dimension consistency
  auto Y_buf = Y_array.request();

  // Check dimensions first
  if (Y_buf.ndim != 2)
    throw std::runtime_error("Error: Y must be a 2D array (p x n)");
  if (Y_buf.shape[0] <= 2 || Y_buf.shape[1] <= 2)
    throw std::runtime_error("Error: p and n must be greater than 2");
  if (L_ii < 0.0)
    throw std::runtime_error("Error: L_ii must be greater than 0.0");

  size_t p = Y_buf.shape[0];
  size_t n = Y_buf.shape[1];

  // Map matrices to Eigen
  MAT_DN_MAP Y(static_cast<double *>(Y_buf.ptr), p, n);

  py::array_t<double> X_array({p, p});
  auto buf_X = X_array.request();
  MAT_DN_MAP X(static_cast<double *>(buf_X.ptr), p, p);

  py::array_t<double> W_array({p, p});
  auto buf_W = W_array.request();
  MAT_DN_MAP W(static_cast<double *>(buf_W.ptr), p, p);

  // Compute covariance matrix S and standard deviation std
  MAT_DN S;
  VEC_DN std;
  std::tie(S, std) = compute_S(Y, "cov", true);

  //   // Scaling Matrics
  //   VEC_DN  scale_cov_cor   = (1.0 / std.array()).matrix(); //
  //   scale_cov_cor.asDiagonal()   * C    * scale_cov_cor.asDiagonal()   = Cor
  //   VEC_DN& scale_cor_cov   = std;                          //
  //   scale_cor_cov.asDiagonal()   * Cor  * scale_cor_cov.asDiagonal()   = C
  //   VEC_DN& scale_icor_icov = scale_cov_cor;                //
  //   scale_icor_icov.asDiagonal() * iCor * scale_icor_icov.asDiagonal() = iCov

  if (verbose > 0) {
    std::cout << "\n#######################################"
              << "\n## QUIC Solver Configuration"
              << "\n#######################################"
              << "\n Y_array shape   : (" << p << ", " << n << ")"
              << "\n L_ij            : " << L_ij
              << "\n Tolerance       : " << tol
              << "\n L_ii            : " << L_ii
              << "\n Max Iterations  : " << max_iter
              << "\n Verbosity Level : " << verbose
              << "\n#######################################\n"
              << std::flush;
  }

  MAT_DN L(p, p);
  L.setConstant(L_ij);
  L.diagonal().setConstant(L_ii);

  X.setIdentity();
  W.setIdentity();

  std::function<std::tuple<double, double, size_t, double, double>()>
      glasso_call = [&]() {
        char _mode = 'D';
        uint32_t _p = p;
        double _opt[1] = {0}; // -log|Θ| + tr(ΘS) + ||Λ ⊙ Θ||_1
        double _cputime[1] = {0};
        uint32_t _iter[1] = {0};
        double _dGap[1] = {0};
        double _neg_logdetX_trSX[1] = {0};
        uint32_t _path_len = 1;
        const double _path[1] = {1.0};
        double _tol = tol;
        uint32_t _max_iter = max_iter;
        int32_t _verbose = std::max(0, verbose - 1);

        // srand(1);
        QUIC(_mode, _p, S.data(), L.data(), _path_len, _path, _tol, _verbose,
             _max_iter, X.data(), W.data(), _opt, _cputime, _iter, _dGap,
             _neg_logdetX_trSX);

        // Catch if something went wrong.
        if (_iter[0] < 0 || std::isnan(_neg_logdetX_trSX[0]))
          std::make_tuple(INF, INF, 0, 0, 0);

        // drop small values
        double EPSILON_loc = tol; // std::sqrt(tol);
        threshold(X, EPSILON_loc);
        threshold(W, EPSILON_loc);

        double X_nnzpr = double((X.array() != 0.0).count()) / double(p);
        double W_nnzpr = double((W.array() != 0.0).count()) / double(p);

        return std::make_tuple(_neg_logdetX_trSX[0], _opt[0], size_t(_iter[0]),
                               X_nnzpr, W_nnzpr);
      };

  auto result = glasso_call();

  // Scale back output
  // X = scale_icor_icov.asDiagonal() * X * scale_icor_icov.asDiagonal();
  // W = scale_cor_cov.asDiagonal() * W * scale_cor_cov.asDiagonal();

  return std::make_tuple(X_array, W_array);
}

PYBIND11_MODULE(quic_pybind, m) {
  m.def("AQUIC", &AQUIC,
        "Adaptive Quadratic Inverse Covariance Matrix Estimation");
  m.def("QUIC", &QUIC_base, "Quadratic Inverse Covariance Matrix Estimation");
}