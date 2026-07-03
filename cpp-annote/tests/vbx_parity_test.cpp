// SPDX-License-Identifier: MIT
// Compare C++ ``cluster_vbx`` iteration traces to Python reference
// ``vbx_reference.npz``.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"
#include "plda_vbx.h"

/// ``b`` is one slice of NumPy ``gamma_trace[it]`` with C-order shape ``(T,
/// S)`` (row-major: ``t`` varies slow, ``s`` fast).
static double max_abs_gamma_ts(const Eigen::MatrixXd& a, const double* b, int T,
                               int S) {
  double m = 0.0;
  for (int t = 0; t < T; ++t) {
    for (int s = 0; s < S; ++s) {
      m = std::max(
          m,
          std::abs(a(t, s) -
                   b[static_cast<std::size_t>(t) * static_cast<std::size_t>(S) +
                     static_cast<std::size_t>(s)]));
    }
  }
  return m;
}

static double max_abs_vec(const Eigen::VectorXd& a, const double* b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a(i) - b[i]));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: vbx_parity_test <vbx_reference.npz>\n";
    return 2;
  }
  const std::string path = argv[1];
  cnpy::npz_t z = cnpy::npz_load(path);
  if (!z.count("fea") || !z.count("Phi") || !z.count("ahc") || !z.count("Fa") ||
      !z.count("Fb") || !z.count("init_smoothing") || !z.count("n_vbx_iters") ||
      !z.count("gamma_trace") || !z.count("pi_trace")) {
    throw std::runtime_error("vbx_reference.npz missing required keys");
  }
  const cnpy::NpyArray& fea_a = z["fea"];
  const cnpy::NpyArray& phi_a = z["Phi"];
  const cnpy::NpyArray& ahc_a = z["ahc"];
  const cnpy::NpyArray& gt_a = z["gamma_trace"];
  const cnpy::NpyArray& pt_a = z["pi_trace"];
  if (fea_a.shape.size() != 2 || phi_a.shape.size() != 1 ||
      ahc_a.shape.size() != 1) {
    throw std::runtime_error("bad fea/Phi/ahc rank");
  }
  const int T = static_cast<int>(fea_a.shape[0]);
  const int D = static_cast<int>(fea_a.shape[1]);
  if (static_cast<int>(phi_a.shape[0]) != D) {
    throw std::runtime_error("Phi length must equal fea column dimension");
  }
  if (static_cast<int>(ahc_a.shape[0]) != T) {
    throw std::runtime_error("ahc length != T");
  }
  if (gt_a.shape.size() != 3 || pt_a.shape.size() != 2) {
    throw std::runtime_error("gamma_trace / pi_trace rank");
  }
  const int n_it = static_cast<int>(gt_a.shape[0]);
  const int S = static_cast<int>(gt_a.shape[2]);
  if (static_cast<int>(pt_a.shape[0]) != n_it ||
      static_cast<int>(gt_a.shape[1]) != T ||
      static_cast<int>(pt_a.shape[1]) != S) {
    throw std::runtime_error("trace shape mismatch");
  }

  Eigen::MatrixXd fea =
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(fea_a.data<double>(), T,
                                                       D);
  Eigen::VectorXd Phi =
      Eigen::Map<const Eigen::VectorXd>(phi_a.data<double>(), D);
  std::vector<int> ahc(static_cast<std::size_t>(T));
  const std::int32_t* ahc_p = ahc_a.data<std::int32_t>();
  for (int t = 0; t < T; ++t) {
    ahc[static_cast<std::size_t>(t)] = static_cast<int>(ahc_p[t]);
  }

  const double Fa = z["Fa"].data<double>()[0];
  const double Fb = z["Fb"].data<double>()[0];
  const double init_smoothing = z["init_smoothing"].data<double>()[0];
  const int max_iters =
      static_cast<int>(z["n_vbx_iters"].data<std::int32_t>()[0]);

  Eigen::MatrixXd gamma;
  Eigen::VectorXd pi;
  std::vector<Eigen::MatrixXd> trace_gamma;
  std::vector<Eigen::VectorXd> trace_pi;
  cppannote::plda_vbx::cluster_vbx(ahc, fea, Phi, Fa, Fb, max_iters,
                                   init_smoothing, gamma, pi, -1.0,
                                   &trace_gamma, &trace_pi);

  if (static_cast<int>(trace_gamma.size()) != n_it ||
      static_cast<int>(trace_pi.size()) != n_it) {
    std::cerr << "FAIL: iter count cpp=" << trace_gamma.size()
              << " expected=" << n_it << "\n";
    return 1;
  }

  const double* gtp = gt_a.data<double>();
  const double* ptp = pt_a.data<double>();
  const size_t gs = static_cast<size_t>(T) * static_cast<size_t>(S);
  double max_g = 0.0;
  double max_p = 0.0;
  for (int it = 0; it < n_it; ++it) {
    max_g = std::max(
        max_g, max_abs_gamma_ts(trace_gamma[static_cast<std::size_t>(it)],
                                gtp + static_cast<std::size_t>(it) * gs, T, S));
    max_p = std::max(max_p, max_abs_vec(trace_pi[static_cast<std::size_t>(it)],
                                        ptp + static_cast<std::size_t>(it) *
                                                  static_cast<size_t>(S),
                                        static_cast<size_t>(S)));
  }
  std::cout << "vbx_parity max_abs_gamma=" << max_g << " max_abs_pi=" << max_p
            << "\n";
  const double tol = 1e-9;
  if (max_g > tol || max_p > tol) {
    std::cerr << "FAIL: exceeds tol " << tol << "\n";
    return 1;
  }
  return 0;
}
