// SPDX-License-Identifier: MIT
// Milestone 4: C++ ``PldaModel(embeddings)`` vs golden ``x0`` / ``fea`` /
// ``Phi`` in ``vbx_reference.npz``.

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cnpy.h"
#include "plda_vbx.h"

static double max_abs_mat(const Eigen::MatrixXd& a, const double* b, int rows,
                          int cols) {
  double m = 0.0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      m = std::max(m, std::abs(a(i, j) - b[static_cast<std::size_t>(i) *
                                               static_cast<std::size_t>(cols) +
                                           static_cast<std::size_t>(j)]));
    }
  }
  return m;
}

static double max_abs_vec(const Eigen::VectorXd& a, const double* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a(i) - b[static_cast<std::size_t>(i)]));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: plda_fea_golden_test <vbx_reference.npz> "
                 "<xvec_transform.npz> <plda.npz> [lda_dim]\n";
    std::cerr << "  Requires NPZ keys: train, x0, fea, Phi (from "
                 "write_vbx_golden_reference.py).\n";
    return 2;
  }
  const std::string ref_path = argv[1];
  const std::string xvec_path = argv[2];
  const std::string plda_path = argv[3];
  const int lda_dim = (argc == 5) ? std::stoi(argv[4]) : 128;

  cnpy::npz_t z = cnpy::npz_load(ref_path);
  if (!z.count("train") || !z.count("x0") || !z.count("fea") ||
      !z.count("Phi")) {
    std::cerr << "NPZ missing train / x0 / fea / Phi — regenerate with "
                 "write_vbx_golden_reference.py\n";
    return 2;
  }
  const cnpy::NpyArray& tr = z["train"];
  const cnpy::NpyArray& x0a = z["x0"];
  const cnpy::NpyArray& fe = z["fea"];
  const cnpy::NpyArray& ph = z["Phi"];
  if (tr.shape.size() != 2 || x0a.shape.size() != 2 || fe.shape.size() != 2 ||
      ph.shape.size() != 1) {
    throw std::runtime_error("bad train/x0/fea/Phi ranks");
  }
  const int T = static_cast<int>(tr.shape[0]);
  const int d_emb = static_cast<int>(tr.shape[1]);
  const int d_fea = static_cast<int>(fe.shape[1]);
  const int n_phi = static_cast<int>(ph.shape[0]);
  if (static_cast<int>(fe.shape[0]) != T ||
      static_cast<int>(x0a.shape[0]) != T ||
      static_cast<int>(x0a.shape[1]) != lda_dim || d_fea != lda_dim ||
      n_phi != lda_dim) {
    throw std::runtime_error("train/x0/fea/Phi shape mismatch vs lda_dim");
  }

  Eigen::MatrixXd train =
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(tr.data<double>(), T,
                                                       d_emb);

  cppannote::plda_vbx::PldaModel model;
  model.load(xvec_path, plda_path, lda_dim);

  const Eigen::MatrixXd x0_cpp = model.xvec_tf(train);
  const double mad_x0 = max_abs_mat(x0_cpp, x0a.data<double>(), T, lda_dim);
  std::cout << "x0 max_abs_diff=" << mad_x0 << "\n";

  const Eigen::MatrixXd fea_cpp = model(train);
  if (fea_cpp.rows() != T || fea_cpp.cols() != lda_dim) {
    throw std::runtime_error("C++ fea shape unexpected");
  }

  const double mad_fea = max_abs_mat(fea_cpp, fe.data<double>(), T, lda_dim);
  std::cout << "fea max_abs_diff=" << mad_fea << "\n";

  Eigen::VectorXd phi_cpp = model.phi_between;
  if (phi_cpp.size() > lda_dim) {
    phi_cpp.conservativeResize(lda_dim);
  }
  if (phi_cpp.size() != lda_dim) {
    throw std::runtime_error("C++ phi length unexpected");
  }
  const double mad_phi = max_abs_vec(phi_cpp, ph.data<double>(), lda_dim);
  std::cout << "Phi max_abs_diff=" << mad_phi << "\n";

  const double tol_x0 = 1e-4;
  const double tol_fea = 1e-4;
  const double tol_phi = 1e-4;
  if (mad_x0 > tol_x0 || mad_fea > tol_fea || mad_phi > tol_phi) {
    std::cerr << "FAIL (tol x0=" << tol_x0 << " fea=" << tol_fea
              << " phi=" << tol_phi << ")\n";
    return 1;
  }
  std::cout << "PASS PLDA x0 + fea + Phi (Milestone 4)\n";
  return 0;
}
