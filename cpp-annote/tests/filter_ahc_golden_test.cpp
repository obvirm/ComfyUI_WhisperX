// SPDX-License-Identifier: MIT
// Milestone 3: ``filter_embeddings`` + ``pdist`` / centroid ``linkage`` /
// ``fcluster`` vs ``vbx_reference.npz``.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"
#include "filter_train.h"
#include "scipy_linkage.h"

static double max_abs_diff(const std::vector<double>& a, const double* b,
                           size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

static double max_abs_mat(const Eigen::MatrixXd& a, const double* b) {
  double m = 0.0;
  const int r = static_cast<int>(a.rows());
  const int c = static_cast<int>(a.cols());
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      m = std::max(
          m,
          std::abs(a(i, j) -
                   b[static_cast<std::size_t>(i) * static_cast<std::size_t>(c) +
                     static_cast<std::size_t>(j)]));
    }
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 2) {
    std::cerr << "Usage: filter_ahc_golden_test <vbx_reference.npz> "
                 "[<utterance_dir>]\n";
    std::cerr << "  With utterance_dir: also checks filter indices vs NPZ "
                 "train_chunk_idx / train_speaker_idx\n"
                 "  and raw train max-abs vs train_n before row-normalize "
                 "(optional keys).\n";
    return 2;
  }
  const std::string ref_path = argv[1];
  cnpy::npz_t z = cnpy::npz_load(ref_path);
  if (!z.count("train_n") || !z.count("pdist_condensed") ||
      !z.count("linkage_Z") || !z.count("ahc")) {
    std::cerr << "NPZ missing train_n / pdist_condensed / linkage_Z / ahc\n";
    return 2;
  }
  const cnpy::NpyArray& tn = z["train_n"];
  const cnpy::NpyArray& pd = z["pdist_condensed"];
  const cnpy::NpyArray& lz = z["linkage_Z"];
  const cnpy::NpyArray& ah = z["ahc"];
  if (tn.shape.size() != 2 || pd.shape.size() != 1 || lz.shape.size() != 1 ||
      ah.shape.size() != 1) {
    throw std::runtime_error("unexpected array ranks");
  }
  const int n = static_cast<int>(tn.shape[0]);
  const int d = static_cast<int>(tn.shape[1]);
  const size_t m = static_cast<size_t>(n * (n - 1) / 2);
  if (pd.num_vals != m ||
      lz.num_vals != static_cast<size_t>(4 * std::max(0, n - 1)) ||
      ah.num_vals != static_cast<size_t>(n)) {
    throw std::runtime_error("size mismatch");
  }

  double fcluster_t = 0.6;
  if (z.count("fcluster_threshold")) {
    fcluster_t = z["fcluster_threshold"].data<double>()[0];
  }

  if (argc == 3) {
    const std::string utter = argv[2];
    cnpy::npz_t emb_npz = cnpy::npz_load(utter + "/embeddings.npz");
    cnpy::npz_t bin_npz =
        cnpy::npz_load(utter + "/binarized_segmentations.npz");
    if (!emb_npz.count("embeddings") || !bin_npz.count("data")) {
      throw std::runtime_error(
          "utterance dir missing embeddings.npz or "
          "binarized_segmentations.npz");
    }
    const cnpy::NpyArray& emb = emb_npz["embeddings"];
    const cnpy::NpyArray& bin = bin_npz["data"];
    if (emb.shape.size() != 3 || bin.shape.size() != 3) {
      throw std::runtime_error("bad emb/bin rank");
    }
    const int C = static_cast<int>(emb.shape[0]);
    const int S = static_cast<int>(emb.shape[1]);
    const int dim = static_cast<int>(emb.shape[2]);
    const int F = static_cast<int>(bin.shape[1]);
    const int Sb = static_cast<int>(bin.shape[2]);
    if (static_cast<int>(bin.shape[0]) != C || Sb != S || dim != d) {
      throw std::runtime_error("emb/bin shape mismatch vs train_n");
    }
    std::vector<float> emb_f(emb.data<float>(),
                             emb.data<float>() + emb.num_vals);
    std::vector<float> bin_f(bin.data<float>(),
                             bin.data<float>() + bin.num_vals);
    std::vector<int> c_idx;
    std::vector<int> s_idx;
    Eigen::MatrixXd train;
    cppannote::filter_train::filter_embeddings_train(
        C, F, S, dim, emb_f.data(), bin_f.data(), 0.2, c_idx, s_idx, train);
    if (static_cast<int>(c_idx.size()) != n) {
      std::cerr << "FAIL: filter T=" << c_idx.size() << " expected " << n
                << "\n";
      return 1;
    }
    if (z.count("train_chunk_idx") && z.count("train_speaker_idx")) {
      const std::int32_t* rc = z["train_chunk_idx"].data<std::int32_t>();
      const std::int32_t* rs = z["train_speaker_idx"].data<std::int32_t>();
      for (int i = 0; i < n; ++i) {
        if (c_idx[static_cast<std::size_t>(i)] != static_cast<int>(rc[i]) ||
            s_idx[static_cast<std::size_t>(i)] != static_cast<int>(rs[i])) {
          std::cerr << "FAIL: filter index mismatch at i=" << i << "\n";
          return 1;
        }
      }
      std::cout << "filter chunk/spk indices match NPZ\n";
    }
    Eigen::MatrixXd train_n_cpp = train;
    for (int i = 0; i < train_n_cpp.rows(); ++i) {
      double norm = train_n_cpp.row(i).norm();
      if (norm > 1e-12) {
        train_n_cpp.row(i) /= norm;
      }
    }
    const double mad_train_n = max_abs_mat(train_n_cpp, tn.data<double>());
    std::cout << "L2-normalized train vs train_n max_abs=" << mad_train_n
              << "\n";
    if (mad_train_n > 1e-9) {
      std::cerr << "FAIL: train_n mismatch after filter\n";
      return 1;
    }
  }

  std::vector<double> xflat(static_cast<size_t>(n) * static_cast<size_t>(d));
  const double* tnp = tn.data<double>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      xflat[static_cast<size_t>(i) * static_cast<size_t>(d) +
            static_cast<size_t>(j)] =
          tnp[static_cast<size_t>(i) * static_cast<size_t>(d) +
              static_cast<size_t>(j)];
    }
  }
  std::vector<double> pd_cpp;
  cppannote::scipy_linkage::pdist_euclidean(xflat, n, d, pd_cpp);
  const double mad_pd = max_abs_diff(pd_cpp, pd.data<double>(), m);
  std::cout << "pdist max_abs_diff=" << mad_pd << "\n";

  std::vector<double> Z_cpp;
  cppannote::scipy_linkage::linkage_centroid_naive(pd_cpp, n, Z_cpp);
  const double mad_z = max_abs_diff(Z_cpp, lz.data<double>(), Z_cpp.size());
  std::cout << "linkage_Z max_abs_diff=" << mad_z << "\n";

  std::vector<int> fc;
  cppannote::scipy_linkage::fcluster_distance(Z_cpp, n, fcluster_t, fc);
  std::vector<int> ahc_cpp;
  cppannote::scipy_linkage::remap_labels_contiguous(fc, ahc_cpp);

  int mism = 0;
  const std::int32_t* ahp = ah.data<std::int32_t>();
  for (int i = 0; i < n; ++i) {
    if (ahc_cpp[static_cast<size_t>(i)] != static_cast<int>(ahp[i])) {
      ++mism;
    }
  }
  std::cout << "ahc mismatches=" << mism << " / " << n
            << " fcluster_threshold=" << fcluster_t << "\n";

  const double tol_pd = 1e-9;
  const double tol_z = 1e-9;
  if (mad_pd > tol_pd || mad_z > tol_z || mism > 0) {
    std::cerr << "FAIL\n";
    return 1;
  }
  std::cout << "PASS filter + AHC chain (Milestone 3)\n";
  return 0;
}
