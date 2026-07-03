// SPDX-License-Identifier: MIT
// Compare C++ VBx clustering to Python golden ``clustering.npz`` (approximate).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustering_vbx.h"
#include "cnpy.h"
#include "plda_vbx.h"

static float max_abs_diff_int8(const std::vector<std::int8_t>& a,
                               const std::int8_t* b, size_t n) {
  float m = 0.f;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m,
                 std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i])));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cerr << "Usage: clustering_golden_test <golden_utterance_dir> "
                 "<xvec_transform.npz> <plda.npz>\n";
    std::cerr << "Env: SKIP_CLUSTERING_GOLDEN=1 to skip (VBx numerics still "
                 "being aligned with SciPy).\n";
    return 0;
  }
  if (const char* sk = std::getenv("SKIP_CLUSTERING_GOLDEN")) {
    if (sk[0] == '1') {
      std::cout
          << "SKIP_CLUSTERING_GOLDEN=1 — skipping VBx golden comparison.\n";
      return 0;
    }
  }
  if (argc != 4) {
    std::cerr << "Usage: clustering_golden_test <golden_utterance_dir> "
                 "<xvec_transform.npz> <plda.npz>\n";
    return 2;
  }
  const std::string dir = argv[1];
  const std::string xvec = argv[2];
  const std::string plda_npz = argv[3];
  const std::string emb_path = dir + "/embeddings.npz";
  const std::string bin_path = dir + "/binarized_segmentations.npz";
  const std::string clu_path = dir + "/clustering.npz";

  cnpy::npz_t emb_npz = cnpy::npz_load(emb_path);
  cnpy::npz_t bin_npz = cnpy::npz_load(bin_path);
  cnpy::npz_t clu_npz = cnpy::npz_load(clu_path);
  if (!emb_npz.count("embeddings") || !bin_npz.count("data") ||
      !clu_npz.count("hard_clusters")) {
    throw std::runtime_error("missing keys in golden npz");
  }
  const cnpy::NpyArray& emb = emb_npz["embeddings"];
  const cnpy::NpyArray& bin = bin_npz["data"];
  const cnpy::NpyArray& gold = clu_npz["hard_clusters"];
  if (emb.shape.size() != 3 || bin.shape.size() != 3) {
    throw std::runtime_error("unexpected ranks");
  }
  const int C = static_cast<int>(emb.shape[0]);
  const int S = static_cast<int>(emb.shape[1]);
  const int dim = static_cast<int>(emb.shape[2]);
  const int Cb = static_cast<int>(bin.shape[0]);
  const int F = static_cast<int>(bin.shape[1]);
  const int Sb = static_cast<int>(bin.shape[2]);
  if (Cb != C || Sb != S) {
    throw std::runtime_error("embeddings / binarized shape mismatch");
  }
  std::vector<float> emb_f(emb.data<float>(), emb.data<float>() + emb.num_vals);
  std::vector<float> bin_f(bin.data<float>(), bin.data<float>() + bin.num_vals);

  cppannote::plda_vbx::PldaModel plda;
  plda.load(xvec, plda_npz, 128);
  cppannote::clustering_vbx::VbxClusteringParams pr;
  pr.threshold = 0.6;
  pr.Fa = 0.07;
  pr.Fb = 0.8;
  pr.min_clusters = 1;
  pr.max_clusters = 1000000000;
  pr.num_clusters = -1;

  std::vector<std::int8_t> hard;
  cppannote::clustering_vbx::vbx_clustering_hard(
      plda, pr, C, F, S, dim, emb_f.data(), bin_f.data(), hard);

  const std::int8_t* gptr = gold.data<std::int8_t>();
  const size_t n = hard.size();
  if (gold.num_vals != n) {
    throw std::runtime_error("hard_clusters size mismatch");
  }
  const float mad = max_abs_diff_int8(hard, gptr, n);
  const int mism = [&]() {
    int c = 0;
    for (size_t i = 0; i < n; ++i) {
      if (hard[i] != gptr[i]) {
        ++c;
      }
    }
    return c;
  }();
  const float frac = static_cast<float>(mism) / static_cast<float>(n);
  std::cout << "hard_clusters max_abs_diff(int8)=" << mad
            << " mismatch_frac=" << frac << "\n";
  if (frac > 0.35f) {
    std::cerr
        << "FAIL: mismatch_frac too high (expected rough parity with Python)\n";
    return 1;
  }
  return 0;
}
