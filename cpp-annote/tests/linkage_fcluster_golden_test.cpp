// SPDX-License-Identifier: MIT
// Compare C++ ``pdist`` + centroid ``linkage`` + ``fcluster`` to SciPy
// reference in ``vbx_reference.npz``.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"
#include "scipy_linkage.h"

static double max_abs_diff(const std::vector<double>& a, const double* b,
                           size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: linkage_fcluster_golden_test <vbx_reference.npz>\n";
    return 2;
  }
  cnpy::npz_t z = cnpy::npz_load(argv[1]);
  if (!z.count("train_n") || !z.count("pdist_condensed") ||
      !z.count("linkage_Z") || !z.count("ahc")) {
    std::cerr << "NPZ missing train_n / pdist_condensed / linkage_Z / ahc "
                 "(regenerate with write_vbx_golden_reference.py)\n";
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

  double fcluster_t = 0.6;
  if (z.count("fcluster_threshold")) {
    fcluster_t = z["fcluster_threshold"].data<double>()[0];
  }
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
  return 0;
}
