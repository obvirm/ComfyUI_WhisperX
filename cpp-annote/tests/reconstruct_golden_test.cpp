// SPDX-License-Identifier: MIT
// C++ parity for pipeline.reconstruct + to_diarization vs golden NPZ
// (discrete_diarization_overlap / discrete_diarization_exclusive).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"

// ---- cppannote.core.SlidingWindow.closest_frame
// --------------------------------
static int closest_frame(double t, double sw_start, double sw_duration,
                         double sw_step) {
  const double x = (t - sw_start - 0.5 * sw_duration) / sw_step;
  return static_cast<int>(std::lrint(x));
}

// Loose crop frame index range for SlidingWindow (segment.py crop, Segment
// focus).
static void crop_loose_frame_range(double focus_start, double focus_end,
                                   double sw_start, double sw_duration,
                                   double sw_step, int& out_i0,
                                   int& out_i1_exclusive) {
  const double i_ = (focus_start - sw_duration - sw_start) / sw_step;
  int i = static_cast<int>(std::ceil(i_));
  const double j_ = (focus_end - sw_start) / sw_step;
  int j = static_cast<int>(std::floor(j_));
  out_i0 = i;
  out_i1_exclusive = j + 1;
}

// Extent of SlidingWindowFeature with n rows (range_to_segment(0, n), i0==0).
static void extent_of_frames(double sw_start, double /*sw_duration*/,
                             double sw_step, size_t n_rows, double& seg_start,
                             double& seg_end) {
  seg_start = sw_start;
  seg_end = sw_start + static_cast<double>(n_rows) * sw_step;
}

// Inference.aggregate (hamming=false, warm_up default 0,0 in call from
// to_diarization)
static void inference_aggregate(
    const std::vector<float>& scores,  // (C, F, K) row-major, may contain NaN
    size_t num_chunks, size_t num_frames_per_chunk, size_t num_classes,
    double chunks_start, double chunks_duration, double chunks_step,
    double out_duration, double out_step, bool skip_average, float epsilon,
    float missing,
    std::vector<float>& out_avg,  // (num_out_frames, K)
    int& num_out_frames) {
  const double out_sw_start = chunks_start;
  const double out_sw_duration = out_duration;
  const double out_sw_step = out_step;

  const double end_t = chunks_start + chunks_duration +
                       static_cast<double>(num_chunks - 1) * chunks_step +
                       0.5 * out_sw_duration;
  num_out_frames =
      closest_frame(end_t, out_sw_start, out_sw_duration, out_sw_step) + 1;
  if (num_out_frames <= 0) {
    throw std::runtime_error("aggregate: non-positive num_out_frames");
  }

  const int nf = num_frames_per_chunk;
  const int nc = static_cast<int>(num_classes);
  std::vector<float> agg(static_cast<size_t>(num_out_frames) * num_classes,
                         0.f);
  std::vector<float> occ(static_cast<size_t>(num_out_frames) * num_classes,
                         0.f);
  std::vector<float> mask_max(static_cast<size_t>(num_out_frames) * num_classes,
                              0.f);

  for (size_t ci = 0; ci < num_chunks; ++ci) {
    const double chunk_start =
        chunks_start + static_cast<double>(ci) * chunks_step;
    const int start_frame =
        closest_frame(chunk_start + 0.5 * out_sw_duration, out_sw_start,
                      out_sw_duration, out_sw_step);

    for (int j = 0; j < nf; ++j) {
      for (int k = 0; k < nc; ++k) {
        const size_t idx =
            (ci * static_cast<size_t>(nf) + static_cast<size_t>(j)) *
                num_classes +
            static_cast<size_t>(k);
        const float raw = scores[idx];
        const float mask = std::isnan(raw) ? 0.f : 1.f;
        float score = raw;
        if (std::isnan(score)) {
          score = 0.f;
        }
        const int fi = start_frame + j;
        if (fi < 0 || fi >= num_out_frames) {
          continue;
        }
        const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) +
                         static_cast<size_t>(k);
        agg[o] += score * mask;
        occ[o] += mask;
        mask_max[o] = std::max(mask_max[o], mask);
      }
    }
  }

  out_avg.resize(static_cast<size_t>(num_out_frames) * num_classes);
  for (int fi = 0; fi < num_out_frames; ++fi) {
    for (int k = 0; k < nc; ++k) {
      const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) +
                       static_cast<size_t>(k);
      if (skip_average) {
        out_avg[o] = agg[o];
      } else {
        out_avg[o] = agg[o] / std::max(occ[o], epsilon);
      }
      if (mask_max[o] == 0.f) {
        out_avg[o] = missing;
      }
    }
  }
}

static std::vector<int> argsort_desc_stable(const float* row, int k) {
  std::vector<int> idx(static_cast<size_t>(k));
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [row](int a, int b) {
    if (row[a] != row[b]) {
      return row[a] > row[b];
    }
    return a < b;
  });
  return idx;
}

// Crop 2D feature (n_samples, n_cols) to loose intersection with [focus_start,
// focus_end].
static void crop_feature_loose(const std::vector<float>& data, int n_samples,
                               int n_cols, double sw_start, double sw_duration,
                               double sw_step, double focus_start,
                               double focus_end, std::vector<float>& out_data,
                               int& out_rows, double& new_sw_start) {
  int i0 = 0;
  int i1 = 0;
  crop_loose_frame_range(focus_start, focus_end, sw_start, sw_duration, sw_step,
                         i0, i1);
  const int clipped0 = std::max(0, i0);
  const int clipped1 = std::min(n_samples, i1);
  if (clipped0 >= clipped1) {
    out_rows = 0;
    out_data.clear();
    new_sw_start = sw_start;
    return;
  }
  out_rows = clipped1 - clipped0;
  out_data.resize(static_cast<size_t>(out_rows) * static_cast<size_t>(n_cols));
  for (int r = 0; r < out_rows; ++r) {
    const int src_row = clipped0 + r;
    for (int c = 0; c < n_cols; ++c) {
      out_data[static_cast<size_t>(r) * static_cast<size_t>(n_cols) +
               static_cast<size_t>(c)] =
          data[static_cast<size_t>(src_row) * static_cast<size_t>(n_cols) +
               static_cast<size_t>(c)];
    }
  }
  new_sw_start = sw_start + static_cast<double>(clipped0) * sw_step;
}

// reconstruct() + to_diarization() — count_row_major (T,1) int8 per frame.
static std::vector<float> reconstruct_to_diarization(
    const std::vector<float>& segmentations,  // (C, F, L) local speakers
    size_t num_chunks, size_t num_frames, size_t local_speakers,
    double seg_chunk_start, double seg_chunk_duration, double seg_chunk_step,
    const std::int8_t* hard_clusters,            // (C, L) row-major
    const std::vector<std::int8_t>& count_flat,  // (T,) one int8 per frame
    double cnt_sw_start, double cnt_sw_duration, double cnt_sw_step) {
  const int C = static_cast<int>(num_chunks);
  const int F = static_cast<int>(num_frames);
  const int L = static_cast<int>(local_speakers);

  int max_clu = -3;
  for (int c = 0; c < C; ++c) {
    for (int j = 0; j < L; ++j) {
      const std::int8_t v = hard_clusters[c * L + j];
      max_clu = std::max(max_clu, static_cast<int>(v));
    }
  }
  const int num_clusters = max_clu + 1;
  if (num_clusters <= 0) {
    throw std::runtime_error("reconstruct: no positive cluster ids");
  }

  std::vector<float> clustered(static_cast<size_t>(C) * static_cast<size_t>(F) *
                                   static_cast<size_t>(num_clusters),
                               std::numeric_limits<float>::quiet_NaN());

  for (int c = 0; c < C; ++c) {
    const float* segm =
        &segmentations[static_cast<size_t>(c) * static_cast<size_t>(F) *
                       static_cast<size_t>(L)];
    const std::int8_t* cluster = &hard_clusters[c * L];
    std::vector<char> seen_k(static_cast<size_t>(num_clusters), 0);
    for (int j = 0; j < L; ++j) {
      const int k = static_cast<int>(cluster[j]);
      if (k == -2) {
        continue;
      }
      if (k >= 0 && k < num_clusters) {
        seen_k[static_cast<size_t>(k)] = 1;
      }
    }
    for (int k = 0; k < num_clusters; ++k) {
      if (!seen_k[static_cast<size_t>(k)]) {
        continue;
      }
      for (int f = 0; f < F; ++f) {
        bool any_finite = false;
        float m = 0.f;
        for (int j = 0; j < L; ++j) {
          if (static_cast<int>(cluster[j]) != k) {
            continue;
          }
          const float v = segm[static_cast<size_t>(f) * static_cast<size_t>(L) +
                               static_cast<size_t>(j)];
          if (std::isnan(v)) {
            continue;
          }
          if (!any_finite) {
            m = v;
            any_finite = true;
          } else {
            m = std::max(m, v);
          }
        }
        const size_t dst = (static_cast<size_t>(c) * static_cast<size_t>(F) +
                            static_cast<size_t>(f)) *
                               static_cast<size_t>(num_clusters) +
                           static_cast<size_t>(k);
        clustered[dst] =
            any_finite ? m : std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  std::vector<float> activations;
  int num_act_frames = 0;
  inference_aggregate(
      clustered, num_chunks, num_frames, static_cast<size_t>(num_clusters),
      seg_chunk_start, seg_chunk_duration, seg_chunk_step, cnt_sw_duration,
      cnt_sw_step, true, 1e-12f, 0.f, activations, num_act_frames);

  const int T = num_act_frames;
  int K = num_clusters;
  int max_spf = 0;
  for (size_t t = 0; t < count_flat.size(); ++t) {
    max_spf = std::max(max_spf, static_cast<int>(count_flat[t]));
  }
  if (max_spf < 0) {
    max_spf = 0;
  }
  if (K < max_spf) {
    std::vector<float> padded(
        static_cast<size_t>(T) * static_cast<size_t>(max_spf), 0.f);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        padded[static_cast<size_t>(t) * static_cast<size_t>(max_spf) +
               static_cast<size_t>(k)] =
            activations[static_cast<size_t>(t) * static_cast<size_t>(K) +
                        static_cast<size_t>(k)];
      }
    }
    activations.swap(padded);
    K = max_spf;
  }

  double act_ext_s = 0.;
  double act_ext_e = 0.;
  extent_of_frames(cnt_sw_start, cnt_sw_duration, cnt_sw_step,
                   static_cast<size_t>(T), act_ext_s, act_ext_e);

  const int Tcnt = static_cast<int>(count_flat.size());
  double cnt_ext_s = 0.;
  double cnt_ext_e = 0.;
  extent_of_frames(cnt_sw_start, cnt_sw_duration, cnt_sw_step,
                   static_cast<size_t>(Tcnt), cnt_ext_s, cnt_ext_e);

  const double inter_s = std::max(act_ext_s, cnt_ext_s);
  const double inter_e = std::min(act_ext_e, cnt_ext_e);

  std::vector<float> act_cropped;
  int act_rows = 0;
  double act_sw0 = 0.;
  crop_feature_loose(activations, T, K, cnt_sw_start, cnt_sw_duration,
                     cnt_sw_step, inter_s, inter_e, act_cropped, act_rows,
                     act_sw0);

  std::vector<float> cnt_2d(static_cast<size_t>(Tcnt) * 1u);
  for (int t = 0; t < Tcnt; ++t) {
    cnt_2d[static_cast<size_t>(t)] =
        static_cast<float>(count_flat[static_cast<size_t>(t)]);
  }
  std::vector<float> cnt_cropped;
  int cnt_rows = 0;
  double cnt_sw0 = 0.;
  crop_feature_loose(cnt_2d, Tcnt, 1, cnt_sw_start, cnt_sw_duration,
                     cnt_sw_step, inter_s, inter_e, cnt_cropped, cnt_rows,
                     cnt_sw0);

  if (act_rows != cnt_rows) {
    std::ostringstream oss;
    oss << "crop row mismatch: activations " << act_rows << " count "
        << cnt_rows;
    throw std::runtime_error(oss.str());
  }
  (void)act_sw0;
  (void)cnt_sw0;

  std::vector<float> binary(
      static_cast<size_t>(act_rows) * static_cast<size_t>(K), 0.f);
  for (int t = 0; t < act_rows; ++t) {
    const float* arow =
        &act_cropped[static_cast<size_t>(t) * static_cast<size_t>(K)];
    std::vector<int> order = argsort_desc_stable(arow, K);
    const int c = static_cast<int>(
        std::lrint(static_cast<double>(cnt_cropped[static_cast<size_t>(t)])));
    const int c_use = std::max(0, std::min(c, K));
    for (int i = 0; i < c_use; ++i) {
      binary[static_cast<size_t>(t) * static_cast<size_t>(K) +
             static_cast<size_t>(order[static_cast<size_t>(i)])] = 1.f;
    }
  }

  return binary;
}

static void load_segmentations_npz(const std::string& path,
                                   std::vector<float>& data, size_t& d0,
                                   size_t& d1, size_t& d2, double& sw_start,
                                   double& sw_dur, double& sw_step) {
  cnpy::npz_t npz = cnpy::npz_load(path);
  if (!npz.count("data")) {
    throw std::runtime_error(path + ": missing data");
  }
  const cnpy::NpyArray& arr = npz["data"];
  sw_start = npz["sliding_window_start"].data<double>()[0];
  sw_dur = npz["sliding_window_duration"].data<double>()[0];
  sw_step = npz["sliding_window_step"].data<double>()[0];
  if (arr.shape.size() != 3) {
    throw std::runtime_error(path + ": expected rank-3 data");
  }
  d0 = arr.shape[0];
  d1 = arr.shape[1];
  d2 = arr.shape[2];
  data.resize(d0 * d1 * d2);
  std::memcpy(data.data(), arr.data<float>(), data.size() * sizeof(float));
}

static void run_one(const std::string& utter_dir, const char* golden_name,
                    const std::vector<std::int8_t>* count_override) {
  const std::string seg_path = utter_dir + "/segmentations.npz";
  const std::string hc_path = utter_dir + "/hard_clusters_final.npz";

  std::vector<float> seg;
  size_t C = 0, F = 0, L = 0;
  double seg_ss = 0, seg_sd = 0, seg_st = 0;
  load_segmentations_npz(seg_path, seg, C, F, L, seg_ss, seg_sd, seg_st);

  cnpy::npz_t hc = cnpy::npz_load(hc_path);
  if (!hc.count("hard_clusters")) {
    throw std::runtime_error("hard_clusters_final.npz missing hard_clusters");
  }
  const cnpy::NpyArray& harr = hc["hard_clusters"];
  if (harr.shape.size() != 2 || harr.shape[0] != C || harr.shape[1] != L) {
    throw std::runtime_error(
        "hard_clusters shape must match segmentations (chunks, "
        "local_speakers)");
  }
  if (harr.word_size != 1) {
    throw std::runtime_error("hard_clusters must be int8");
  }
  const std::int8_t* hptr = harr.data<std::int8_t>();

  const std::string cnt_path = utter_dir + "/speaker_count_capped.npz";
  cnpy::npz_t cn = cnpy::npz_load(cnt_path);
  if (!cn.count("data")) {
    throw std::runtime_error("speaker_count_capped.npz missing data");
  }
  const cnpy::NpyArray& carr = cn["data"];
  if (carr.shape.size() != 2 || carr.shape[1] != 1) {
    throw std::runtime_error("count must be (T,1)");
  }
  const size_t Tcnt = carr.shape[0];
  std::vector<std::int8_t> count_flat(Tcnt);
  if (carr.word_size != 1) {
    throw std::runtime_error("count must be 1-byte int");
  }
  if (count_override != nullptr) {
    if (count_override->size() != Tcnt) {
      throw std::runtime_error("count_override length mismatch");
    }
    count_flat = *count_override;
  } else {
    std::memcpy(count_flat.data(), carr.data<std::int8_t>(),
                Tcnt * sizeof(std::int8_t));
  }

  double cnt_ss = cn["sliding_window_start"].data<double>()[0];
  double cnt_sd = cn["sliding_window_duration"].data<double>()[0];
  double cnt_st = cn["sliding_window_step"].data<double>()[0];

  std::vector<float> got =
      reconstruct_to_diarization(seg, C, F, L, seg_ss, seg_sd, seg_st, hptr,
                                 count_flat, cnt_ss, cnt_sd, cnt_st);

  const std::string gold_path = utter_dir + "/" + golden_name;
  cnpy::npz_t gn = cnpy::npz_load(gold_path);
  if (!gn.count("data")) {
    throw std::runtime_error(std::string(golden_name) + " missing data");
  }
  const cnpy::NpyArray& g = gn["data"];
  if (g.word_size != 4) {
    throw std::runtime_error("golden discrete must be float32");
  }
  const float* gptr = g.data<float>();
  if (g.shape.size() != 2) {
    throw std::runtime_error("golden discrete must be rank-2");
  }
  const size_t out_el = got.size();
  if (out_el != g.num_vals || got.size() / g.shape[1] != g.shape[0]) {
    throw std::runtime_error("golden shape mismatch vs computed");
  }
  float mad = 0.f;
  for (size_t i = 0; i < out_el; ++i) {
    mad = std::max(mad, std::abs(got[i] - gptr[i]));
  }
  if (mad != 0.f) {
    std::cerr << "FAIL " << golden_name << " max_abs_diff=" << mad << "\n";
    throw std::runtime_error("discrete diarization mismatch");
  }
  std::cout << "PASS " << golden_name << " (" << g.shape[0] << " x "
            << g.shape[1] << ")\n";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: reconstruct_golden_test <golden_utterance_dir>\n";
    std::cerr << "  Requires segmentations.npz, hard_clusters_final.npz, "
                 "speaker_count_capped.npz,\n";
    std::cerr << "  discrete_diarization_overlap.npz, "
                 "discrete_diarization_exclusive.npz\n";
    return 2;
  }
  const std::string utter_dir = argv[1];

  try {
    run_one(utter_dir, "discrete_diarization_overlap.npz", nullptr);

    cnpy::npz_t cn = cnpy::npz_load(utter_dir + "/speaker_count_capped.npz");
    const cnpy::NpyArray& carr = cn["data"];
    const size_t Tcnt = carr.shape[0];
    std::vector<std::int8_t> excl(Tcnt);
    const std::int8_t* src = carr.data<std::int8_t>();
    for (size_t t = 0; t < Tcnt; ++t) {
      const int v = static_cast<int>(src[t]);
      excl[t] = static_cast<std::int8_t>(std::max(0, std::min(1, v)));
    }
    run_one(utter_dir, "discrete_diarization_exclusive.npz", &excl);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
