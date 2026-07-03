// SPDX-License-Identifier: MIT
// Recompute speaker_count (trim + sum + aggregate + rint→uint8) from golden
// binarized_segmentations.npz and compare to speaker_count_initial.npz.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static double json_double_field(const std::string& json, const char* key) {
  std::string pat = std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("JSON parse: missing float field \"") +
                             key + "\"");
  }
  return std::stod(m[1].str());
}

// golden_speaker_bounds.json: "max_speakers": null means no cap (numpy inf).
static int json_max_speakers_cap(const std::string& json) {
  std::regex re("\"max_speakers\"\\s*:\\s*(null|[-+0-9]+)");
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(
        "golden_speaker_bounds.json: missing \"max_speakers\"");
  }
  if (m[1].str() == "null") {
    return INT_MAX;
  }
  return std::stoi(m[1].str());
}

// NumPy: np.minimum(uint8, cap).astype(np.int8)  (via uint8 for values 0..255).
static std::int8_t cap_count_to_int8(std::uint8_t initial_u8,
                                     int max_speakers_cap) {
  const int m = std::min(static_cast<int>(initial_u8), max_speakers_cap);
  return static_cast<std::int8_t>(static_cast<std::uint8_t>(m));
}

// cppannote.core.SlidingWindow.closest_frame
static int closest_frame(double t, double sw_start, double sw_duration,
                         double sw_step) {
  const double x = (t - sw_start - 0.5 * sw_duration) / sw_step;
  return static_cast<int>(std::lrint(x));
}

static void trim_warmup_inplace(std::vector<float>& data, size_t num_chunks,
                                size_t& num_frames, size_t num_classes,
                                double warm0, double warm1, double& chunk_start,
                                double& chunk_duration) {
  const size_t n_left =
      static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm0));
  const size_t n_right =
      static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm1));
  if (n_left + n_right >= num_frames) {
    throw std::runtime_error("trim: warm_up removes all frames");
  }
  const size_t new_frames = num_frames - n_left - n_right;
  std::vector<float> out(num_chunks * new_frames * num_classes);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < new_frames; ++f) {
      for (size_t k = 0; k < num_classes; ++k) {
        out[(c * new_frames + f) * num_classes + k] =
            data[(c * num_frames + (f + n_left)) * num_classes + k];
      }
    }
  }
  data.swap(out);
  num_frames = new_frames;
  chunk_start += warm0 * chunk_duration;
  chunk_duration *= (1.0 - warm0 - warm1);
}

static std::vector<float> aggregate_sum_then_average(
    const std::vector<float>& summed,  // (C, F, 1) row-major
    size_t num_chunks, size_t num_frames_per_chunk, double chunks_start,
    double chunks_duration, double chunks_step, double rf_duration,
    double rf_step, float epsilon, float missing) {
  const int num_classes = 1;
  const double out_sw_start = chunks_start;
  const double out_sw_duration = rf_duration;
  const double out_sw_step = rf_step;

  const double end_t = chunks_start + chunks_duration +
                       static_cast<double>(num_chunks - 1) * chunks_step +
                       0.5 * rf_duration;
  const int num_out_frames =
      closest_frame(end_t, out_sw_start, out_sw_duration, out_sw_step) + 1;
  if (num_out_frames <= 0) {
    throw std::runtime_error("aggregate: non-positive num_out_frames");
  }

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

    for (size_t j = 0; j < num_frames_per_chunk; ++j) {
      const size_t idx = (ci * num_frames_per_chunk + j) * num_classes;
      const float raw = summed[idx];
      const float mask = std::isnan(raw) ? 0.f : 1.f;
      float score = raw;
      if (std::isnan(score)) {
        score = 0.f;
      }
      const int fi = start_frame + static_cast<int>(j);
      if (fi < 0 || fi >= num_out_frames) {
        continue;
      }
      const size_t o = static_cast<size_t>(fi);
      agg[o] += score * mask;
      occ[o] += mask;
      mask_max[o] = std::max(mask_max[o], mask);
    }
  }

  std::vector<float> avg(static_cast<size_t>(num_out_frames) * num_classes);
  for (int i = 0; i < num_out_frames; ++i) {
    const size_t o = static_cast<size_t>(i);
    const float d = std::max(occ[o], epsilon);
    avg[o] = agg[o] / d;
    if (mask_max[o] == 0.f) {
      avg[o] = missing;
    }
  }
  return avg;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: speaker_count_golden_test <golden_utterance_dir> "
                 "<receptive_field.json>\n";
    std::cerr << "  utterance dir must contain binarized_segmentations.npz and "
                 "speaker_count_initial.npz\n";
    std::cerr << "  optional: golden_speaker_bounds.json (from dump) for "
                 "speaker_count_capped.npz check\n";
    return 2;
  }
  const std::string utter_dir = argv[1];
  const std::string rf_path = argv[2];

  const std::string bin_npz = utter_dir + "/binarized_segmentations.npz";
  const std::string count_npz = utter_dir + "/speaker_count_initial.npz";
  const std::string capped_npz = utter_dir + "/speaker_count_capped.npz";
  const std::string bounds_json = utter_dir + "/golden_speaker_bounds.json";

  const std::string rf_json = read_text_file(rf_path);
  const double rf_duration = json_double_field(rf_json, "duration");
  const double rf_step = json_double_field(rf_json, "step");
  (void)json_double_field(rf_json, "start");  // reserved / sanity

  cnpy::npz_t bin = cnpy::npz_load(bin_npz);
  if (!bin.count("data")) {
    throw std::runtime_error("binarized_segmentations.npz missing 'data'");
  }
  const cnpy::NpyArray& darr = bin["data"];
  if (darr.shape.size() != 3) {
    throw std::runtime_error(
        "binarized data must be rank-3 (chunks, frames, classes)");
  }
  const size_t num_chunks = darr.shape[0];
  const size_t num_frames = darr.shape[1];
  const size_t num_classes = darr.shape[2];
  if (!bin.count("sliding_window_start") ||
      !bin.count("sliding_window_duration") ||
      !bin.count("sliding_window_step")) {
    throw std::runtime_error(
        "binarized_segmentations.npz missing sliding_window_* scalars");
  }
  double chunk_start = bin["sliding_window_start"].data<double>()[0];
  double chunk_duration = bin["sliding_window_duration"].data<double>()[0];
  const double chunk_step = bin["sliding_window_step"].data<double>()[0];

  const float* src = darr.data<float>();
  std::vector<float> row(num_chunks * num_frames * num_classes);
  std::memcpy(row.data(), src, row.size() * sizeof(float));

  const double warm0 = 0.0;
  const double warm1 = 0.0;
  size_t nf = num_frames;
  trim_warmup_inplace(row, num_chunks, nf, num_classes, warm0, warm1,
                      chunk_start, chunk_duration);

  std::vector<float> summed(num_chunks * nf * 1);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < nf; ++f) {
      float s = 0.f;
      for (size_t k = 0; k < num_classes; ++k) {
        s += row[(c * nf + f) * num_classes + k];
      }
      summed[c * nf + f] = s;
    }
  }

  const float epsilon = 1e-12f;
  const float missing = 0.f;
  std::vector<float> avg = aggregate_sum_then_average(
      summed, num_chunks, nf, chunk_start, chunk_duration, chunk_step,
      rf_duration, rf_step, epsilon, missing);

  cnpy::npz_t gold = cnpy::npz_load(count_npz);
  if (!gold.count("data")) {
    throw std::runtime_error("speaker_count_initial.npz missing 'data'");
  }
  const cnpy::NpyArray& garr = gold["data"];
  if (garr.shape.size() != 2 || garr.shape[1] != 1) {
    throw std::runtime_error("golden speaker count must be (T, 1)");
  }
  const size_t out_t = garr.shape[0];
  if (avg.size() != out_t) {
    std::ostringstream oss;
    oss << "length mismatch: computed " << avg.size() << " golden " << out_t;
    throw std::runtime_error(oss.str());
  }

  size_t mism = 0;
  std::vector<std::uint8_t> initial_u8(out_t);
  for (size_t i = 0; i < out_t; ++i) {
    const double r = std::rint(static_cast<double>(avg[i]));
    const auto expect = static_cast<uint8_t>(garr.data<uint8_t>()[i]);
    const auto got = static_cast<uint8_t>(std::max(0.0, std::min(255.0, r)));
    initial_u8[i] = got;
    if (got != expect) {
      if (mism < 5) {
        std::cerr << "mismatch at " << i << ": got " << static_cast<int>(got)
                  << " expected " << static_cast<int>(expect)
                  << " (avg=" << avg[i] << ")\n";
      }
      ++mism;
    }
  }

  if (mism != 0) {
    std::cerr << "FAIL: " << mism << " uint8 mismatches after rint\n";
    return 1;
  }
  std::cout << "PASS speaker_count_initial (C++ vs golden), frames=" << out_t
            << "\n";

  std::ifstream cap_probe(capped_npz);
  if (!cap_probe) {
    std::cout << "SKIP speaker_count_capped (no speaker_count_capped.npz)\n";
    return 0;
  }
  cap_probe.close();

  int max_cap = INT_MAX;
  std::ifstream b(bounds_json);
  if (b) {
    std::ostringstream ss;
    ss << b.rdbuf();
    max_cap = json_max_speakers_cap(ss.str());
  }

  cnpy::npz_t cap_g = cnpy::npz_load(capped_npz);
  if (!cap_g.count("data")) {
    throw std::runtime_error("speaker_count_capped.npz missing 'data'");
  }
  const cnpy::NpyArray& carr = cap_g["data"];
  if (carr.shape.size() != 2 || carr.shape[1] != 1 ||
      carr.shape[0] != static_cast<size_t>(out_t)) {
    throw std::runtime_error("speaker_count_capped shape must match initial");
  }
  if (carr.word_size != 1) {
    throw std::runtime_error(
        "speaker_count_capped: expected 1-byte integer elements (int8)");
  }

  size_t cap_mism = 0;
  for (size_t i = 0; i < out_t; ++i) {
    const std::int8_t got = cap_count_to_int8(initial_u8[i], max_cap);
    const std::int8_t expect = carr.data<std::int8_t>()[i];
    if (got != expect) {
      if (cap_mism < 5) {
        std::cerr << "capped mismatch at " << i << ": got "
                  << static_cast<int>(got) << " expected "
                  << static_cast<int>(expect) << "\n";
      }
      ++cap_mism;
    }
  }
  if (cap_mism != 0) {
    std::cerr << "FAIL: " << cap_mism << " speaker_count_capped mismatches\n";
    return 1;
  }
  std::cout << "PASS speaker_count_capped (C++ vs golden)\n";
  return 0;
}
