// SPDX-License-Identifier: MIT
// Compare ONNX Runtime segmentation over the full sliding window (all chunks)
// to golden ``binarized_segmentations.npz`` / ``segmentations.npz``.
//
// If this fails, oracle ``hard_clusters_final.npz`` (from Python) is
// mis-matched with C++ segmentations and DER will be inflated even when
// clustering is "oracle".

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"
#include "wav_pcm_float32.h"

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static int json_int_field(const std::string& json, const char* key) {
  std::string pat = std::string("\"") + key + "\"\\s*:\\s*([0-9]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("JSON parse: missing int field \"") +
                             key + "\"");
  }
  return std::stoi(m[1].str());
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

static double json_chunk_step_sec(const std::string& json,
                                  double fallback_from_dur) {
  std::regex re("\"chunk_step_sec\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch m;
  if (std::regex_search(json, m, re)) {
    return std::stod(m[1].str());
  }
  return fallback_from_dur;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: full_segmentation_window_parity_test "
                 "<segmentation.onnx> <golden_utterance_dir> <wav_file>\n";
    std::cerr << "  Compares ORT output for every chunk to golden "
                 "binarized_segmentations.npz (same layout as diarize).\n";
    return 2;
  }
  try {
    const std::string onnx_path = argv[1];
    const std::string golden_dir = argv[2];
    const std::string wav_path = argv[3];
    if (onnx_path.size() < 5 ||
        onnx_path.substr(onnx_path.size() - 5) != ".onnx") {
      std::cerr << "Expected .onnx path\n";
      return 2;
    }
    const std::string json_path =
        onnx_path.substr(0, onnx_path.size() - 5) + ".json";
    const std::string bin_npz = golden_dir + "/binarized_segmentations.npz";

    const std::string json = read_text_file(json_path);
    const int sr_model = json_int_field(json, "sample_rate");
    const int num_channels = json_int_field(json, "num_channels");
    const int chunk_num_samples = json_int_field(json, "chunk_num_samples");
    const double chunk_dur_sec = json_double_field(json, "chunk_duration_sec");
    const double chunk_step_sec =
        json_chunk_step_sec(json, 0.1 * chunk_dur_sec);

    cnpy::npz_t bin = cnpy::npz_load(bin_npz);
    if (!bin.count("data")) {
      throw std::runtime_error("binarized_segmentations.npz missing data");
    }
    const cnpy::NpyArray& darr = bin["data"];
    if (darr.shape.size() != 3) {
      throw std::runtime_error("golden binarized must be rank-3 (C,F,K)");
    }
    const int64_t C_gold = static_cast<int64_t>(darr.shape[0]);
    const int F_gold = static_cast<int>(darr.shape[1]);
    const int K_gold = static_cast<int>(darr.shape[2]);
    const float* gold_ptr = darr.data<float>();

    int wav_sr = 0;
    std::vector<float> audio_in =
        wav_pcm::load_wav_pcm16_mono_float32(wav_path, wav_sr);
    std::vector<float> audio =
        wav_pcm::linear_resample(audio_in, wav_sr, sr_model);
    const int64_t num_samples = static_cast<int64_t>(audio.size());

    const int step_samples = static_cast<int>(
        std::lrint(chunk_step_sec * static_cast<double>(sr_model)));
    if (step_samples <= 0 || chunk_num_samples <= 0) {
      throw std::runtime_error("bad chunk/step samples from json");
    }
    int64_t num_chunks = 0;
    if (num_samples >= chunk_num_samples) {
      num_chunks = (num_samples - chunk_num_samples) / step_samples + 1;
    }
    const bool has_last =
        (num_samples < chunk_num_samples) ||
        ((num_samples - chunk_num_samples) % step_samples > 0);
    const int64_t total_chunks = num_chunks + (has_last ? 1 : 0);

    std::cout << "audio_samples=" << num_samples << " sr_model=" << sr_model
              << " total_chunks=" << total_chunks << " golden_C=" << C_gold
              << " F=" << F_gold << " K=" << K_gold << "\n";

    if (total_chunks != C_gold) {
      std::cerr << "FAIL: chunk count mismatch (wav-derived " << total_chunks
                << " vs golden " << C_gold << ")\n";
      std::cerr << "  Fix: use the same WAV file as dump_diarization_golden / "
                   "eval (same length and truncation).\n";
      return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "full_seg_parity");
    Ort::SessionOptions session_options;
    Ort::Session session(env, onnx_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
    Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, alloc);
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> seg_out(static_cast<size_t>(total_chunks) *
                               static_cast<size_t>(F_gold) *
                               static_cast<size_t>(K_gold));

    auto run_chunk = [&](const float* wav_data, int64_t chunk_idx) {
      const std::array<int64_t, 3> in_shape{1, num_channels, chunk_num_samples};
      Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(wav_data),
          static_cast<size_t>(1 * num_channels * chunk_num_samples),
          in_shape.data(), in_shape.size());
      const char* in_names[] = {in_name.get()};
      const char* out_names[] = {out_name.get()};
      auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                              out_names, 1);
      float* op = outs[0].GetTensorMutableData<float>();
      auto sh = outs[0].GetTensorTypeAndShapeInfo().GetShape();
      if (sh.size() != 3 || sh[0] != 1 || static_cast<int>(sh[1]) != F_gold ||
          static_cast<int>(sh[2]) != K_gold) {
        throw std::runtime_error("unexpected ORT output shape vs golden");
      }
      std::memcpy(
          &seg_out[static_cast<size_t>(chunk_idx) *
                   static_cast<size_t>(F_gold) * static_cast<size_t>(K_gold)],
          op, static_cast<size_t>(F_gold * K_gold) * sizeof(float));
    };

    for (int64_t c = 0; c < num_chunks; ++c) {
      const int64_t off = c * step_samples;
      std::vector<float> buf(static_cast<size_t>(num_channels) *
                                 static_cast<size_t>(chunk_num_samples),
                             0.f);
      for (int ch = 0; ch < num_channels; ++ch) {
        for (int i = 0; i < chunk_num_samples; ++i) {
          const int64_t si = off + i;
          float v = 0.f;
          if (si >= 0 && si < num_samples) {
            v = audio[static_cast<size_t>(si)];
          }
          buf[static_cast<size_t>(ch) * static_cast<size_t>(chunk_num_samples) +
              static_cast<size_t>(i)] = v;
        }
      }
      run_chunk(buf.data(), c);
    }
    if (has_last) {
      const int64_t off = num_chunks * step_samples;
      std::vector<float> buf(static_cast<size_t>(num_channels) *
                                 static_cast<size_t>(chunk_num_samples),
                             0.f);
      for (int ch = 0; ch < num_channels; ++ch) {
        for (int i = 0; i < chunk_num_samples; ++i) {
          const int64_t si = off + i;
          float v = 0.f;
          if (si >= 0 && si < num_samples) {
            v = audio[static_cast<size_t>(si)];
          }
          buf[static_cast<size_t>(ch) * static_cast<size_t>(chunk_num_samples) +
              static_cast<size_t>(i)] = v;
        }
      }
      run_chunk(buf.data(), num_chunks);
    }

    double mad = 0.0;
    size_t n_large = 0;
    const size_t n_el = seg_out.size();
    const float rtol = 1e-2f;
    const float atol = 1e-3f;
    size_t bad_allclose = 0;
    size_t bin_mismatch = 0;
    for (size_t i = 0; i < n_el; ++i) {
      const float a = seg_out[i];
      const float b = gold_ptr[i];
      const double d =
          std::abs(static_cast<double>(a) - static_cast<double>(b));
      mad = std::max(mad, d);
      if (d > 1e-3) {
        ++n_large;
      }
      if (!(std::abs(a - b) <= atol + rtol * std::abs(b))) {
        ++bad_allclose;
      }
      const bool ab = (a > 0.5f);
      const bool bb = (b > 0.5f);
      if (ab != bb) {
        ++bin_mismatch;
      }
    }
    const double frac_large =
        static_cast<double>(n_large) / static_cast<double>(n_el);
    const double frac_bin =
        static_cast<double>(bin_mismatch) / static_cast<double>(n_el);

    std::cout << "max_abs_diff(ORT_seg, golden_bin) = " << mad << "\n";
    std::cout << "fraction(|diff| > 1e-3) = " << frac_large << " (" << n_large
              << "/" << n_el << ")\n";
    std::cout << "binary_mismatch((a>0.5)!=(b>0.5)) count = " << bin_mismatch
              << " fraction = " << frac_bin << "\n";

    if (bad_allclose > 0) {
      int printed = 0;
      for (size_t i = 0; i < n_el; ++i) {
        const float a = seg_out[i];
        const float b = gold_ptr[i];
        if (!(std::abs(a - b) <= atol + rtol * std::abs(b))) {
          const int64_t K64 = K_gold;
          const int64_t F64 = F_gold;
          const int64_t idx = static_cast<int64_t>(i);
          const int64_t c = idx / (F64 * K64);
          const int64_t rem = idx % (F64 * K64);
          const int64_t f = rem / K64;
          const int64_t k = rem % K64;
          std::cerr << "  allclose_mismatch[" << printed << "] flat_i=" << i
                    << " chunk=" << c << " frame=" << f << " class=" << k
                    << " ORT=" << a << " golden=" << b
                    << " tol=" << (atol + rtol * std::abs(b)) << "\n";
          if (++printed >= 5) {
            break;
          }
        }
      }
    }

    const bool bin_ok = frac_bin <= 1e-4;
    const bool float_ok =
        (static_cast<double>(bad_allclose) / static_cast<double>(n_el)) <= 1e-4;
    if (!bin_ok) {
      std::cerr << "FAIL: binary segmentation disagreements exceed 1e-4 of "
                   "cells — oracle clusters are unsafe.\n";
      return 1;
    }
    if (!float_ok) {
      std::cerr << "FAIL: float allclose disagreements exceed 1e-4 of cells.\n";
      return 1;
    }
    if (bad_allclose > 0) {
      std::cout << "PASS (binary parity); NOTE: minor float allclose "
                   "violations vs golden ("
                << bad_allclose
                << " cells) — expected at hard 0/1 boundaries under rtol="
                << rtol << ", atol=" << atol << ".\n";
    } else {
      std::cout << "PASS full-window segmentation ORT vs golden "
                   "binarized_segmentations\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
