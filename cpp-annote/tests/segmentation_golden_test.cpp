// SPDX-License-Identifier: MIT
// Compare first-chunk ONNX Runtime segmentation output to Python golden NPZ.

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

static bool json_bool_field(const std::string& json, const char* key) {
  std::string pat = std::string("\"") + key + "\"\\s*:\\s*(true|false)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("JSON parse: missing bool field \"") +
                             key + "\"");
  }
  return m[1].str() == "true";
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("size mismatch in max_abs_diff");
  }
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: segmentation_golden_test "
                 "<community1-segmentation.onnx> <golden_utterance_dir>\n";
    std::cerr << "  golden_utterance_dir must contain segmentations.npz and "
                 "first_chunk_waveform.npz\n";
    std::cerr << "  (re-run cpp/scripts/dump_diarization_golden.py if "
                 "first_chunk_waveform.npz is missing)\n";
    return 2;
  }
  const std::string onnx_path = argv[1];
  const std::string golden_dir = argv[2];
  if (onnx_path.size() < 5 ||
      onnx_path.substr(onnx_path.size() - 5) != ".onnx") {
    std::cerr << "Expected .onnx path\n";
    return 2;
  }
  const std::string json_path =
      onnx_path.substr(0, onnx_path.size() - 5) + ".json";

  const std::string seg_npz = golden_dir + "/segmentations.npz";
  const std::string wav_npz = golden_dir + "/first_chunk_waveform.npz";

  const std::string json = read_text_file(json_path);
  const int chunk_num_samples = json_int_field(json, "chunk_num_samples");
  const int num_channels = json_int_field(json, "num_channels");
  (void)json_bool_field(
      json,
      "export_includes_powerset_to_multilabel");  // sanity: exported graph

  cnpy::npz_t w_npz = cnpy::npz_load(wav_npz);
  if (!w_npz.count("waveforms")) {
    throw std::runtime_error(
        "first_chunk_waveform.npz missing 'waveforms' — re-run golden dump");
  }
  const cnpy::NpyArray& wav_arr = w_npz["waveforms"];
  const float* wav_data = nullptr;
  if (wav_arr.shape.size() == 3) {
    if (static_cast<int>(wav_arr.shape[0]) != 1 ||
        static_cast<int>(wav_arr.shape[1]) != num_channels ||
        static_cast<int>(wav_arr.shape[2]) != chunk_num_samples) {
      std::ostringstream oss;
      oss << "waveforms shape mismatch: got [";
      for (size_t i = 0; i < wav_arr.shape.size(); ++i) {
        oss << (i ? ", " : "") << wav_arr.shape[i];
      }
      oss << "] expected [1, " << num_channels << ", " << chunk_num_samples
          << "]";
      throw std::runtime_error(oss.str());
    }
    wav_data = wav_arr.data<float>();
  } else if (wav_arr.shape.size() == 2) {
    // Legacy bundles: Audio.crop returns (channel, time) only.
    if (static_cast<int>(wav_arr.shape[0]) != num_channels ||
        static_cast<int>(wav_arr.shape[1]) != chunk_num_samples) {
      std::ostringstream oss;
      oss << "waveforms (channel, time) shape mismatch: got [";
      for (size_t i = 0; i < wav_arr.shape.size(); ++i) {
        oss << (i ? ", " : "") << wav_arr.shape[i];
      }
      oss << "] expected [" << num_channels << ", " << chunk_num_samples << "]";
      throw std::runtime_error(oss.str());
    }
    wav_data = wav_arr.data<float>();
  } else {
    throw std::runtime_error(
        "waveforms must be rank-3 (batch, channel, samples) or rank-2 "
        "(channel, samples)");
  }

  cnpy::npz_t seg_npz_m = cnpy::npz_load(seg_npz);
  if (!seg_npz_m.count("data")) {
    throw std::runtime_error("segmentations.npz missing 'data'");
  }
  const cnpy::NpyArray& seg_arr = seg_npz_m["data"];
  if (seg_arr.shape.size() != 3) {
    throw std::runtime_error("segmentations data must be rank-3");
  }
  const size_t n_frames = seg_arr.shape[1];
  const size_t n_classes = seg_arr.shape[2];
  const float* seg_ref = seg_arr.data<float>();
  const size_t out_el = n_frames * n_classes;
  std::vector<float> golden_first_chunk(out_el);
  std::memcpy(golden_first_chunk.data(), seg_ref, out_el * sizeof(float));

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "segmentation_golden_test");
  Ort::SessionOptions session_options;
  Ort::Session session(env, onnx_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions alloc;
  Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
  Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, alloc);

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 3> in_shape{1, num_channels, chunk_num_samples};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(wav_data),
      static_cast<size_t>(1 * num_channels * chunk_num_samples),
      in_shape.data(), in_shape.size());

  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};
  auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                          out_names, 1);

  float* out_ptr = outs[0].GetTensorMutableData<float>();
  auto out_info = outs[0].GetTensorTypeAndShapeInfo();
  const auto out_shape = out_info.GetShape();
  if (out_shape.size() != 3 || out_shape[0] != 1 ||
      static_cast<size_t>(out_shape[1]) != n_frames ||
      static_cast<size_t>(out_shape[2]) != n_classes) {
    std::ostringstream oss;
    oss << "Unexpected ORT output shape: [";
    for (size_t i = 0; i < out_shape.size(); ++i) {
      oss << (i ? ", " : "") << out_shape[i];
    }
    oss << "] expected [1, " << n_frames << ", " << n_classes << "]";
    throw std::runtime_error(oss.str());
  }
  std::vector<float> ort_out(out_el);
  std::memcpy(ort_out.data(), out_ptr, out_el * sizeof(float));

  const float mad = max_abs_diff(ort_out, golden_first_chunk);
  const float rtol = 1e-2f;
  const float atol = 1e-3f;
  bool pass = true;
  for (size_t i = 0; i < out_el; ++i) {
    const float a = ort_out[i];
    const float b = golden_first_chunk[i];
    if (!(std::abs(a - b) <= atol + rtol * std::abs(b))) {
      pass = false;
      break;
    }
  }

  std::cout << "max_abs_diff(first_chunk) = " << mad << "\n";
  if (!pass) {
    std::cerr << "FAIL: elementwise allclose(rtol=" << rtol << ", atol=" << atol
              << ") did not hold\n";
    return 1;
  }
  std::cout << "PASS segmentation first chunk (ONNX Runtime vs golden NPZ)\n";
  return 0;
}
