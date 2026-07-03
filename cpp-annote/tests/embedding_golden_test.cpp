// SPDX-License-Identifier: MIT
// Compare ONNX Runtime embedding output to Python golden (Torch
// ``embeddings.npz``).
//  * Default: chunk 0 / speaker 0 vs ``embedding_chunk0_spk0_ort.npz`` (ORT
//  inputs from dump).
//  * ``--all``: recompute ORT embeddings for every chunk/speaker from WAV +
//  golden
//    ``binarized_segmentations.npz`` (same recipe as ``CppAnnote::diarize``) vs
//    full ``embeddings.npz``.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"
#include "compute_fbank.h"
#include "embedding_ort_infer.h"
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

static double json_double_req(const std::string& json, const char* key) {
  std::string pat = std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("json missing \"") + key + "\"");
  }
  return std::stod(m[1].str());
}

static bool try_regex_double(const std::string& json,
                             const std::string& key_esc, double& out) {
  const std::string pat = "\"" + key_esc + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = std::stod(m[1].str());
  return true;
}

static bool try_embedding_exclude_overlap(const std::string& json, bool& out) {
  const std::string pat = "\"embedding_exclude_overlap\"\\s*:\\s*(true|false)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = (m[1].str() == "true");
  return true;
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0.f;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

static bool row_all_finite(const float* p, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(p[i])) {
      return false;
    }
  }
  return true;
}

static bool row_all_nan(const float* p, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (!std::isnan(p[i])) {
      return false;
    }
  }
  return true;
}

static bool allclose_row(const float* a, const float* b, int dim, float rtol,
                         float atol) {
  for (int i = 0; i < dim; ++i) {
    if (!(std::abs(a[i] - b[i]) <= atol + rtol * std::abs(b[i]))) {
      return false;
    }
  }
  return true;
}

static int run_chunk0(const std::string& onnx_path,
                      const std::string& golden_dir) {
  const std::string json_path =
      onnx_path.substr(0, onnx_path.size() - 5) + ".json";
  const std::string npz_path = golden_dir + "/embedding_chunk0_spk0_ort.npz";

  const std::string json = read_text_file(json_path);
  const int embedding_dim = json_int_field(json, "embedding_dim");

  cnpy::npz_t npz = cnpy::npz_load(npz_path);
  if (!npz.count("fbank") || !npz.count("weights") ||
      !npz.count("expected_embedding")) {
    throw std::runtime_error(
        "missing keys in embedding_chunk0_spk0_ort.npz — re-run "
        "cpp/scripts/dump_diarization_golden.py");
  }
  const cnpy::NpyArray& fb = npz["fbank"];
  const cnpy::NpyArray& wt = npz["weights"];
  const cnpy::NpyArray& exp = npz["expected_embedding"];

  if (fb.shape.size() != 3 || wt.shape.size() != 2 || exp.shape.size() != 1) {
    throw std::runtime_error("unexpected tensor ranks in golden npz");
  }
  if (static_cast<int>(exp.shape[0]) != embedding_dim) {
    throw std::runtime_error(
        "expected_embedding dim does not match embedding.json");
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embedding_golden_test");
  Ort::SessionOptions session_options;
  Ort::Session session(env, onnx_path.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions alloc;

  Ort::AllocatedStringPtr in0 = session.GetInputNameAllocated(0, alloc);
  Ort::AllocatedStringPtr in1 = session.GetInputNameAllocated(1, alloc);
  Ort::AllocatedStringPtr out0 = session.GetOutputNameAllocated(0, alloc);

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const float* fbank_ptr = fb.data<float>();
  const float* weights_ptr = wt.data<float>();
  std::array<int64_t, 3> fbank_shape{
      static_cast<int64_t>(fb.shape[0]),
      static_cast<int64_t>(fb.shape[1]),
      static_cast<int64_t>(fb.shape[2]),
  };
  std::array<int64_t, 2> weights_shape{
      static_cast<int64_t>(wt.shape[0]),
      static_cast<int64_t>(wt.shape[1]),
  };

  Ort::Value fbank_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(fbank_ptr), fb.num_vals, fbank_shape.data(),
      fbank_shape.size());
  Ort::Value weights_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(weights_ptr), wt.num_vals, weights_shape.data(),
      weights_shape.size());

  const char* in_names[] = {in0.get(), in1.get()};
  Ort::Value inputs[2];
  if (std::string(in0.get()) == "fbank") {
    inputs[0] = std::move(fbank_tensor);
    inputs[1] = std::move(weights_tensor);
  } else {
    inputs[0] = std::move(weights_tensor);
    inputs[1] = std::move(fbank_tensor);
  }
  const char* out_names[] = {out0.get()};
  auto outs =
      session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);

  float* out_ptr = outs[0].GetTensorMutableData<float>();
  auto out_info = outs[0].GetTensorTypeAndShapeInfo();
  const auto out_shape = out_info.GetShape();
  size_t out_n = 1;
  for (int64_t d : out_shape) {
    out_n *= static_cast<size_t>(d);
  }
  if (out_n != static_cast<size_t>(embedding_dim)) {
    std::ostringstream oss;
    oss << "ORT output size " << out_n << " != embedding_dim " << embedding_dim;
    throw std::runtime_error(oss.str());
  }

  const float* exp_ptr = exp.data<float>();
  const float mad = max_abs_diff(out_ptr, exp_ptr, embedding_dim);
  const float rtol = 1e-2f;
  const float atol = 1e-3f;
  bool pass = true;
  for (int i = 0; i < embedding_dim; ++i) {
    const float a = out_ptr[i];
    const float b = exp_ptr[i];
    if (!(std::abs(a - b) <= atol + rtol * std::abs(b))) {
      pass = false;
      break;
    }
  }

  std::cout << "max_abs_diff(embedding) = " << mad << "\n";
  if (!pass) {
    std::cerr << "FAIL: embedding allclose(rtol=" << rtol << ", atol=" << atol
              << ")\n";
    return 1;
  }
  std::cout << "PASS embedding chunk0 spk0 (ONNX Runtime vs golden NPZ)\n";
  return 0;
}

static std::string parent_dir(const std::string& p) {
  const size_t pos = p.find_last_of("/\\");
  if (pos == std::string::npos || pos == 0) {
    return ".";
  }
  return p.substr(0, pos);
}

static int run_all_chunks(const std::string& emb_onnx,
                          const std::string& golden_dir,
                          const std::string& wav_path,
                          const std::string& seg_onnx_path) {
  const std::string emb_json_path =
      emb_onnx.substr(0, emb_onnx.size() - 5) + ".json";
  const std::string seg_json_path =
      seg_onnx_path.substr(0, seg_onnx_path.size() - 5) + ".json";
  const std::string emb_json = read_text_file(emb_json_path);
  const std::string seg_json = read_text_file(seg_json_path);

  const int embed_sr =
      static_cast<int>(json_double_req(emb_json, "sample_rate"));
  const int embed_mel =
      static_cast<int>(json_double_req(emb_json, "num_mel_bins"));
  const float embed_fl_ms =
      static_cast<float>(json_double_req(emb_json, "frame_length_ms"));
  const float embed_fs_ms =
      static_cast<float>(json_double_req(emb_json, "frame_shift_ms"));
  const int embed_dim =
      static_cast<int>(json_double_req(emb_json, "embedding_dim"));
  const bool fbank_first =
      cppannote::embedding_ort::embedding_json_inputs_fbank_first(emb_json);

  const int sr_model =
      static_cast<int>(json_double_req(seg_json, "sample_rate"));
  const int chunk_num_samples =
      static_cast<int>(json_double_req(seg_json, "chunk_num_samples"));
  double chunk_step_sec = 0.1 * json_double_req(seg_json, "chunk_duration_sec");
  {
    double step_override = chunk_step_sec;
    if (try_regex_double(seg_json, "chunk_step_sec", step_override)) {
      chunk_step_sec = step_override;
    }
  }
  const double chunk_dur_sec = json_double_req(seg_json, "chunk_duration_sec");

  bool embedding_exclude_overlap = false;
  const std::string snap_path =
      parent_dir(golden_dir) + "/pipeline_snapshot.json";
  try {
    const std::string snap = read_text_file(snap_path);
    try_embedding_exclude_overlap(snap, embedding_exclude_overlap);
  } catch (const std::exception&) {
    // optional file
  }

  cnpy::npz_t gold_npz = cnpy::npz_load(golden_dir + "/embeddings.npz");
  cnpy::npz_t bin_npz =
      cnpy::npz_load(golden_dir + "/binarized_segmentations.npz");
  if (!gold_npz.count("embeddings") || !bin_npz.count("data")) {
    throw std::runtime_error(
        "embeddings.npz / binarized_segmentations.npz missing keys");
  }
  const cnpy::NpyArray& gold_arr = gold_npz["embeddings"];
  const cnpy::NpyArray& bin_arr = bin_npz["data"];
  if (gold_arr.shape.size() != 3 || bin_arr.shape.size() != 3) {
    throw std::runtime_error("expected rank-3 embeddings and binarized");
  }
  const int C = static_cast<int>(gold_arr.shape[0]);
  const int S = static_cast<int>(gold_arr.shape[1]);
  const int dim = static_cast<int>(gold_arr.shape[2]);
  const int Cb = static_cast<int>(bin_arr.shape[0]);
  const int F = static_cast<int>(bin_arr.shape[1]);
  const int Sb = static_cast<int>(bin_arr.shape[2]);
  if (Cb != C || Sb != S || dim != embed_dim) {
    throw std::runtime_error(
        "embeddings / binarized / embedding.json dim mismatch");
  }
  const float* gold = gold_arr.data<float>();
  std::vector<float> binarized(bin_arr.data<float>(),
                               bin_arr.data<float>() + bin_arr.num_vals);

  int wav_sr_in = 0;
  std::vector<float> audio_in =
      wav_pcm::load_wav_pcm16_mono_float32(wav_path, wav_sr_in);
  std::vector<float> audio =
      wav_pcm::linear_resample(audio_in, wav_sr_in, sr_model);
  const int64_t num_samples = static_cast<int64_t>(audio.size());

  const int step_samples = static_cast<int>(
      std::lrint(chunk_step_sec * static_cast<double>(sr_model)));
  if (step_samples <= 0 || chunk_num_samples <= 0) {
    throw std::runtime_error("bad chunk/step samples from segmentation json");
  }
  int64_t num_chunks = 0;
  if (num_samples >= chunk_num_samples) {
    num_chunks = (num_samples - chunk_num_samples) / step_samples + 1;
  }
  const bool has_last = (num_samples < chunk_num_samples) ||
                        ((num_samples - chunk_num_samples) % step_samples > 0);
  const int64_t total_chunks = num_chunks + (has_last ? 1 : 0);
  if (total_chunks != static_cast<int64_t>(C)) {
    std::ostringstream oss;
    oss << "WAV yields total_chunks=" << total_chunks
        << " but golden embeddings have C=" << C
        << " (use the same WAV as the golden dump)";
    throw std::runtime_error(oss.str());
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embedding_golden_test_all");
  Ort::SessionOptions session_options;
  Ort::Session session(env, emb_onnx.c_str(), session_options);
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions alloc;

  const int min_num_samples =
      cppannote::embedding_ort::discover_min_num_samples_embedding(
          session, mem, alloc, fbank_first, embed_sr, embed_mel, embed_fl_ms,
          embed_fs_ms, embed_dim);
  int min_num_frames = cppannote::embedding_ort::fbank_num_frames_for_samples(
      embed_sr, embed_mel, embed_fl_ms, embed_fs_ms, min_num_samples);
  if (min_num_frames < 1) {
    min_num_frames = 1;
  }

  std::cerr << "[EMBEDDING_FULL] min_num_samples=" << min_num_samples
            << " min_num_frames=" << min_num_frames << " F=" << F
            << " chunk_num_samples(seg_json)=" << chunk_num_samples
            << " embed_sr=" << embed_sr << " sr_model=" << sr_model
            << " embedding_exclude_overlap="
            << (embedding_exclude_overlap ? 1 : 0) << "\n";

  const size_t nslots = static_cast<size_t>(C) * static_cast<size_t>(S);
  std::vector<int> slot_n_keep(nslots, -1);
  std::vector<int> slot_Tf(nslots, -1);
  std::vector<int> slot_min_nf_seg(nslots, 0);
  std::vector<float> slot_sum_clean(nslots, 0.f);
  std::vector<char> slot_prefer_clean(nslots, 0);

  std::vector<float> ort_emb(static_cast<size_t>(C) * static_cast<size_t>(S) *
                                 static_cast<size_t>(dim),
                             std::numeric_limits<float>::quiet_NaN());

  for (int64_t ci = 0; ci < total_chunks; ++ci) {
    const int64_t off =
        (ci < num_chunks) ? ci * step_samples : num_chunks * step_samples;
    std::vector<float> chunk_mono(static_cast<size_t>(chunk_num_samples));
    for (int i = 0; i < chunk_num_samples; ++i) {
      const int64_t si = off + i;
      float v = 0.f;
      if (si >= 0 && si < num_samples) {
        v = audio[static_cast<size_t>(si)];
      }
      chunk_mono[static_cast<size_t>(i)] = v;
    }
    std::vector<float> chunk_for_fbank = chunk_mono;
    int wav_sr_use = sr_model;
    if (sr_model != embed_sr) {
      chunk_for_fbank =
          wav_pcm::linear_resample(chunk_mono, sr_model, embed_sr);
      wav_sr_use = embed_sr;
    }
    const int chunk_embed_samples = static_cast<int>(chunk_for_fbank.size());
    const int min_nf_seg = embedding_exclude_overlap
                               ? static_cast<int>(std::ceil(
                                     static_cast<double>(F) *
                                     static_cast<double>(min_num_samples) /
                                     static_cast<double>(chunk_embed_samples)))
                               : -1;
    int Tf = 0;
    int Mfb = 0;
    std::vector<float> fbank_all;
    cppannote::fbank::wespeaker_like_fbank(
        static_cast<float>(wav_sr_use), embed_mel, embed_fl_ms, embed_fs_ms,
        chunk_for_fbank.data(), static_cast<int>(chunk_for_fbank.size()),
        fbank_all, Tf, Mfb);
    if (Tf < 1 || Mfb != embed_mel) {
      throw std::runtime_error(
          "embedding fbank: unexpected frames or mel dimension");
    }
    for (int sp = 0; sp < S; ++sp) {
      std::vector<float> clean_col(static_cast<size_t>(F), 0.f);
      std::vector<float> full_col(static_cast<size_t>(F), 0.f);
      for (int f = 0; f < F; ++f) {
        float rowsum = 0.f;
        for (int j = 0; j < S; ++j) {
          rowsum += binarized[static_cast<size_t>(
              (static_cast<size_t>(ci) * static_cast<size_t>(F) +
               static_cast<size_t>(f)) *
                  static_cast<size_t>(S) +
              static_cast<size_t>(j))];
        }
        const float overlap_ok = (rowsum < 2.f - 1e-5f) ? 1.f : 0.f;
        const float v = binarized[static_cast<size_t>(
            (static_cast<size_t>(ci) * static_cast<size_t>(F) +
             static_cast<size_t>(f)) *
                static_cast<size_t>(S) +
            static_cast<size_t>(sp))];
        full_col[static_cast<size_t>(f)] = v;
        clean_col[static_cast<size_t>(f)] = v * overlap_ok;
      }
      float sum_clean = 0.f;
      for (int f = 0; f < F; ++f) {
        if (clean_col[static_cast<size_t>(f)] > 0.5f) {
          sum_clean += 1.f;
        }
      }
      const bool prefer_clean =
          (min_nf_seg < 0) ? true
                           : (sum_clean > static_cast<float>(min_nf_seg));
      const std::vector<float>& src = prefer_clean ? clean_col : full_col;
      int n_active_seg = 0;
      for (int f = 0; f < F; ++f) {
        if (src[static_cast<size_t>(f)] > 0.5f) {
          ++n_active_seg;
        }
      }
      const size_t sidx = static_cast<size_t>(ci) * static_cast<size_t>(S) +
                          static_cast<size_t>(sp);
      slot_n_keep[sidx] = n_active_seg;
      slot_Tf[sidx] = Tf;
      slot_min_nf_seg[sidx] = min_nf_seg;
      slot_sum_clean[sidx] = sum_clean;
      slot_prefer_clean[sidx] = prefer_clean ? 1 : 0;
      float* dst = &ort_emb[(static_cast<size_t>(ci) * static_cast<size_t>(S) +
                             static_cast<size_t>(sp)) *
                            static_cast<size_t>(dim)];
      cppannote::embedding_ort::run_embedding_ort(
          session, mem, alloc, fbank_first, fbank_all.data(), Tf, Mfb,
          src.data(), F, dst, dim);
    }
  }

  const float rtol = 1e-2f;
  const float atol = 1e-3f;
  size_t both_nan = 0;
  size_t both_finite = 0;
  size_t gold_finite_ort_nan = 0;
  size_t gold_nan_ort_finite = 0;
  size_t other_mismatch = 0;
  size_t slot_fail_allclose = 0;
  double sum_mad = 0.0;
  float global_max = 0.f;
  int worst_c = 0;
  int worst_s = 0;
  float worst_slot_mad = 0.f;

  for (int c = 0; c < C; ++c) {
    for (int s = 0; s < S; ++s) {
      const float* gp = &gold[static_cast<size_t>(
          (static_cast<size_t>(c) * static_cast<size_t>(S) +
           static_cast<size_t>(s)) *
          static_cast<size_t>(dim))];
      const float* op = &ort_emb[static_cast<size_t>(
          (static_cast<size_t>(c) * static_cast<size_t>(S) +
           static_cast<size_t>(s)) *
          static_cast<size_t>(dim))];
      const bool gf = row_all_finite(gp, dim);
      const bool of = row_all_finite(op, dim);
      const bool gn = row_all_nan(gp, dim);
      const bool on = row_all_nan(op, dim);
      if (gn && on) {
        ++both_nan;
        continue;
      }
      if (gf && of) {
        ++both_finite;
        const float mad = max_abs_diff(gp, op, static_cast<size_t>(dim));
        sum_mad += static_cast<double>(mad);
        if (mad > global_max) {
          global_max = mad;
        }
        if (mad > worst_slot_mad) {
          worst_slot_mad = mad;
          worst_c = c;
          worst_s = s;
        }
        if (!allclose_row(op, gp, dim, rtol, atol)) {
          ++slot_fail_allclose;
        }
        continue;
      }
      if (gf && on) {
        ++gold_finite_ort_nan;
        continue;
      }
      if (gn && of) {
        ++gold_nan_ort_finite;
        continue;
      }
      ++other_mismatch;
    }
  }

  const double mean_mad =
      both_finite ? sum_mad / static_cast<double>(both_finite) : 0.0;
  const double fail_frac = both_finite
                               ? static_cast<double>(slot_fail_allclose) /
                                     static_cast<double>(both_finite)
                               : 0.0;

  std::cout << "embedding_full: C=" << C << " S=" << S << " dim=" << dim
            << "\n";
  std::cout << "  slots both_nan=" << both_nan << " both_finite=" << both_finite
            << " gold_finite_ort_nan=" << gold_finite_ort_nan
            << " gold_nan_ort_finite=" << gold_nan_ort_finite
            << " other=" << other_mismatch << "\n";
  std::cout << "  mean_max_abs_diff_per_both_finite_slot=" << mean_mad << "\n";
  std::cout << "  worst_both_finite (c,s)=(" << worst_c << "," << worst_s
            << ") max_abs=" << worst_slot_mad << "\n";
  std::cout << "  global_max_abs_diff=" << global_max << "\n";
  std::cout << "  allclose_fail_slots=" << slot_fail_allclose
            << " fail_frac=" << fail_frac << "\n";

  std::map<int, int> deficit_hist;
  const char* csv_path = std::getenv("EMBEDDING_FULL_DUMP_CSV");
  std::ofstream csv_out;
  std::size_t csv_data_rows = 0;
  if (csv_path != nullptr && csv_path[0] != '\0') {
    csv_out.open(csv_path);
    if (!csv_out) {
      throw std::runtime_error(
          std::string("cannot open EMBEDDING_FULL_DUMP_CSV: ") + csv_path);
    }
    csv_out << "c,s,n_active_seg,Tf,F,min_num_frames,min_nf_seg,sum_clean,"
               "prefer_clean,deficit\n";
  }
  for (int c = 0; c < C; ++c) {
    for (int s = 0; s < S; ++s) {
      const size_t sidx = static_cast<size_t>(c) * static_cast<size_t>(S) +
                          static_cast<size_t>(s);
      const float* gp = &gold[sidx * static_cast<size_t>(dim)];
      const float* op = &ort_emb[sidx * static_cast<size_t>(dim)];
      if (!row_all_finite(gp, dim)) {
        continue;
      }
      if (row_all_finite(op, dim)) {
        continue;
      }
      const int nk = slot_n_keep[sidx];
      const int deficit = min_num_frames - nk;
      deficit_hist[deficit] +=
          1;  // nk = segmentation frames active in used mask
      if (csv_out.is_open()) {
        csv_out << c << "," << s << "," << nk << "," << slot_Tf[sidx] << ","
                << F << "," << min_num_frames << "," << slot_min_nf_seg[sidx]
                << "," << slot_sum_clean[sidx] << ","
                << static_cast<int>(slot_prefer_clean[sidx]) << "," << deficit
                << "\n";
        ++csv_data_rows;
      }
    }
  }
  if (!deficit_hist.empty()) {
    for (const auto& pr : deficit_hist) {
      std::cerr
          << "[EMBEDDING_NAN_MISMATCH] deficit=min_num_frames-n_active_seg="
          << pr.first << " count=" << pr.second << "\n";
    }
  }
  if (csv_out.is_open()) {
    csv_out.close();
    if (csv_data_rows > 0) {
      std::cerr << "[EMBEDDING_FULL] wrote " << csv_data_rows
                << " nan-mismatch row(s) to " << csv_path << "\n";
    }
  }

  const char* ff_env = std::getenv("EMBEDDING_FULL_MAX_FAIL_FRAC");
  const double max_fail_frac = ff_env ? std::stod(ff_env) : 0.10;
  const char* nan_env = std::getenv("EMBEDDING_FULL_MAX_OR_NAN_SLOTS");
  const std::size_t max_ort_nan =
      nan_env ? static_cast<std::size_t>(std::stoll(nan_env)) : 200u;
  if (gold_finite_ort_nan > max_ort_nan) {
    std::cerr << "FAIL: gold_finite_ort_nan=" << gold_finite_ort_nan
              << " > EMBEDDING_FULL_MAX_OR_NAN_SLOTS=" << max_ort_nan << "\n";
    return 1;
  }
  if (gold_nan_ort_finite > 0u) {
    std::cerr << "FAIL: gold_nan_ort_finite=" << gold_nan_ort_finite
              << " (unexpected)\n";
    return 1;
  }
  if (other_mismatch > 0u) {
    std::cerr << "FAIL: partial NaN / mixed rows not handled: "
              << other_mismatch << "\n";
    return 1;
  }
  if (fail_frac > max_fail_frac) {
    std::cerr << "FAIL: allclose fail_frac " << fail_frac << " > "
              << max_fail_frac
              << " (set EMBEDDING_FULL_MAX_FAIL_FRAC to relax)\n";
    return 1;
  }
  std::cout << "PASS embedding full tensor (ORT vs embeddings.npz)\n";
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--help") {
    std::cerr << "Usage:\n"
              << "  embedding_golden_test <community1-embedding.onnx> "
                 "<golden_utterance_dir>\n"
              << "    chunk0/spk0 vs embedding_chunk0_spk0_ort.npz\n"
              << "  embedding_golden_test <embedding.onnx> "
                 "<golden_utterance_dir> --all <audio.wav> "
                 "<community1-segmentation.onnx>\n"
              << "    ORT vs full embeddings.npz (same WAV as dump; optional "
                 "pipeline_snapshot.json parent for "
                 "embedding_exclude_overlap)\n"
              << "Env:\n"
              << "  EMBEDDING_FULL_MAX_FAIL_FRAC (default 0.10) on "
                 "mutually-finite slots vs allclose.\n"
              << "  EMBEDDING_FULL_MAX_OR_NAN_SLOTS (default 200) max slots "
                 "where golden is finite but ORT is "
                 "all-NaN; set 0 for strict NaN-mask parity.\n"
              << "  EMBEDDING_FULL_DUMP_CSV=/path/rows.csv  gold-finite & "
                 "ORT-NaN diagnostics per slot "
                 "(n_active_seg = used-mask frames > 0.5).\n";
    return 0;
  }
  if (argc >= 4 && std::string(argv[3]) == "--all") {
    if (argc != 6) {
      std::cerr << "Expected: ... --all <wav> <segmentation.onnx>\n";
      return 2;
    }
    try {
      return run_all_chunks(argv[1], argv[2], argv[4], argv[5]);
    } catch (const std::exception& e) {
      std::cerr << "FAIL: " << e.what() << "\n";
      return 1;
    }
  }
  if (argc != 3) {
    std::cerr << "Usage: embedding_golden_test <community1-embedding.onnx> "
                 "<golden_utterance_dir>\n";
    std::cerr << "  requires embedding_chunk0_spk0_ort.npz (re-run "
                 "dump_diarization_golden.py)\n";
    return 2;
  }
  try {
    return run_chunk0(argv[1], argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
