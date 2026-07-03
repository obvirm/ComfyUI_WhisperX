// SPDX-License-Identifier: MIT
// Milestone 0: frozen golden utterance — required NPZ keys, ranks, shapes, and
// word sizes. If you regenerate ``cpp/golden/...`` from
// ``dump_diarization_golden.py``, update the constexpr block below and re-run
// this test.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"

namespace {

// ---- Expected layout: ``callhome_eng_data_idx0_head120s`` (commit-frozen
// bundle) ---------------
constexpr int kChunks = 111;
constexpr int kFramesPerChunk = 589;
constexpr int kLocalSpeakers = 3;
constexpr int kEmbedDim = 256;
constexpr int kNumClusters = 2;  // from ``clustering.npz`` / VBx output width
constexpr std::size_t kSpeakerCountFrames = 7112;
constexpr int kVbxTrain = 154;
constexpr int kVbxLda = 128;
constexpr int kVbxGammaSpeakers = 9;
constexpr int kVbxTraceIters = 20;
constexpr int kFbankTf = 998;
constexpr int kMel = 80;
constexpr int kChunkWaveSamples = 160000;

void throw_shape(const cnpy::NpyArray& a, const std::string& ctx) {
  std::ostringstream oss;
  oss << ctx << ": bad shape [";
  for (size_t i = 0; i < a.shape.size(); ++i) {
    if (i) {
      oss << ", ";
    }
    oss << a.shape[i];
  }
  oss << "] word_size=" << a.word_size << " num_vals=" << a.num_vals;
  throw std::runtime_error(oss.str());
}

void require_key(const cnpy::npz_t& z, const char* name) {
  if (!z.count(name)) {
    throw std::runtime_error(std::string("missing npz key: ") + name);
  }
}

void require_rank(const cnpy::NpyArray& a, size_t rank, const char* ctx) {
  if (a.shape.size() != rank) {
    throw_shape(a, ctx);
  }
}

void require_shape3(const cnpy::NpyArray& a, size_t d0, size_t d1, size_t d2,
                    size_t word_size, const char* ctx) {
  require_rank(a, 3, ctx);
  if (a.shape[0] != d0 || a.shape[1] != d1 || a.shape[2] != d2 ||
      a.word_size != word_size) {
    throw_shape(a, ctx);
  }
}

void require_shape2(const cnpy::NpyArray& a, size_t d0, size_t d1,
                    size_t word_size, const char* ctx) {
  require_rank(a, 2, ctx);
  if (a.shape[0] != d0 || a.shape[1] != d1 || a.word_size != word_size) {
    throw_shape(a, ctx);
  }
}

void require_shape1(const cnpy::NpyArray& a, size_t d0, size_t word_size,
                    const char* ctx) {
  require_rank(a, 1, ctx);
  if (a.shape[0] != d0 || a.word_size != word_size) {
    throw_shape(a, ctx);
  }
}

void require_scalar_f64(const cnpy::NpyArray& a, const char* ctx) {
  if (a.num_vals != 1 || a.word_size != sizeof(double)) {
    throw_shape(a, ctx);
  }
}

void require_file_exists(const std::string& path, const char* ctx) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error(std::string("missing file (") + ctx +
                             "): " + path);
  }
}

void check_sliding_window_meta(const cnpy::npz_t& z, const char* file_ctx) {
  const char* keys[] = {"sliding_window_start", "sliding_window_duration",
                        "sliding_window_step"};
  for (const char* k : keys) {
    const std::string ctx = std::string(file_ctx) + "::" + k;
    require_key(z, k);
    require_scalar_f64(z.at(k), ctx.c_str());
  }
}

}  // namespace

#ifndef PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR
#define PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR \
  "cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s"
#endif
#ifndef PYANNOTE_FROZEN_GOLDEN_ROOT
#define PYANNOTE_FROZEN_GOLDEN_ROOT "cpp/golden/callhome_eng_idx0"
#endif

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cerr << "Usage: frozen_golden_inputs_test [utterance_dir]\n";
    std::cerr << "  Default utterance dir: "
                 "PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR (CMake).\n";
    std::cerr << "  Parent bundle: PYANNOTE_FROZEN_GOLDEN_ROOT (CMake).\n";
    std::cerr << "  Validates NPZ keys, ranks, shapes, and dtypes (word_size) "
                 "for the frozen bundle.\n";
    return 0;
  }
  const std::string utter =
      (argc >= 2) ? argv[1] : std::string(PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR);
  const std::string root = std::string(PYANNOTE_FROZEN_GOLDEN_ROOT);

  try {
    require_file_exists(root + "/receptive_field.json",
                        "bundle receptive_field");
    require_file_exists(root + "/pipeline_snapshot.json",
                        "bundle pipeline_snapshot");
    require_file_exists(root + "/manifest.json", "bundle manifest");
    require_file_exists(utter + "/label_mapping.json",
                        "utterance label_mapping");
    require_file_exists(utter + "/golden_speaker_bounds.json",
                        "utterance golden_speaker_bounds");

    cnpy::npz_t seg = cnpy::npz_load(utter + "/segmentations.npz");
    require_key(seg, "data");
    require_shape3(seg.at("data"), kChunks, kFramesPerChunk, kLocalSpeakers,
                   sizeof(float), "segmentations.data");
    check_sliding_window_meta(seg, "segmentations.npz");

    cnpy::npz_t bin = cnpy::npz_load(utter + "/binarized_segmentations.npz");
    require_key(bin, "data");
    require_shape3(bin.at("data"), kChunks, kFramesPerChunk, kLocalSpeakers,
                   sizeof(float), "binarized.data");
    check_sliding_window_meta(bin, "binarized_segmentations.npz");

    cnpy::npz_t emb = cnpy::npz_load(utter + "/embeddings.npz");
    require_key(emb, "embeddings");
    require_shape3(emb.at("embeddings"), kChunks, kLocalSpeakers, kEmbedDim,
                   sizeof(float), "embeddings.embeddings");

    cnpy::npz_t ort = cnpy::npz_load(utter + "/embedding_chunk0_spk0_ort.npz");
    require_key(ort, "fbank");
    require_key(ort, "weights");
    require_key(ort, "expected_embedding");
    require_shape3(ort.at("fbank"), 1u, static_cast<size_t>(kFbankTf),
                   static_cast<size_t>(kMel), sizeof(float), "ort.fbank");
    require_shape2(ort.at("weights"), 1u, static_cast<size_t>(kFramesPerChunk),
                   sizeof(float), "ort.weights");
    require_shape1(ort.at("expected_embedding"), static_cast<size_t>(kEmbedDim),
                   sizeof(float), "ort.expected_embedding");

    cnpy::npz_t clu = cnpy::npz_load(utter + "/clustering.npz");
    require_key(clu, "hard_clusters");
    require_key(clu, "soft_clusters");
    require_key(clu, "centroids");
    require_shape2(clu.at("hard_clusters"), static_cast<size_t>(kChunks),
                   static_cast<size_t>(kLocalSpeakers), sizeof(std::int8_t),
                   "clustering.hard_clusters");
    require_shape3(clu.at("soft_clusters"), static_cast<size_t>(kChunks),
                   static_cast<size_t>(kLocalSpeakers),
                   static_cast<size_t>(kNumClusters), sizeof(float),
                   "clustering.soft_clusters");
    require_shape2(clu.at("centroids"), static_cast<size_t>(kNumClusters),
                   static_cast<size_t>(kEmbedDim), sizeof(float),
                   "clustering.centroids");

    cnpy::npz_t hcf = cnpy::npz_load(utter + "/hard_clusters_final.npz");
    require_key(hcf, "hard_clusters");
    require_key(hcf, "inactive_speakers");
    require_shape2(hcf.at("hard_clusters"), static_cast<size_t>(kChunks),
                   static_cast<size_t>(kLocalSpeakers), sizeof(std::int8_t),
                   "hard_clusters_final.hard_clusters");
    require_shape2(hcf.at("inactive_speakers"), static_cast<size_t>(kChunks),
                   static_cast<size_t>(kLocalSpeakers), 1u,
                   "hard_clusters_final.inactive_speakers(bool)");

    cnpy::npz_t sci = cnpy::npz_load(utter + "/speaker_count_initial.npz");
    require_key(sci, "data");
    require_shape2(sci.at("data"), kSpeakerCountFrames, 1u,
                   sizeof(std::uint8_t), "speaker_count_initial.data");
    check_sliding_window_meta(sci, "speaker_count_initial.npz");

    cnpy::npz_t scc = cnpy::npz_load(utter + "/speaker_count_capped.npz");
    require_key(scc, "data");
    require_shape2(scc.at("data"), kSpeakerCountFrames, 1u, sizeof(std::int8_t),
                   "speaker_count_capped.data");
    check_sliding_window_meta(scc, "speaker_count_capped.npz");

    cnpy::npz_t disc_o =
        cnpy::npz_load(utter + "/discrete_diarization_overlap.npz");
    require_key(disc_o, "data");
    require_shape2(disc_o.at("data"), kSpeakerCountFrames, 2u, sizeof(float),
                   "discrete_overlap.data");
    check_sliding_window_meta(disc_o, "discrete_diarization_overlap.npz");

    cnpy::npz_t disc_e =
        cnpy::npz_load(utter + "/discrete_diarization_exclusive.npz");
    require_key(disc_e, "data");
    require_shape2(disc_e.at("data"), kSpeakerCountFrames, 2u, sizeof(float),
                   "discrete_exclusive.data");
    check_sliding_window_meta(disc_e, "discrete_diarization_exclusive.npz");

    cnpy::npz_t wav = cnpy::npz_load(utter + "/first_chunk_waveform.npz");
    require_key(wav, "waveforms");
    require_key(wav, "sample_rate");
    require_shape3(wav.at("waveforms"), 1u, 1u,
                   static_cast<size_t>(kChunkWaveSamples), sizeof(float),
                   "first_chunk_waveform");
    require_rank(wav.at("sample_rate"), 0, "sample_rate rank");
    if (wav.at("sample_rate").word_size != sizeof(std::int32_t) ||
        wav.at("sample_rate").num_vals != 1) {
      throw_shape(wav.at("sample_rate"), "sample_rate dtype");
    }

    cnpy::npz_t vbx = cnpy::npz_load(utter + "/vbx_reference.npz");
    const char* vbx_keys[] = {"x0",
                              "fea",
                              "Phi",
                              "ahc",
                              "train",
                              "train_n",
                              "pdist_condensed",
                              "linkage_Z",
                              "Fa",
                              "Fb",
                              "init_smoothing",
                              "n_vbx_iters",
                              "gamma_trace",
                              "pi_trace",
                              "train_chunk_idx",
                              "train_speaker_idx",
                              "fcluster_threshold"};
    for (const char* k : vbx_keys) {
      require_key(vbx, k);
    }
    require_shape2(vbx.at("x0"), static_cast<size_t>(kVbxTrain),
                   static_cast<size_t>(kVbxLda), sizeof(double), "vbx.x0");
    require_shape2(vbx.at("fea"), static_cast<size_t>(kVbxTrain),
                   static_cast<size_t>(kVbxLda), sizeof(double), "vbx.fea");
    require_shape1(vbx.at("Phi"), static_cast<size_t>(kVbxLda), sizeof(double),
                   "vbx.Phi");
    require_shape1(vbx.at("ahc"), static_cast<size_t>(kVbxTrain),
                   sizeof(std::int32_t), "vbx.ahc");
    require_shape2(vbx.at("train"), static_cast<size_t>(kVbxTrain),
                   static_cast<size_t>(kEmbedDim), sizeof(double), "vbx.train");
    require_shape2(vbx.at("train_n"), static_cast<size_t>(kVbxTrain),
                   static_cast<size_t>(kEmbedDim), sizeof(double),
                   "vbx.train_n");
    const size_t pd_len = static_cast<size_t>(kVbxTrain) *
                          (static_cast<size_t>(kVbxTrain) - 1) / 2;
    require_shape1(vbx.at("pdist_condensed"), pd_len, sizeof(double),
                   "vbx.pdist_condensed");
    require_shape1(vbx.at("linkage_Z"), static_cast<size_t>(kVbxTrain - 1) * 4u,
                   sizeof(double), "vbx.linkage_Z");
    require_shape1(vbx.at("train_chunk_idx"), static_cast<size_t>(kVbxTrain),
                   sizeof(std::int32_t), "vbx.train_chunk_idx");
    require_shape1(vbx.at("train_speaker_idx"), static_cast<size_t>(kVbxTrain),
                   sizeof(std::int32_t), "vbx.train_speaker_idx");
    require_scalar_f64(vbx.at("fcluster_threshold"), "vbx.fcluster_threshold");
    require_scalar_f64(vbx.at("Fa"), "vbx.Fa");
    require_scalar_f64(vbx.at("Fb"), "vbx.Fb");
    require_scalar_f64(vbx.at("init_smoothing"), "vbx.init_smoothing");
    if (vbx.at("n_vbx_iters").num_vals != 1 ||
        vbx.at("n_vbx_iters").word_size != sizeof(std::int32_t)) {
      throw_shape(vbx.at("n_vbx_iters"), "vbx.n_vbx_iters");
    }
    require_shape3(vbx.at("gamma_trace"), static_cast<size_t>(kVbxTraceIters),
                   static_cast<size_t>(kVbxTrain),
                   static_cast<size_t>(kVbxGammaSpeakers), sizeof(double),
                   "vbx.gamma_trace");
    require_shape2(vbx.at("pi_trace"), static_cast<size_t>(kVbxTraceIters),
                   static_cast<size_t>(kVbxGammaSpeakers), sizeof(double),
                   "vbx.pi_trace");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }

  std::cout << "PASS frozen golden inputs: " << utter << "\n";
  return 0;
}
