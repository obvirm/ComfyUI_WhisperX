// SPDX-License-Identifier: MIT
// CLI for diarization; all engine details are hidden behind the pimpl-based
// CppAnnote class (cpp-annote.h).  Audio is fed through a streaming session
// managed via create_stream / add_audio_to_stream / stop_stream.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpp-annote.h"
#include "wav_pcm_float32.h"

namespace fs = std::filesystem;

static std::string get_arg(int argc, char** argv, const char* key,
                           const std::string& def = "") {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == key) {
      return argv[i + 1];
    }
  }
  return def;
}

static bool has_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) {
      return true;
    }
  }
  return false;
}

struct DiarJob {
  std::string wav;
  std::string out;
};

static void trim_inplace(std::string& s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
    s.erase(s.begin());
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
    s.pop_back();
  }
}

static std::vector<std::string> split_tab_row(const std::string& line) {
  std::vector<std::string> cols;
  size_t i = 0;
  while (i < line.size()) {
    const size_t j = line.find('\t', i);
    if (j == std::string::npos) {
      cols.push_back(line.substr(i));
      break;
    }
    cols.push_back(line.substr(i, j - i));
    i = j + 1;
  }
  for (auto& c : cols) {
    trim_inplace(c);
  }
  return cols;
}

static std::vector<DiarJob> load_manifest_jobs(const std::string& manifest_path,
                                               const std::string& out_dir) {
  std::vector<DiarJob> jobs;
  std::ifstream f(manifest_path);
  if (!f) {
    throw std::runtime_error("open manifest failed: " + manifest_path);
  }
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    trim_inplace(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto cols = split_tab_row(line);
    while (!cols.empty() && cols.back().empty()) {
      cols.pop_back();
    }
    if (cols.size() == 1) {
      if (out_dir.empty()) {
        throw std::runtime_error(
            "manifest " + manifest_path + " line " + std::to_string(lineno) +
            ": single-column lines require --out-dir (wav path only)");
      }
      const fs::path wv(cols[0]);
      jobs.push_back(
          {cols[0],
           (fs::path(out_dir) / (wv.stem().string() + ".json")).string()});
    } else if (cols.size() == 2) {
      jobs.push_back({cols[0], cols[1]});
    } else {
      throw std::runtime_error(
          "manifest " + manifest_path + " line " + std::to_string(lineno) +
          ": expected 1 or 2 tab-separated fields (wav | wav<TAB>out)");
    }
  }
  if (jobs.empty()) {
    throw std::runtime_error("manifest has no data rows: " + manifest_path);
  }
  return jobs;
}

static std::vector<DiarJob> load_wav_list_jobs(const std::string& list_path,
                                               const fs::path& out_dir) {
  std::vector<DiarJob> jobs;
  std::ifstream f(list_path);
  if (!f) {
    throw std::runtime_error("open wav-list failed: " + list_path);
  }
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    trim_inplace(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const fs::path wv(line);
    const std::string stem = wv.stem().string();
    jobs.push_back({line, (out_dir / (stem + ".json")).string()});
  }
  if (jobs.empty()) {
    throw std::runtime_error("wav-list has no paths: " + list_path);
  }
  return jobs;
}

static void print_timing(const char* tag, const std::string& path,
                         double audio_sec, double wall_sec) {
  char buf[256];
  if (audio_sec > 0. && wall_sec > 0.) {
    std::snprintf(buf, sizeof(buf),
                  "[%s] %s  audio=%.2fs  wall=%.3fs  RTF=%.2fx\n", tag,
                  path.c_str(), audio_sec, wall_sec, audio_sec / wall_sec);
  } else {
    std::snprintf(buf, sizeof(buf), "[%s] %s  audio=%.2fs  wall=%.3fs\n", tag,
                  path.c_str(), audio_sec, wall_sec);
  }
  std::cerr << buf;
}

static void run_diarize(cppannote::CppAnnote& engine,
                        const std::vector<DiarJob>& jobs,
                        double cluster_cadence, double analyze_cadence,
                        bool continue_on_error) {
  int n_fail = 0;
  double total_audio_sec = 0.;
  double total_wall_sec = 0.;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    try {
      const DiarJob& job = jobs[i];

      int wav_sr = 0;
      std::vector<float> mono =
          wav_pcm::load_wav_pcm16_mono_float32(job.wav, wav_sr);
      const double audio_sec = wav_sr > 0 ? static_cast<double>(mono.size()) /
                                                static_cast<double>(wav_sr)
                                          : 0.;
      if (!job.out.empty()) {
        const fs::path outp(job.out);
        const fs::path parent = outp.parent_path();
        if (!parent.empty()) {
          fs::create_directories(parent);
        }
      }

      int32_t stream_id = engine.create_stream(cluster_cadence, analyze_cadence);
      engine.start_stream(stream_id);

      constexpr double kSimChunkSec = 1.0;
      const std::size_t chunk_samples = static_cast<std::size_t>(
          std::max(1., kSimChunkSec * static_cast<double>(wav_sr)));
      const auto t0 = std::chrono::steady_clock::now();
      std::size_t offset = 0;
      while (offset < mono.size()) {
        const std::size_t n = std::min(chunk_samples, mono.size() - offset);
        engine.add_audio_to_stream(stream_id, mono.data() + offset,
                                   static_cast<uint64_t>(n),
                                   static_cast<int32_t>(wav_sr));
        offset += n;
      }

      cppannote::DiarizationResults results = engine.stop_stream(stream_id);
      engine.free_stream(stream_id);

      const auto t1 = std::chrono::steady_clock::now();
      const double wall_sec = std::chrono::duration<double>(t1 - t0).count();

      if (job.out.empty()) {
        results.write_json(std::cout);
      } else {
        results.write_json(job.out);
        if (results.turns.empty()) {
          std::cerr << "[diarize] Wrote empty diarization -> " << job.out
                    << "\n";
        } else {
          std::cerr << "[diarize] Wrote " << job.out << " ("
                    << results.turns.size() << " turns)\n";
        }
      }

      total_audio_sec += audio_sec;
      total_wall_sec += wall_sec;
      print_timing("diarize", job.wav, audio_sec, wall_sec);
    } catch (const std::exception& e) {
      std::cerr << "ERROR";
      if (jobs.size() > 1) {
        std::cerr << " [" << (i + 1) << "/" << jobs.size() << "] "
                  << jobs[i].wav;
      }
      std::cerr << ": " << e.what() << "\n";
      ++n_fail;
      if (!continue_on_error) {
        throw;
      }
    }
  }
  if (jobs.size() > 1) {
    print_timing("diarize total", std::to_string(jobs.size()) + " files",
                 total_audio_sec, total_wall_sec);
  }
  if (n_fail > 0) {
    throw std::runtime_error(std::to_string(n_fail) + " job(s) failed");
  }
}

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help")) {
    std::cerr
        << "cpp-annote-cli — WAV + compiled-in ORT models + VBx -> "
           "diarization JSON.\n\n"
        << "Audio is fed through a streaming session that caches ORT results "
           "incrementally\n"
        << "and runs VBx clustering on a configurable cadence.\n\n"
        << "Single file:\n"
        << "  --wav PATH\n"
        << "  --out PATH                 output diarization.json (omit to "
           "print to stdout)\n\n"
        << "Multi-file — tab-separated manifest (one job per line, # comments "
           "OK):\n"
        << "  --manifest PATH\n"
        << "    1 field:   wav   (requires --out-dir -> OUT/<wav_stem>.json)\n"
        << "    2 fields:  wav<TAB>out.json\n"
        << "  --out-dir PATH             required for 1-column manifest lines; "
           "also for --wav-list\n\n"
        << "Multi-file — one WAV path per line:\n"
        << "  --wav-list PATH            requires --out-dir; writes "
           "OUT/<stem>.json per line\n\n"
        << "Tuning:\n"
        << "  --cluster-cadence N        re-cluster every N seconds of new "
           "audio (default 2.0)\n"
        << "  --analyze-cadence N        step between seg+emb model runs in "
           "seconds (>0, <=10; default: model's chunk_step_sec)\n\n"
        << "Model override (use external ONNX files instead of compiled-in "
           "models):\n"
        << "  --segmentation-onnx PATH   path to segmentation .onnx file\n"
        << "  --embedding-onnx PATH      path to embedding .onnx file\n\n"
        << "Other:\n"
        << "  --continue-on-error            print error and continue; exit 1 "
           "if any failed\n";
    return 2;
  }
  try {
    const std::string manifest_path = get_arg(argc, argv, "--manifest");
    const std::string wav_list_path = get_arg(argc, argv, "--wav-list");
    const std::string wav_path = get_arg(argc, argv, "--wav");
    const std::string out_dir = get_arg(argc, argv, "--out-dir");
    const bool continue_on_error = has_flag(argc, argv, "--continue-on-error");

    const std::string cluster_str = get_arg(argc, argv, "--cluster-cadence");
    const double cluster_cadence =
        cluster_str.empty() ? 2.0 : std::stod(cluster_str);
    const std::string analyze_str = get_arg(argc, argv, "--analyze-cadence");
    const double analyze_cadence =
        analyze_str.empty() ? 0.0 : std::stod(analyze_str);

    const std::string seg_onnx = get_arg(argc, argv, "--segmentation-onnx");
    const std::string emb_onnx = get_arg(argc, argv, "--embedding-onnx");

    std::vector<DiarJob> jobs;
    if (!manifest_path.empty()) {
      if (!wav_list_path.empty() || !wav_path.empty()) {
        throw std::runtime_error(
            "use only one of --manifest, --wav-list, or --wav");
      }
      jobs = load_manifest_jobs(manifest_path, out_dir);
    } else if (!wav_list_path.empty()) {
      if (!wav_path.empty()) {
        throw std::runtime_error(
            "use only one of --manifest, --wav-list, or --wav");
      }
      if (out_dir.empty()) {
        throw std::runtime_error("--wav-list requires --out-dir");
      }
      jobs = load_wav_list_jobs(wav_list_path, fs::path(out_dir));
    } else {
      const std::string out_path = get_arg(argc, argv, "--out");
      if (wav_path.empty()) {
        throw std::runtime_error("missing --wav (see --help)");
      }
      if (!out_dir.empty()) {
        throw std::runtime_error(
            "--out-dir is only for --manifest or --wav-list batch mode");
      }
      jobs.push_back({wav_path, out_path});
    }

    cppannote::CppAnnote engine =
        (seg_onnx.empty() && emb_onnx.empty())
            ? cppannote::CppAnnote()
            : cppannote::CppAnnote(seg_onnx, emb_onnx);

    run_diarize(engine, jobs, cluster_cadence, analyze_cadence,
                continue_on_error);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
