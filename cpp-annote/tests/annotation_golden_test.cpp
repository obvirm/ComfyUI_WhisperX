// SPDX-License-Identifier: MIT
// C++ parity for to_annotation (Binarize onset/offset 0.5 + min_duration_*) vs
// golden JSON.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "annotation_support.h"
#include "cnpy.h"

struct CanonicalTurn {
  double start = 0.;
  double end = 0.;
  std::string speaker;

  bool operator<(const CanonicalTurn& o) const {
    if (start != o.start) {
      return start < o.start;
    }
    if (end != o.end) {
      return end < o.end;
    }
    return speaker < o.speaker;
  }
};

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static bool try_regex_double(const std::string& json,
                             const std::string& key_escaped, double& out) {
  const std::string pat = "\"" + key_escaped + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = std::stod(m[1].str());
  return true;
}

// Binarize (signal.py) for 2D SlidingWindowFeature: hysteresis on frame
// middles.
static void binarize_column(
    const float* k_scores, int num_frames, double sw_start, double sw_duration,
    double sw_step, double onset, double offset, double pad_onset,
    double pad_offset, std::vector<std::pair<double, double>>& regions_out) {
  if (num_frames <= 0) {
    return;
  }
  std::vector<double> ts(static_cast<size_t>(num_frames));
  for (int i = 0; i < num_frames; ++i) {
    ts[static_cast<size_t>(i)] =
        sw_start + static_cast<double>(i) * sw_step + 0.5 * sw_duration;
  }

  double start = ts[0];
  bool is_active = static_cast<double>(k_scores[0]) > onset;

  for (int i = 1; i < num_frames; ++i) {
    const double t = ts[static_cast<size_t>(i)];
    const double y = static_cast<double>(k_scores[i]);
    if (is_active) {
      if (y < offset) {
        regions_out.emplace_back(start - pad_onset, t + pad_offset);
        start = t;
        is_active = false;
      }
    } else {
      if (y > onset) {
        start = t;
        is_active = true;
      }
    }
  }
  if (is_active) {
    const double t_end = ts[static_cast<size_t>(num_frames - 1)];
    regions_out.emplace_back(start - pad_onset, t_end + pad_offset);
  }
}

static std::map<int, std::string> parse_label_mapping(const std::string& json) {
  std::map<int, std::string> m;
  std::regex re("\"([0-9]+)\"\\s*:\\s*\"([^\"]*)\"");
  for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end;
       ++it) {
    const int k = std::stoi((*it)[1].str());
    m[k] = (*it)[2].str();
  }
  if (m.empty()) {
    throw std::runtime_error("label_mapping.json: no entries parsed");
  }
  return m;
}

static std::vector<CanonicalTurn> parse_diarization_json(
    const std::string& json) {
  std::vector<CanonicalTurn> rows;
  std::regex block(
      "\\{\\s*\"start\"\\s*:\\s*([-+0-9.eE]+)\\s*,\\s*\"end\"\\s*:\\s*([-+0-9."
      "eE]+)\\s*,"
      "\\s*\"speaker\"\\s*:\\s*\"([^\"]*)\"\\s*\\}");
  for (std::sregex_iterator it(json.begin(), json.end(), block), end; it != end;
       ++it) {
    CanonicalTurn r;
    r.start = std::stod((*it)[1].str());
    r.end = std::stod((*it)[2].str());
    r.speaker = (*it)[3].str();
    rows.push_back(std::move(r));
  }
  if (rows.empty() && json.find('[') != std::string::npos &&
      json.find(']') != std::string::npos) {
    // Fallback: flexible key order within each object
    std::regex re_start("\"start\"\\s*:\\s*([-+0-9.eE]+)");
    std::regex re_end("\"end\"\\s*:\\s*([-+0-9.eE]+)");
    std::regex re_sp("\"speaker\"\\s*:\\s*\"([^\"]*)\"");
    size_t pos = 0;
    while (pos < json.size()) {
      const size_t b = json.find('{', pos);
      if (b == std::string::npos) {
        break;
      }
      const size_t e = json.find('}', b);
      if (e == std::string::npos) {
        break;
      }
      const std::string chunk = json.substr(b, e - b + 1);
      std::smatch ms, me, mp;
      if (std::regex_search(chunk, ms, re_start) &&
          std::regex_search(chunk, me, re_end) &&
          std::regex_search(chunk, mp, re_sp)) {
        CanonicalTurn r;
        r.start = std::stod(ms[1].str());
        r.end = std::stod(me[1].str());
        r.speaker = mp[1].str();
        rows.push_back(std::move(r));
      }
      pos = e + 1;
    }
  }
  return rows;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: annotation_golden_test <golden_utterance_dir>\n";
    std::cerr << "  Needs discrete_diarization_overlap.npz, "
                 "label_mapping.json, diarization.json,\n";
    std::cerr << "  discrete_diarization_exclusive.npz, "
                 "exclusive_diarization.json,\n";
    std::cerr << "  and parent ../pipeline_snapshot.json (optional; defaults "
                 "min_duration_* to 0).\n";
    return 2;
  }
  const std::string utter = argv[1];
  try {
    const std::string snap_path = utter + "/../pipeline_snapshot.json";
    double min_off = 0.;
    double min_on = 0.;
    std::ifstream snap_probe(snap_path);
    if (snap_probe) {
      snap_probe.close();
      const std::string sj = read_text_file(snap_path);
      (void)try_regex_double(sj, "segmentation\\.min_duration_off", min_off);
      (void)try_regex_double(sj, "segmentation\\.min_duration_on", min_on);
    }

    const double onset = 0.5;
    const double offset = 0.5;
    const double pad_onset = 0.0;
    const double pad_offset = 0.0;

    const std::map<int, std::string> labels =
        parse_label_mapping(read_text_file(utter + "/label_mapping.json"));

    // Binarize.__call__: support(collar=min_duration_off) only if pad_* or
    // min_duration_off > 0.
    const bool apply_annotation_support =
        (pad_onset > 0.0 || pad_offset > 0.0 || min_off > 0.0);

    auto run_pair = [&](const char* discrete_npz, const char* golden_json) {
      cnpy::npz_t npz = cnpy::npz_load(utter + "/" + discrete_npz);
      if (!npz.count("data")) {
        throw std::runtime_error(std::string(discrete_npz) + " missing data");
      }
      const cnpy::NpyArray& arr = npz["data"];
      if (arr.shape.size() != 2) {
        throw std::runtime_error(
            "discrete diarization must be rank-2 (frames, speakers)");
      }
      const int num_frames = static_cast<int>(arr.shape[0]);
      const int num_classes = static_cast<int>(arr.shape[1]);
      const double sw_start = npz["sliding_window_start"].data<double>()[0];
      const double sw_dur = npz["sliding_window_duration"].data<double>()[0];
      const double sw_step = npz["sliding_window_step"].data<double>()[0];

      const float* src = arr.data<float>();

      std::map<int, std::vector<cppannote::Segment>> by_label;
      for (int k = 0; k < num_classes; ++k) {
        std::vector<float> col(static_cast<size_t>(num_frames));
        for (int t = 0; t < num_frames; ++t) {
          col[static_cast<size_t>(t)] =
              src[static_cast<size_t>(t) * static_cast<size_t>(num_classes) +
                  static_cast<size_t>(k)];
        }
        std::vector<std::pair<double, double>> regs;
        binarize_column(col.data(), num_frames, sw_start, sw_dur, sw_step,
                        onset, offset, pad_onset, pad_offset, regs);
        for (const auto& pr : regs) {
          by_label[k].push_back(cppannote::Segment{pr.first, pr.second});
        }
      }

      std::vector<std::tuple<int, double, double>> turns;
      if (apply_annotation_support) {
        const std::vector<std::pair<int, cppannote::Segment>> merged =
            cppannote::annotation_support(by_label, min_off);
        for (const auto& item : merged) {
          const cppannote::Segment& seg = item.second;
          if (seg.duration() >= min_on - 1e-12) {
            turns.emplace_back(item.first, seg.start, seg.end);
          }
        }
      } else {
        for (int k = 0; k < num_classes; ++k) {
          for (const cppannote::Segment& seg : by_label[static_cast<int>(k)]) {
            if (seg.duration() >= min_on - 1e-12) {
              turns.emplace_back(k, seg.start, seg.end);
            }
          }
        }
      }

      std::vector<CanonicalTurn> got;
      for (const auto& tr : turns) {
        const int lab = std::get<0>(tr);
        auto it = labels.find(lab);
        if (it == labels.end()) {
          throw std::runtime_error(
              "label_mapping missing key for speaker index " +
              std::to_string(lab));
        }
        got.push_back({std::get<1>(tr), std::get<2>(tr), it->second});
      }

      std::sort(got.begin(), got.end());
      std::vector<CanonicalTurn> gold =
          parse_diarization_json(read_text_file(utter + "/" + golden_json));
      std::sort(gold.begin(), gold.end());

      if (got.size() != gold.size()) {
        std::ostringstream oss;
        oss << golden_json << ": turn count mismatch got " << got.size()
            << " gold " << gold.size();
        throw std::runtime_error(oss.str());
      }
      const double tol = 1e-5;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i].speaker != gold[i].speaker) {
          std::ostringstream oss;
          oss << golden_json << " idx " << i << " speaker got \""
              << got[i].speaker << "\" gold \"" << gold[i].speaker << "\"";
          throw std::runtime_error(oss.str());
        }
        if (std::abs(got[i].start - gold[i].start) > tol ||
            std::abs(got[i].end - gold[i].end) > tol) {
          std::ostringstream oss;
          oss << golden_json << " idx " << i << " bounds got [" << got[i].start
              << "," << got[i].end << "] gold [" << gold[i].start << ","
              << gold[i].end << "]";
          throw std::runtime_error(oss.str());
        }
      }
      std::cout << "PASS " << golden_json << " (" << got.size() << " turns)\n";
    };

    run_pair("discrete_diarization_overlap.npz", "diarization.json");
    run_pair("discrete_diarization_exclusive.npz",
             "exclusive_diarization.json");
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
