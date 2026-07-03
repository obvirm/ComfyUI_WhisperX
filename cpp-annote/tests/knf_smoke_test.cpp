// SPDX-License-Identifier: MIT
// Minimal link/run check: kaldi-native-fbank + kissfft produce at least one
// fbank frame.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

int main() {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.dither = 0.0f;
  knf::OnlineFbank fbank(opts);

  std::vector<float> wav(
      static_cast<std::size_t>(opts.frame_opts.samp_freq / 10), 0.f);
  fbank.AcceptWaveform(opts.frame_opts.samp_freq, wav.data(),
                       static_cast<int32_t>(wav.size()));
  fbank.InputFinished();

  if (fbank.NumFramesReady() < 1) {
    std::cerr << "FAIL: expected at least one fbank frame\n";
    return 1;
  }
  if (fbank.Dim() < 1) {
    std::cerr << "FAIL: expected positive fbank dimension\n";
    return 1;
  }
  const float* frame0 = fbank.GetFrame(0);
  if (frame0 == nullptr) {
    std::cerr << "FAIL: null frame pointer\n";
    return 1;
  }

  std::cout << "knf_smoke_test OK frames=" << fbank.NumFramesReady()
            << " dim=" << fbank.Dim() << "\n";
  return 0;
}
