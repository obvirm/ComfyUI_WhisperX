// Auto-generated from artifacts JSON
#include "community1_ort_json_embedded.h"
namespace cppannote::embedded_community1 {

const char segmentation_json[] = R"pyannote_embed({
  "model_type": "segmentation",
  "checkpoint": "pyannote/speaker-diarization-community-1",
  "subfolder": "segmentation",
  "revision": null,
  "sample_rate": 16000,
  "num_channels": 1,
  "chunk_duration_sec": 10.0,
  "chunk_step_sec": 1.0,
  "chunk_num_samples": 160000,
  "powerset": true,
  "export_includes_powerset_to_multilabel": true,
  "input_names": [
    "waveforms"
  ],
  "output_names": [
    "segmentation"
  ],
  "opset": 17,
  "notes": "Input layout (batch, channel, samples) matches Inference.infer. Output matches post-conversion multilabel when powerset and skip_powerset_conversion=False."
})pyannote_embed";
const std::size_t segmentation_json_size = sizeof(segmentation_json) - 1;

const char embedding_json[] = R"pyannote_embed({
  "model_type": "embedding",
  "checkpoint": "pyannote/speaker-diarization-community-1",
  "subfolder": "embedding",
  "revision": null,
  "sample_rate": 16000,
  "num_channels": 1,
  "num_mel_bins": 80,
  "frame_length_ms": 25.0,
  "frame_shift_ms": 10.0,
  "fbank_centering_span": null,
  "embedding_dim": 256,
  "example_num_samples_for_fbank_ref": 80000,
  "example_num_frames": 498,
  "input_names": [
    "fbank",
    "weights"
  ],
  "output_names": [
    "embedding"
  ],
  "opset": 17,
  "notes": "ONNX inputs are Kaldi log-fbank features matching pyannote ``BaseWeSpeakerResNet.compute_fbank`` output: shape (batch, num_frames, num_mel_bins). Compute fbank outside ONNX (torchaudio kaldi.fbank + scaling/mean per checkpoint). weights (batch, num_frames): 1.0 keep frame, 0.0 drop; must match num_frames for fbank."
})pyannote_embed";
const std::size_t embedding_json_size = sizeof(embedding_json) - 1;

}  // namespace cppannote::embedded_community1
