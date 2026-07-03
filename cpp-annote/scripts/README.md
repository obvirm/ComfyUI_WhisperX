# ONNX export scripts (C++ port)

These scripts export **`pyannote/speaker-diarization-community-1`** (or compatible checkpoints) to ONNX for the C++ diarization port described in [`../porting-plan.md`](../porting-plan.md).

## Prerequisites

- Install **pyannote-audio** from the repo root (editable is fine):

  ```bash
  pip install -e ".[export]"
  ```

- Accept the model license on Hugging Face and set a token:

  ```bash
  export HF_TOKEN=hf_...
  ```

## Usage

From the **repository root** (so `pyannote.audio` imports resolve):

```bash
./scripts/export.sh
```

That installs `[export]`, writes ONNX + JSON under `cpp/artifacts/`, then runs **`verify_onnx_parity.py`** (Torch vs ONNX Runtime on CPU).

To export only, or to re-check after changing artifacts:

```bash
python cpp/scripts/export_segmentation_onnx.py -o cpp/artifacts/community1-segmentation.onnx
python cpp/scripts/export_embedding_onnx.py -o cpp/artifacts/community1-embedding.onnx
python cpp/scripts/verify_onnx_parity.py
```

### Parity verification

`verify_onnx_parity.py` loads the same HF weights as export, runs **segmentation** (`SegmentationInferenceWrapper` vs ORT `waveforms`) and **embedding** (`EmbeddingFbankOnnxWrapper` vs ORT `fbank` + `weights`, with fbank from `compute_fbank`). Defaults: `cpp/artifacts/community1-*.onnx`. Tune tolerances if needed: `--rtol`, `--atol`. Skip one model with `--skip-segmentation` / `--skip-embedding`.

### Golden traces (full Python pipeline)

`dump_diarization_golden.py` mirrors `SpeakerDiarization.apply()` and writes NPZ + JSON per utterance for C++ regression tests (segmentations, binarized, counts, embeddings, clustering, discrete diarization, final JSON hypotheses).

```bash
./scripts/dump_golden.sh cpp/golden/my_run path/to/audio.wav
# or
python cpp/scripts/dump_diarization_golden.py -o cpp/golden/my_run path/to/a.wav path/to/b.wav
```

Optional: `--num-speakers`, `--min-speakers`, `--max-speakers`. Outputs `manifest.json`, `pipeline_snapshot.json`, and `receptive_field.json` at the bundle root.

Per utterance, **`embedding_chunk0_spk0_ort.npz`** holds `fbank`, `weights`, and `expected_embedding` for chunk 0 / local speaker 0 (used by **`embedding_golden_test`** in C++).

Each utterance folder also includes **`first_chunk_waveform.npz`** (first `Inference` window, for C++ segmentation tests). Bundles created **before** this field existed can be repaired with:

```bash
python cpp/scripts/add_first_chunk_waveform.py cpp/golden/<run>/<utterance_dir> path/to/same.wav
```

### CallHome: Python vs C++ DER on the first *n* clips

`eval_callhome_cpp_vs_python.py` downloads the first **n** rows from the English CallHome split (config `eng`, split `data` by default), truncates each to `--max-seconds`, runs `dump_diarization_golden.py` once for receptive field / pipeline snapshot / optional per-clip `golden_speaker_bounds.json`, runs `cpp-annote-cli` in batch (ORT segmentation + ORT embedding + VBx in C++), then reports **Diarization Error Rate** (`pyannote.metrics`) against the dataset’s timestamp/speaker reference for both the full Python pipeline and the C++ path.

```bash
# From repo root; needs HF_TOKEN, datasets, torchaudio, built cpp/build/cpp-annote-cli,
# cpp/artifacts/community1-segmentation.onnx and cpp/artifacts/community1-embedding.onnx (or pass --cpp-embedding-onnx).
uv run python cpp/scripts/eval_callhome_cpp_vs_python.py -n 20 --work-dir .callhome_eval_work
```

Use `--skip-cpp` to only refresh golden + Python DER, or `--skip-dump` if wavs and `golden/` under `--work-dir` already match the requested indices.

### Callhome example WAV (Hugging Face)

The dataset [`diarizers-community/callhome`](https://huggingface.co/datasets/diarizers-community/callhome) exposes split **`data`** (not `train` in the current revision). Fetch a short clip then dump:

```bash
pip install datasets   # if needed
python cpp/scripts/fetch_callhome_example_wav.py \
  --subset eng --split data --index 0 --max-seconds 120 \
  -o cpp/golden/examples/callhome_eng_data_idx0_head120s.wav
./scripts/dump_golden.sh cpp/golden/callhome_eng_idx0 cpp/golden/examples/callhome_eng_data_idx0_head120s.wav
```

See **`cpp/golden/examples/README.md`** for details.

Each export command writes:

- `*.onnx` — ONNX graph
- `*.json` — sample rates, chunk sizes, tensor names, and notes for C++ loaders

### Options

Both scripts accept:

- `--checkpoint` — HF repo id or local directory (default: `pyannote/speaker-diarization-community-1`)
- `--subfolder` — `segmentation` or `embedding` (defaults match community-1 layout)
- `--revision` — HF git revision
- `--token` — overrides `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
- `--opset` — ONNX opset (default: 17)

Segmentation only:

- `--skip-powerset-conversion` — export raw logits; only if you decode powerset in C++

## Relation to pyannote-onnx

The upstream [pyannote-onnx `export_onnx.py`](https://github.com/pengzhendong/pyannote-onnx/blob/master/export_onnx.py) exports `Model` with a dummy tensor. **`export_segmentation_onnx.py`** does the same for the **community-1** HF layout and, when the task uses **powerset**, wraps the graph with `SegmentationInferenceWrapper` so the ONNX output matches **`Inference.infer`** (hard multilabel), which is what the diarization pipeline consumes when `specifications.powerset` is true.

## Troubleshooting

- **`RepositoryNotFoundError` / 401** — Check `HF_TOKEN` and that you accepted the model conditions on Hugging Face.
- **`onnx` / `onnxruntime` import errors** — Run `pip install -e ".[export]"`.
- **Opset or operator warnings** — Try `--opset 18` or update ONNX; file an issue if export fails on a specific checkpoint revision.
- **`torch` / `torchvision` mismatch** (import errors inside `lightning` / `torchmetrics`) — Align versions (e.g. reinstall `torchvision` for your `torch` build) so `import pyannote.audio` works before running export.
