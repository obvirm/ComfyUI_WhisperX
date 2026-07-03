# Golden dump examples

## Callhome (`diarizers-community/callhome`)

The current Hugging Face revision uses a single split named **`data`** (there is no separate **`train`** split in the builder). The **English** subset is a practical default.

**Suggested row:** config **`eng`**, split **`data`**, index **`0`** — first stored conversation (mono, 16 kHz in the copy we tested).

### 1. Save a short WAV locally

From the repository root (requires `datasets`, `torchaudio`; use the same env as pyannote-audio):

```bash
python cpp/scripts/fetch_callhome_example_wav.py \
  --subset eng --split data --index 0 --max-seconds 120 \
  -o cpp/golden/examples/callhome_eng_data_idx0_head120s.wav
```

### 2. Run the golden dump

```bash
export HF_TOKEN=hf_...   # pyannote community-1 is gated
./scripts/dump_golden.sh cpp/golden/callhome_eng_idx0 cpp/golden/examples/callhome_eng_data_idx0_head120s.wav
```

Dataset card: [diarizers-community/callhome](https://huggingface.co/datasets/diarizers-community/callhome).
