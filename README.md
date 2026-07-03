# ComfyUI-WhisperCPP

🎤 **WhisperCPP** — ComfyUI custom node untuk speech-to-text menggunakan [whisper.cpp](https://github.com/ggml-org/whisper.cpp) dengan **full CPP parameter support**.

Binding ctypes langsung ke C API whisper.cpp — tanpa third-party Python wrappers.  
Cross-platform: Windows, Linux, macOS.

## Features
- **whisper.cpp Native** — Direct ctypes binding, full control
- **All CPP Parameters** — Setiap `whisper_full_params` field di-expose
- **All GGML Models** — tiny → large-v3-turbo
- **Optional Alignment** — Word-level via wav2vec2 (whisperx)
- **Optional Diarization** — Speaker diarization via pyannote (whisperx)
- **Multiple Outputs** — SRT, VTT, TXT, TSV, JSON, AUD
- **Cross-Platform** — Windows/Linux/macOS
- **Git Submodule** — whisper.cpp source included, build your own binary

## Installation
```bash
cd ComfyUI/custom_nodes/
git clone --recurse-submodules https://github.com/obvirm/ComfyUI-WhisperCPP
cd ComfyUI-WhisperCPP
pip install -r requirements.txt
python build_whisper_cpp.py
```

## Usage
1. Add **WhisperCPP Transcription** node
2. Connect AUDIO → `audio`
3. Select `model` (auto-downloaded)
4. Set `language` (or None for auto-detect)
5. Toggle `show_advance_settings` for ALL whisper.cpp params

## Architecture
```
├── whisper.cpp/          # Git submodule (ggml-org/whisper.cpp)
├── whispercpp/           # Python package (our ctypes binding)
│   ├── whisper_lib.py    # ctypes binding ke C API
│   ├── model.py          # GGML model management
│   └── audio.py          # Audio processing
├── whispercpp_node.py    # ComfyUI node
├── build_whisper_cpp.py  # Build script
└── js/whispercpp_node.js # Frontend UI
```
