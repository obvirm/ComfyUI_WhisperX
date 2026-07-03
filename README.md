# ComfyUI-WhisperCPP

ComfyUI custom node for speech-to-text using whisper.cpp with full CPP parameter support via ctypes binding.

## Features

- **whisper.cpp native** - Direct ctypes binding, no third-party Python wrappers
- **All whisper_full_params** - Every field exposed as node inputs
- **All GGML models** - tiny to large-v3-turbo
- **Optional alignment** - Word-level via wav2vec2 (whisperx)
- **Optional diarization** - Speaker diarization via pyannote (whisperx)
- **Multiple outputs** - SRT, VTT, TXT, TSV, JSON, AUD
- **GPU acceleration** - Vulkan, OpenCL, CUDA (all compiled into one binary)

## GPU Backends

| Backend | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Vulkan  | Yes     | Yes   | Yes   |
| OpenCL  | Yes     | Yes   | -     |
| CUDA    | Yes*    | Yes   | -     |
| Metal   | -       | -     | Yes   |
| HIP     | -       | Yes   | -     |

*CUDA toolset requires compatible CMake + VS integration.

Auto-detect at build time with `--gpu auto` or force enable with `--gpu all`.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone --recurse-submodules https://github.com/obvirm/ComfyUI-WhisperCPP
cd ComfyUI-WhisperCPP
pip install -r requirements.txt
python build_whisper_cpp.py
```

### Build Options

```bash
# Auto-detect GPU backends (default)
python build_whisper_cpp.py

# Force all available backends
python build_whisper_cpp.py --gpu all

# Specific backend
python build_whisper_cpp.py --gpu cuda
python build_whisper_cpp.py --gpu vulkan

# CPU only
python build_whisper_cpp.py --gpu none
```

## Usage

1. Add WhisperCPP Transcription node
2. Connect AUDIO to audio input
3. Select model (auto-downloaded on first use)
4. Set language (or None for auto-detect)
5. Toggle show_advance_settings to access all whisper.cpp params

## Architecture

```
├── whisper.cpp/          git submodule (ggml-org/whisper.cpp)
├── whispercpp/           Python package (ctypes binding)
│   ├── whisper_lib.py    ctypes binding to C API
│   ├── model.py          GGML model management
│   └── audio.py          Audio processing
├── whispercpp_node.py    ComfyUI node
├── build_whisper_cpp.py  Cross-platform build script
├── js/whispercpp_node.js Frontend collapse widget
└── whisper.dll/.so/.dylib Built library (gitignored)
```
