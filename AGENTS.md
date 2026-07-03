# AGENTS.md — ComfyUI-WhisperCPP

## Tujuan
Custom node ComfyUI untuk transkripsi audio pakai whisper.cpp (C API via ctypes), tanpa whisperx.  
Full pipeline: ASR (whisper.cpp native) → Alignment opsional (torchaudio wav2vec2) → Diarization opsional (pyannote.audio).

## Aturan Main

### 1. GAK BOLEH NGE-GAS TANPA VERIFIKASI
- Setiap perubahan: **test dulu**. Jangan ngarep "mungkin work".
- End-to-end: bikin workflow ComfyUI beneran, inject prompt, verifikasi output.
- Jangan pernah nebak properti API tanpa riset dulu.

### 2. Struktur Folder
```
ComfyUI-WhisperCPP/
├── __init__.py              # register node, NODE_CLASS_MAPPINGS
├── whispercpp_node.py       # class WhisperCPPNode (ComfyUI node)
├── whispercpp/
│   ├── __init__.py
│   ├── whisper_lib.py       # ctypes binding ke whisper.cpp (C struct, load, transcribe)
│   ├── model.py             # download manager (GGML_MODEL_KEYS, progress bar)
│   └── audio.py             # audio loader / preprocessor
│   └── ext/
│       ├── __init__.py
│       ├── alignment.py     # forced alignment via torchaudio wav2vec2 (opsional)
│       └── diarization.py   # speaker diarization via pyannote (opsional)
├── js/
│   └── whispercpp_node.js   # frontend toggle (show_advance_cpp, show_advance_ext)
├── whispercpp.json          # model config
└── MEMD.md                  # project memory
```

### 3. Output WAJIB lewat Node Connection
- `filename_prefix` dan `output_format` **DILARANG** — output cuma lewat 7 socket:
  `text`, `segments_json`, `srt`, `vtt`, `tsv`, `aud`, `json_result`
- Semua format generate di memory (`_make_outputs`), gak ada file I/O.

### 4. JS Toggle (Frontend)
- **Dua toggle independent**: `show_advance_cpp` dan `show_advance_ext`.
- Approach: grab widget reference di `onNodeCreated`, simpen di closure.
- Show/Hide: **JANGAN pakai `widget.hidden`** (belum tentu real).
- **JANGAN main-main `this.widgets` dengan cara yang bikin reference ilang.**
- Selalu `setTimeout` biar widget kebentuk dulu.

### 5. VAD Parameter Safety
- Kalo `vad=True` tapi `vad_model_path` gak diisi → **disable VAD otomatis**, log warning.
- Jangan pernah biarin `vad_model_path = NULL` pas `vad=True` — itu crash.

### 6. DTW (Context Params)
- `dtw_token_timestamps`, `dtw_aheads_preset`, `dtw_n_top` → **context params** (set pas `load_model`), **bukan** full params.
- Jangan masukin ke `field_map` di `_build_full_params` — mereka ada di `WhisperContextParams`, bukan `WhisperFullParams`.

### 7. Struct Alignment
- Setiap Python struct (`WhisperContextParams`, `WhisperFullParams`) → **cocokin offset sama C**.
- `WhisperAheads`: nested struct `{n_heads, heads}`, bukan dua field pisah.
- Always verify: `ctypes.sizeof(Struct) == sizeof(C_struct)`.

### 8. Grammar
- `grammar_penalty` default 0.0 (override C default 100.0). Harmless kalo `grammar_rules = nullptr`, tapi tetep waspada.

### 9. Extension Import
- `whispercpp/ext/alignment.py` dan `whispercpp/ext/diarization.py`: import via `from .whispercpp.ext.alignment import ...`.

### 10. Testing
- Test dengan cara ComfyUI load: `importlib.util.spec_from_file_location(...)`.
- Test transcribe beneran: sine wave → verify segments > 0.
- Test alignment: wav2vec2 → verify timestamp.
- Test VAD: pastiin gak crash kalo model path kosong.

### 11. MEMD
- Setiap keputusan penting → catet di MEMD.md.
- Jangan sampe ada "lupa" masalah yang udah kelar.

### 12. Git
- Commit message: jelas, pake prefix (`fix:`, `perf:`, `refactor:`, dll).
- `git add -A` baru `git commit`.
- Jangan nge-edit file JS pake `edit` tool yang bisa korup array — kalo ragu, `write` ulang aja.
