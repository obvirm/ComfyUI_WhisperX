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
│       ├── uvr.py          # vocal separation via audio-separator (UVR MDX-Net)
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
- Test UVR: create Separator, run separate, verify vocal output.
- Test alignment: wav2vec2 → verify timestamp.
- Test VAD: pastiin gak crash kalo model path kosong.

### 11. IMPLEMENTASI END-TO-END (JANGAN SETENGAH-SETENGAH)
- **JANGAN pernah nambah parameter/toggle tanpa implementasi processing-nya.**
  Contoh: `separate_vocals` → harus ada `uvr.py` + hook di `transcribe()`.
- Kalo nambah fitur: selesaiin dari ujung ke ujung (Python + JS + test).
- Kalo gak bisa selesaiin sekarang: jangan nambah parameternya dulu.

### 12. MEMD
- Setiap keputusan penting → catet di MEMD.md.
- Jangan sampe ada "lupa" masalah yang udah kelar.

### 12. Git
- Commit message: jelas, pake prefix (`fix:`, `perf:`, `refactor:`, dll).
- `git add -A` baru `git commit`.
- Jangan nge-edit file JS pake `edit` tool yang bisa korup array — kalo ragu, `write` ulang aja.

### 13. Semua C++ Module WAJIB DLL + ctypes
- **TIDAK BOLEH CLI subprocess** — rapuh, beda OS beda masalah.
- Semua C++ module: bikin C API header → DLL/so/dylib → ctypes binding.
- Pattern: `bs_roformer.h` → `bs_roformer.dll` → `bs_roformer_lib.py` (ctypes).
- Cross-platform: beda platform tinggal ganti `.dll`/`.so`/`.dylib`, binding Python-nya SAMA.

### 14. Vendor Prefix
- `BSR_API` untuk dllexport/dllimport (MSVC) + visibility (GCC/Clang).
- Jangan lupa `target_compile_definitions(... PRIVATE BSR_BUILD_DLL)` di CMake.
- Function naming: `bs_roformer_*` (lowercase, underscore, prefix modul).

### 15. ctypes Binding Pattern
```python
# 1. Load DLL
lib = ctypes.CDLL(path)
# 2. Setup argtypes + restype
lib.function_name.argtypes = [ctypes.c_void_p, ...]
lib.function_name.restype = ctypes.c_int
# 3. Wrapper class dengan __del__ buat cleanup otomatis
```

### 16. DILARANG AMBIL JALAN PINTAS
- **Jangan pernah milih jalan termudah** — pilih jalan yang bener.
- CLI subprocess itu jalan pintas. `onnxruntime Python langsung` tanpa nanya juga jalan pintas.
- Cross-platform: kalo build/implementasi cuma beres di Windows, itu BELUM SELESAI.
- Sebelum ngoding: baca pola yang udah jalan di proyek ini (DLL + ctypes).
- Kalo ragu: **TANYA DULU**. Jangan asumsi.
- Tujuan: konsisten, bukan cepet selesai.
- Jangan pikir "ini cukup buat user senang" — pikir "ini bener apa enggak".

### 17. AUDIT METHODOLOGY

#### Static Analysis (Code Review)
```bash
# Cari patterns berbahaya
grep -rn "eval(\|exec(" --include="*.py"
grep -rn "password\|secret\|token" --include="*.py"
grep -rn "os.system\|subprocess" --include="*.py"
```

#### Runtime Testing (Actual Run)
```python
# Import dan test
from whispercpp.whisper_lib import WhisperCPP
from whispercpp.auto_download import _get_version
from whispercpp.gpu_detect import detect_gpu

print(f'Version: {_get_version()}')
print(f'GPU: {detect_gpu()}')
```

#### Memory Analysis
```python
import tracemalloc
tracemalloc.start()

# Run code
from whispercpp.whisper_lib import WhisperCPP
wcpp = WhisperCPP()

# Check memory
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:5]:
    print(stat)
```

#### Thread Safety Test
```python
import threading
from whispercpp.whisper_lib import WhisperCPP

def test_concurrent():
    wcpp = WhisperCPP()
    threads = []
    for _ in range(10):
        t = threading.Thread(target=lambda: wcpp._lock.acquire())
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
```

#### Code Smell Detection
```bash
# Cari code smells
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.py"
grep -rn "bare except\|except:" --include="*.py"
```

#### Audit Checklist
| Category | Method | Tools |
|---|---|---|
| Security | Static + Runtime | grep, Python |
| Memory | Runtime | tracemalloc |
| Thread | Runtime | threading |
| Performance | Static + Runtime | cProfile |
| Code Quality | Static | flake8, black |
