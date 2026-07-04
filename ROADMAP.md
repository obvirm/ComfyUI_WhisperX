# Roadmap

## v1.1.0 — Anti-Hallucination & Quality

- [ ] **Confidence Filter** — skip segmen dengan `no_speech_prob` tinggi / confidence rendah. Filter otomatis biar output gak berisi hallucinasi.
- [ ] **GBNF Grammar** — expose grammar params di UI. Output JSON, formatting tertentu, atau schema kustom. grammar sudah support di whisper.cpp, tinggal diwire.

## v1.2.0 — Flexibility

- [ ] **Model Hot-Swap** — ganti model (tiny, base, small, medium, large, large-v3-turbo) via dropdown tanpa restart ComfyUI.
- [ ] **Output Presets** — simpan konfigurasi node (language, VAD, alignment params, temperature, dll) sebagai preset yang bisa dipanggil lagi.

## v1.3.0 — Scale

- [ ] **Batch Processing** — multiple audio files masuk, diproses satu pipeline: UVR → VAD → ASR → Alignment → Diarization. Output kumulatif.
