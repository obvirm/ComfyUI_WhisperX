#ifndef CPP_ANNOTE_C_H
#define CPP_ANNOTE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#    if defined(CPPANNOTE_BUILD_DLL)
#        define CPPANNOTE_API __declspec(dllexport)
#    else
#        define CPPANNOTE_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define CPPANNOTE_API __attribute__((visibility("default")))
#else
#    define CPPANNOTE_API
#endif

struct cpp_annote_context;

/**
 * Init dengan path ONNX models dari artifacts/.
 * seg_path — path ke community1-segmentation.onnx
 * emb_path — path ke community1-embedding.onnx
 * Returns NULL kalo gagal.
 */
CPPANNOTE_API struct cpp_annote_context* cpp_annote_init(
    const char* seg_path, const char* emb_path);

/** Free context. */
CPPANNOTE_API void cpp_annote_free(struct cpp_annote_context* ctx);

/**
 * Diarize audio mono 16kHz float32.
 *
 * audio      — buffer float32 [n_samples]
 * n_samples  — jumlah sample
 * sr         — sample rate (biasanya 16000)
 * out_json   — output: JSON string (caller must free with cpp_annote_free_string)
 *
 * Returns 0 sukses, -1 error.
 * JSON format:
 *   [{"start": 0.68, "end": 6.62, "speaker": 0}, ...]
 */
CPPANNOTE_API int cpp_annote_diarize(
    struct cpp_annote_context* ctx,
    const float* audio,
    int n_samples,
    int sr,
    char** out_json);

/**
 * VAD-only: return segments tanpa speaker ID.
 * output: JSON array [{"start": ..., "end": ...}, ...]
 */
CPPANNOTE_API int cpp_annote_vad(
    struct cpp_annote_context* ctx,
    const float* audio,
    int n_samples,
    int sr,
    char** out_json);

/** Free string returned by cpp_annote_diarize / cpp_annote_vad. */
CPPANNOTE_API void cpp_annote_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif /* CPP_ANNOTE_C_H */
