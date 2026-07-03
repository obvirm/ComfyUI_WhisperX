// C API wrapper untuk cpp-annote — enables ctypes binding from Python.
// Pattern: sama seperti bs_roformer_c_api.cpp dan whisper.h.

#include "cpp-annote-c.h"
#include "cpp-annote.h"

#include <cstring>
#include <new>
#include <sstream>
#include <string>

struct cpp_annote_context {
    cppannote::CppAnnote* engine;
};

cpp_annote_context* cpp_annote_init(const char* seg_path, const char* emb_path) {
    if (!seg_path || !emb_path || !seg_path[0] || !emb_path[0]) return nullptr;

    auto ctx = new (std::nothrow) cpp_annote_context();
    if (!ctx) return nullptr;

    try {
        ctx->engine = new cppannote::CppAnnote(
            std::string(seg_path),
            std::string(emb_path));
    } catch (...) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void cpp_annote_free(cpp_annote_context* ctx) {
    if (!ctx) return;
    delete ctx->engine;
    delete ctx;
}

int cpp_annote_diarize(
    cpp_annote_context* ctx,
    const float* audio,
    int n_samples,
    int sr,
    char** out_json)
{
    if (!ctx || !ctx->engine || !audio || !out_json) return -1;
    if (n_samples <= 0) return -1;

    try {
        auto results = ctx->engine->diarize(audio, (uint64_t)n_samples, sr);

        // Serialize to JSON
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < results.turns.size(); i++) {
            if (i > 0) ss << ",";
            ss << "{"
               << "\"start\":" << results.turns[i].start << ","
               << "\"end\":" << results.turns[i].end << ","
               << "\"speaker\":" << results.turns[i].speaker
               << "}";
        }
        ss << "]";

        std::string json = ss.str();
        *out_json = (char*)std::malloc(json.size() + 1);
        if (!*out_json) return -1;
        std::memcpy(*out_json, json.c_str(), json.size() + 1);
        return 0;
    } catch (...) {
        return -1;
    }
}

int cpp_annote_vad(
    cpp_annote_context* ctx,
    const float* audio,
    int n_samples,
    int sr,
    char** out_json)
{
    if (!ctx || !ctx->engine || !audio || !out_json) return -1;
    if (n_samples <= 0) return -1;

    try {
        auto results = ctx->engine->diarize(audio, (uint64_t)n_samples, sr);

        // VAD = merge all turns into speech segments, no speaker info
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < results.turns.size(); i++) {
            if (i > 0) ss << ",";
            ss << "{"
               << "\"start\":" << results.turns[i].start << ","
               << "\"end\":" << results.turns[i].end
               << "}";
        }
        ss << "]";

        std::string json = ss.str();
        *out_json = (char*)std::malloc(json.size() + 1);
        if (!*out_json) return -1;
        std::memcpy(*out_json, json.c_str(), json.size() + 1);
        return 0;
    } catch (...) {
        return -1;
    }
}

void cpp_annote_free_string(char* str) {
    std::free(str);
}
