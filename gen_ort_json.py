import json

seg_path = "E:/Ai/ComfyUI/ComfyUI/custom_nodes/ComfyUI-WhisperXX/cpp-annote/artifacts/community1-segmentation.json"
emb_path = "E:/Ai/ComfyUI/ComfyUI/custom_nodes/ComfyUI-WhisperXX/cpp-annote/artifacts/community1-embedding.json"
outdir = "E:/Ai/ComfyUI/ComfyUI/custom_nodes/ComfyUI-WhisperXX/cpp-annote/src/"

seg = json.load(open(seg_path))
emb = json.load(open(emb_path))

seg_str = json.dumps(seg, indent=2)
emb_str = json.dumps(emb, indent=2)

with open(outdir + "community1_ort_json_embedded.h", "w") as f:
    f.write("""\
// Auto-generated from artifacts JSON
#ifndef COMMUNITY1_ORT_JSON_EMBEDDED_H_
#define COMMUNITY1_ORT_JSON_EMBEDDED_H_

#include <cstddef>

namespace cppannote::embedded_community1 {

extern const char segmentation_json[];
extern const std::size_t segmentation_json_size;
extern const char embedding_json[];
extern const std::size_t embedding_json_size;

}  // namespace cppannote::embedded_community1
#endif
""")

with open(outdir + "community1_ort_json_embedded.cpp", "w") as f:
    f.write("""\
// Auto-generated from artifacts JSON
#include "community1_ort_json_embedded.h"
namespace cppannote::embedded_community1 {

const char segmentation_json[] = R"""!!!PYANNOTE_EMBED!!!(""" + seg_str + """)!!!PYANNOTE_EMBED!!!""";
const std::size_t segmentation_json_size = sizeof(segmentation_json) - 1;

const char embedding_json[] = R"""!!!PYANNOTE_EMBED!!!(""" + emb_str + """)!!!PYANNOTE_EMBED!!!""";
const std::size_t embedding_json_size = sizeof(embedding_json) - 1;

}  // namespace cppannote::embedded_community1
""")

print("Generated OK")
print("seg:", len(seg_str), "bytes")
print("emb:", len(emb_str), "bytes")
