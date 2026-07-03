#!/usr/bin/env python3
"""Generate community1_ort_json_embedded.h/cpp from artifacts JSON files."""
import json, sys

seg_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/community1-segmentation.json"
emb_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/community1-embedding.json"
outdir = sys.argv[3] if len(sys.argv) > 3 else "src/"

seg = json.load(open(seg_path))
emb = json.load(open(emb_path))

# Header
with open(outdir + "community1_ort_json_embedded.h", "w") as f:
    f.write("""// Auto-generated from artifacts JSON
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

# CPP
seg_str = json.dumps(seg, indent=2)
emb_str = json.dumps(emb, indent=2)

with open(outdir + "community1_ort_json_embedded.cpp", "w") as f:
    f.write("""// Auto-generated from artifacts JSON

#include "community1_ort_json_embedded.h"

namespace cppannote::embedded_community1 {{

const char segmentation_json[] = R"pyannote_embed({seg})pyannote_embed";
const std::size_t segmentation_json_size = sizeof(segmentation_json) - 1;

const char embedding_json[] = R"pyannote_embed({emb})pyannote_embed";
const std::size_t embedding_json_size = sizeof(embedding_json) - 1;

}}  // namespace cppannote::embedded_community1
""".format(seg=seg_str, emb=emb_str))

print("Generated OK")
