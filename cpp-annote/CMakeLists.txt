# C++ bootstrap: ONNX Runtime + golden NPZ parity (segmentation first chunk).
cmake_minimum_required(VERSION 3.16)
project(pyannote_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Eigen (header-only, from third-party/) ---------------------------------
add_library(eigen_mpl2 INTERFACE)
target_compile_definitions(eigen_mpl2 INTERFACE EIGEN_MPL2_ONLY=1)
target_include_directories(eigen_mpl2 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third-party")

# ---- cnpy (NPZ / NPY, from third-party/) ------------------------------------
add_library(cnpy_static STATIC "${CMAKE_CURRENT_SOURCE_DIR}/third-party/cnpy/cnpy.cpp")
target_include_directories(cnpy_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third-party/cnpy")
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy_static PUBLIC ZLIB::ZLIB)

# ---- kissfft (from third-party/) ---------------------------------------------
set(KISSFFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/kissfft")
add_library(kissfft STATIC
  "${KISSFFT_DIR}/kiss_fft.c"
  "${KISSFFT_DIR}/kiss_fftr.c"
)
target_include_directories(kissfft PUBLIC "${KISSFFT_DIR}")
target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=float)

# ---- kaldi-native-fbank (from third-party/) ----------------------------------
set(KNF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/kaldi-native-fbank")
set(KNF_CSRC "${KNF_DIR}/csrc")
add_library(kaldi-native-fbank-core STATIC
  "${KNF_CSRC}/feature-fbank.cc"
  "${KNF_CSRC}/feature-functions.cc"
  "${KNF_CSRC}/feature-mfcc.cc"
  "${KNF_CSRC}/feature-raw-audio-samples.cc"
  "${KNF_CSRC}/feature-window.cc"
  "${KNF_CSRC}/istft.cc"
  "${KNF_CSRC}/kaldi-math.cc"
  "${KNF_CSRC}/mel-computations.cc"
  "${KNF_CSRC}/online-feature.cc"
  "${KNF_CSRC}/rfft.cc"
  "${KNF_CSRC}/stft.cc"
  "${KNF_CSRC}/whisper-feature.cc"
)
target_include_directories(kaldi-native-fbank-core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third-party")
target_link_libraries(kaldi-native-fbank-core PUBLIC kissfft)

# ---- ONNX Runtime (prebuilt SDK) -------------------------------------------
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "onnxruntime root" FORCE)
endif()
if(NOT ONNXRUNTIME_ROOT)
  message(
    FATAL_ERROR
      "Set ONNXRUNTIME_ROOT to a prebuilt ONNX Runtime tree "
      "(folders include/ and lib/ or lib64/), e.g. after unpacking "
      "onnxruntime-osx-arm64-*.tgz. See cpp/README.md."
  )
endif()

find_path(
  ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include/onnxruntime" "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH
)
find_library(
  ORT_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
  NO_DEFAULT_PATH
)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(
    FATAL_ERROR
      "Could not find onnxruntime under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
  )
endif()

# ---- segmentation golden test ---------------------------------------------
add_executable(segmentation_golden_test tests/segmentation_golden_test.cpp)
target_include_directories(segmentation_golden_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  segmentation_golden_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(APPLE)
  target_link_libraries(segmentation_golden_test PRIVATE "-framework Foundation")
endif()

add_executable(full_segmentation_window_parity_test tests/full_segmentation_window_parity_test.cpp)
target_include_directories(full_segmentation_window_parity_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(full_segmentation_window_parity_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  full_segmentation_window_parity_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(APPLE)
  target_link_libraries(full_segmentation_window_parity_test PRIVATE "-framework Foundation")
endif()

add_executable(
  embedding_golden_test
  tests/embedding_golden_test.cpp
  src/compute_fbank.cpp
  src/embedding_ort_infer.cpp
)
target_include_directories(embedding_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(embedding_golden_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  embedding_golden_test PRIVATE cnpy_static "${ORT_LIBRARY}" kaldi-native-fbank-core
)
if(APPLE)
  target_link_libraries(embedding_golden_test PRIVATE "-framework Foundation")
endif()

add_executable(speaker_count_golden_test tests/speaker_count_golden_test.cpp)
target_link_libraries(speaker_count_golden_test PRIVATE cnpy_static)

add_executable(frozen_golden_inputs_test tests/frozen_golden_inputs_test.cpp)
target_compile_definitions(
  frozen_golden_inputs_test
  PRIVATE "PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s\""
          "PYANNOTE_FROZEN_GOLDEN_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/golden/callhome_eng_idx0\""
)
target_link_libraries(frozen_golden_inputs_test PRIVATE cnpy_static)

add_executable(reconstruct_golden_test tests/reconstruct_golden_test.cpp)
target_link_libraries(reconstruct_golden_test PRIVATE cnpy_static)

add_executable(annotation_golden_test tests/annotation_golden_test.cpp)
target_include_directories(annotation_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(annotation_golden_test PRIVATE cnpy_static)

add_executable(
  clustering_golden_test
  tests/clustering_golden_test.cpp
  src/scipy_linkage.cpp
  src/filter_train.cpp
  src/plda_vbx.cpp
  src/clustering_vbx.cpp
  src/parity_log.cpp
)
target_include_directories(clustering_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(clustering_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(vbx_parity_test tests/vbx_parity_test.cpp src/plda_vbx.cpp)
target_include_directories(vbx_parity_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vbx_parity_test PRIVATE cnpy_static eigen_mpl2)

add_executable(plda_fea_golden_test tests/plda_fea_golden_test.cpp src/plda_vbx.cpp)
target_include_directories(plda_fea_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(plda_fea_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(
  linkage_fcluster_golden_test
  tests/linkage_fcluster_golden_test.cpp
  src/scipy_linkage.cpp
)
target_include_directories(linkage_fcluster_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(linkage_fcluster_golden_test PRIVATE cnpy_static)

add_executable(
  filter_ahc_golden_test
  tests/filter_ahc_golden_test.cpp
  src/filter_train.cpp
  src/scipy_linkage.cpp
)
target_include_directories(filter_ahc_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(filter_ahc_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(
  cpp-annote-cli
  tools/cpp-annote-cli.cpp
  src/cpp-annote.cpp
  src/cpp-annote-streaming.cpp
  src/community1_cpp_annote_embedded.cpp
  src/community1_ort_json_embedded.cpp
  src/compute_fbank.cpp
  src/embedding_ort_infer.cpp
  src/filter_train.cpp
  src/clustering_vbx.cpp
  src/plda_vbx.cpp
  src/scipy_linkage.cpp
  src/parity_log.cpp
)
target_include_directories(cpp-annote-cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(cpp-annote-cli PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(cpp-annote-cli PRIVATE cnpy_static "${ORT_LIBRARY}" eigen_mpl2 kaldi-native-fbank-core)
if(APPLE)
  target_link_libraries(cpp-annote-cli PRIVATE "-framework Foundation")
endif()

add_executable(knf_smoke_test tests/knf_smoke_test.cpp)
target_link_libraries(knf_smoke_test PRIVATE kaldi-native-fbank-core)

#================================================
# Shared Library (C API) — for ctypes binding
#================================================

option(CPPANNOTE_BUILD_SHARED "Build shared library (cpp-annote.dll/.so)" ON)

if(CPPANNOTE_BUILD_SHARED)
    # Same sources as CLI, but WITHOUT the embedded ORT models (52MB)
    # and WITHOUT the embedded CppAnnote config (2.5MB)
    # File-based ONNX models from artifacts/ are used instead.
    # Add C API wrapper instead.
    add_library(cpp_annote_shared SHARED
        src/cpp-annote.cpp
        src/cpp-annote-streaming.cpp
        src/community1_cpp_annote_embedded.cpp
        src/community1_ort_json_embedded.cpp
        src/compute_fbank.cpp
        src/embedding_ort_infer.cpp
        src/filter_train.cpp
        src/clustering_vbx.cpp
        src/plda_vbx.cpp
        src/scipy_linkage.cpp
        src/parity_log.cpp
        src/cpp-annote-c.cpp
    )
    
    target_include_directories(cpp_annote_shared PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_include_directories(cpp_annote_shared PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(cpp_annote_shared PRIVATE
        cnpy_static "${ORT_LIBRARY}" eigen_mpl2 kaldi-native-fbank-core
    )
    
    if(MSVC)
        target_compile_options(cpp_annote_shared PRIVATE /W3 /utf-8)
    else()
        target_compile_options(cpp_annote_shared PRIVATE -Wall -Wextra)
    endif()
    
    set_target_properties(cpp_annote_shared PROPERTIES
        OUTPUT_NAME "cpp_annote"
        MSVC_RUNTIME_DYNAMIC ON
    )
    
    target_compile_definitions(cpp_annote_shared PRIVATE CPPANNOTE_BUILD_DLL CPPANNOTE_NO_EMBEDDED)
    
    if(APPLE)
        target_link_libraries(cpp_annote_shared PRIVATE "-framework Foundation")
    endif()
endif()

message(STATUS "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
message(STATUS "ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}")
message(STATUS "ORT_LIBRARY=${ORT_LIBRARY}")
