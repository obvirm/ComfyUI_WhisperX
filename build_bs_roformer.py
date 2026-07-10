#!/usr/bin/env python3
"""Build BSRoformer.cpp sebagai shared library (DLL/so/dylib).
Cross-platform: Windows (MSVC), Linux (GCC/Clang), macOS (Clang).
"""

import argparse, os, platform, shutil, subprocess, sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
BSR_DIR  = NODE_DIR / "bs_roformer.cpp"
BUILD_DIR = BSR_DIR / "build"
IS_WIN   = platform.system() == "Windows"
IS_MAC   = platform.system() == "Darwin"
GGML_DIR = NODE_DIR / "whisper.cpp" / "ggml"

if IS_WIN: LIB_EXT = "dll"
elif IS_MAC: LIB_EXT = "dylib"
else: LIB_EXT = "so"
LIB_NAME = f"bs_roformer.{LIB_EXT}"

def _check_vulkan() -> bool:
    """Check if Vulkan is actually available (glslc installed)."""
    return shutil.which("glslc") is not None


def find_lib():
    """Cari hasil build library."""
    candidates = [
        BUILD_DIR / LIB_NAME,
        BUILD_DIR / "bin" / LIB_NAME,
        BUILD_DIR / "src" / LIB_NAME,
        BUILD_DIR / "lib" / LIB_NAME,
    ]
    if IS_WIN:
        for cfg in ["Release", "Debug", ""]:
            candidates.append(BUILD_DIR / cfg / LIB_NAME)
            candidates.append(BUILD_DIR / "bin" / cfg / LIB_NAME)
            candidates.append(BUILD_DIR / "src" / cfg / LIB_NAME)
    if not IS_WIN:
        import glob
        for p in ["build/*.so*", "build/src/*.so*", "build/lib/*.so*",
                   "build/*.dylib", "build/src/*.dylib", "build/lib/*.dylib"]:
            for f in glob.glob(str(NODE_DIR / "bs_roformer.cpp" / p)):
                candidates.append(Path(f))
    return next((p for p in candidates if p.is_file()), None)


def main():
    parser = argparse.ArgumentParser(description="Build BSRoformer shared library")
    parser.add_argument("--clean", action="store_true", help="Clean build dir")
    parser.add_argument("--build-type", choices=["Release","Debug"], default="Release")
    parser.add_argument("--no-copy", action="store_true", help="Jangan copy ke root")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend")
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan backend")
    parser.add_argument("--opencl", action="store_true", help="Enable OpenCL backend")
    parser.add_argument("--gpu", action="store_true", help="Enable all available GPU backends (DEPRECATED — now default)")
    parser.add_argument("--cpu-only", action="store_true", help="Build CPU-only (skip GPU backends)")
    args = parser.parse_args()
    # FULL backend: --gpu sekarang default ON kecuali --cpu-only
    if not args.cpu_only:
        args.gpu = True

    if not BSR_DIR.is_dir() or not (BSR_DIR / "CMakeLists.txt").is_file():
        print("ERROR: bs_roformer.cpp submodule not found.")
        sys.exit(1)

    if args.clean and BUILD_DIR.is_dir():
        shutil.rmtree(BUILD_DIR)
        print("Cleaned build dir")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Generator — force VS2022 (v143) di Windows agar cocok dengan CUDA 13.x
    if IS_WIN:
        generator = "Visual Studio 17 2022"
    elif IS_MAC:
        generator = "Unix Makefiles"
    else:
        generator = "Unix Makefiles"

    # GPU backends — FULL (semua diwajibkan kecuali --cpu-only ATAU platform gak support)
    # macOS gak support CUDA/OpenCL/Vulkan di ggml -> paksa OFF (Metal sudah di-whisper).
    cuda_on = "ON" if ((args.cuda or args.gpu) and not IS_MAC) else "OFF"
    # Vulkan butuh glslc; kalau gak ada CMake akan warning tapi build tetap jalan.
    # macOS gak ada Vulkan/OpenCL -> OFF.
    # Vulkan butuh glslc (Vulkan SDK). Kalau gak ada (CI Linux tanpa Vulkan SDK),
    # paksa OFF supaya configure gak gagal (ggml-vulkan FindVulkan butuh glslc).
    import shutil as _shutil
    _has_glslc = _shutil.which("glslc") is not None
    vulkan_on = "ON" if ((args.vulkan or args.gpu) and not IS_MAC and _has_glslc) else "OFF"
    opencl_on = "ON" if (args.opencl or args.gpu) and not IS_MAC else "OFF"

    # Auto-detect GPU if --gpu (jangan matiin CUDA kalau nvidia-smi gak ada di CI;
    # CUDA tetap di-build, device gak ketemu -> whisper/bsr fallback ke CPU di runtime)
    if args.gpu and not IS_MAC:
        # cuda_on tetap ON (FULL). Vulkan/OpenCL otomatis terdeteksi ggml saat configure.
        if IS_WIN:
            # Force VS2022 toolset untuk CUDA
            cmd_toolset = ["-T", "v143"]
        else:
            cmd_toolset = []
    else:
        cmd_toolset = []

    # Configure
    print(f"Configuring ({args.build_type}) CUDA={cuda_on} Vulkan={vulkan_on} OpenCL={opencl_on}...")
    cmd = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(BSR_DIR),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-G", generator,
        f"-DGGML_CUDA={cuda_on}",
        f"-DGGML_VULKAN={vulkan_on}",
        f"-DGGML_OPENCL={opencl_on}",
        f"-DBSR_BUILD_CLI=OFF",
        f"-DBSR_BUILD_SHARED=ON",
        f"-DBSR_BUILD_TESTS=OFF",
        # Backends di-build sbg MODULE terpisah & di-dlopen MALAS oleh ggml
        # (GGML_BACKEND_DL=ON). libbs_roformer.so TIDAK NEEDED backend .so, jadi di
        # host tanpa GPU backend CUDA gagal load silently & CPU dipakai. Di host GPU
        # backend aktif otomatis. (Sama dgn libwhisper — robust di CI tanpa GPU.)
        f"-DGGML_BACKEND_DL=ON",
        f"-DGGML_DIR={GGML_DIR}",
    ]
    # Inject toolset (-T v143) untuk Windows CUDA build
    if IS_WIN and args.gpu and cmd_toolset:
        cmd.extend(cmd_toolset)
    print(f"  {' '.join(str(c) for c in cmd)}")
    if subprocess.run(cmd).returncode != 0:
        print("CONFIGURATION FAILED")
        sys.exit(1)

    # Build
    print(f"Building ({args.build_type})...")
    build_cmd = ["cmake", "--build", str(BUILD_DIR), "--config", args.build_type,
                 "--target", "bs_roformer_shared"]
    if IS_WIN:
        build_cmd.append("--")
        build_cmd.append("/m")
    else:
        build_cmd.extend(["-j", str(os.cpu_count() or 4)])
    if subprocess.run(build_cmd).returncode != 0:
        print("BUILD FAILED")
        sys.exit(1)

    # Copy
    src = find_lib()
    if not src:
        print(f"Build done but {LIB_NAME} not found in {BUILD_DIR}")
        sys.exit(1)

    if not args.no_copy:
        dst = NODE_DIR / LIB_NAME
        shutil.copy2(str(src), str(dst))
        mb = dst.stat().st_size / (1024*1024)
        print(f"Copied {LIB_NAME} ({mb:.2f} MB)")

    print("BUILD COMPLETE")


if __name__ == "__main__":
    main()
