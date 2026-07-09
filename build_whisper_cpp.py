#!/usr/bin/env python3
"""Build whisper.cpp — auto-detect ALL GPU backends.
Supports: CUDA, Vulkan, Metal, HIP/ROCm, SYCL, OpenCL, OpenVINO
CPU: AVX, AVX2, AVX512, AMX, OpenMP, BLAS, LLAMAFILE
"""
import argparse, os, platform, shutil, subprocess, sys
from pathlib import Path

NODE_DIR    = Path(__file__).resolve().parent
WHISPER_DIR = NODE_DIR / "whisper.cpp"
BUILD_DIR   = WHISPER_DIR / "build"
IS_WIN      = platform.system() == "Windows"
IS_LINUX    = platform.system() == "Linux"
IS_MAC      = platform.system() == "Darwin"
LIB_EXT     = {"Windows": "dll", "Linux": "so", "Darwin": "dylib"}[platform.system()]
LIB_NAME    = f"whisper.{LIB_EXT}"


def _sh(*args, **kwargs):
    try:
        r = subprocess.run(args, capture_output=True, text=True, **kwargs)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return -1, "", f"not found: {args[0]}"


def detect_all():
    info = {"gpu": {}}

    info["platform"] = f"{platform.system()} {platform.release()} ({platform.machine()})"

    rc, out, _ = _sh("cmake", "--version")
    info["cmake"] = out.split("\n")[0] if rc == 0 else "NOT FOUND"

    if IS_WIN:
        rc, _, _ = _sh("cl.exe")
        info["compiler"] = "MSVC" if rc == 0 else "NOT FOUND"
    elif IS_MAC:
        _, out, _ = _sh("clang", "--version")
        info["compiler"] = out.split("\n")[0] if out else "NOT FOUND"
    else:
        _, out, _ = _sh("gcc", "--version")
        info["compiler"] = out.split("\n")[0] if out else "NOT FOUND"

    # ============ DETECT GPU BACKENDS ============

    # 1. CUDA (NVIDIA)
    cuda = "NOT DETECTED"
    if os.environ.get("CUDA_PATH"):
        cuda = f"CUDA_PATH={os.environ['CUDA_PATH']}"
    rc, out, _ = _sh("nvcc", "--version")
    if rc == 0:
        for line in out.split("\n"):
            if "release" in line:
                cuda = line.strip(); break
    info["gpu"]["CUDA"] = cuda

    # 2. Vulkan (AMD, NVIDIA, Intel)
    vk = "NOT DETECTED"
    if os.environ.get("VULKAN_SDK"):
        vk = f"SDK={os.environ['VULKAN_SDK']}"
    rc, out, _ = _sh("vulkaninfo", "--summary")
    if rc == 0:
        for line in out.split("\n"):
            if "GPU" in line and ":" in line:
                vk = line.strip(); break
        else:
            vk = "vulkaninfo OK"
    rc, out, _ = _sh("glslc", "--version")
    if rc == 0 and "NOT" in vk:
        vk = f"glslc: {out.split(chr(10))[0]}"
    info["gpu"]["Vulkan"] = vk

    # 3. Metal (Apple)
    if IS_MAC:
        rc, _, _ = _sh("xcrun", "-sdk", "macosx", "-f", "metallib")
        info["gpu"]["Metal"] = "OK" if rc == 0 else "NOT FOUND"
    else:
        info["gpu"]["Metal"] = "N/A (macOS only)"

    # 4. HIP/ROCm (AMD Linux)
    if IS_LINUX:
        hip = os.environ.get("ROCM_PATH", "") or "/opt/rocm"
        if os.path.isdir(hip):
            info["gpu"]["HIP/ROCm"] = f"ROCM={hip}"
        else:
            info["gpu"]["HIP/ROCm"] = "NOT DETECTED"
    else:
        info["gpu"]["HIP/ROCm"] = "N/A (Linux only)"

    # 5. SYCL (Intel)
    rc, _, _ = _sh("sycl-ls")
    info["gpu"]["SYCL (Intel)"] = "detected" if rc == 0 else "NOT DETECTED"

    # 6. OpenCL (legacy cross-platform)
    ocl = "NOT DETECTED"
    if IS_WIN:
        sysroot = os.environ.get("SystemRoot", "C:\\Windows")
        for sd in ["System32", "SysWOW64"]:
            if os.path.isfile(os.path.join(sysroot, sd, "OpenCL.dll")):
                ocl = f"found in {sd}"; break
    elif IS_LINUX:
        if os.path.isfile("/usr/lib/x86_64-linux-gnu/libOpenCL.so") or \
           os.path.isfile("/usr/lib/libOpenCL.so"):
            ocl = "libOpenCL.so found"
    else:
        rc, _, _ = _sh("oclconfig")
        if rc == 0: ocl = "detected"
    info["gpu"]["OpenCL"] = ocl

    # 7. OpenVINO (Intel)
    ov = "NOT DETECTED"
    if os.environ.get("OpenVINO_DIR") or os.environ.get("INTEL_OPENVINO_DIR"):
        ov = "found"
    info["gpu"]["OpenVINO (Intel)"] = ov

    # 8. BLAS (generic acceleration)
    info["gpu"]["BLAS"] = "auto (CMake)"

    # 9. LLAMAFILE (CPU optimization)
    info["gpu"]["LLAMAFILE (CPU)"] = "auto (CMake)"

    # 10. RPC (distributed)
    info["gpu"]["RPC"] = "disabled by default"

    return info


def print_report(info):
    w = 72
    print("=" * w)
    print("  WhisperCPP Build — Auto-Detection Report")
    print("=" * w)
    print(f"  Platform : {info['platform']}")
    print(f"  Compiler : {info['compiler']}")
    print(f"  CMake    : {info['cmake']}")
    print("-" * w)
    print("  GPU / Acceleration Backends:")
    for name, status in info["gpu"].items():
        ok = "NOT" not in status and "N/A" not in status and "disabled" not in status
        icon = "+" if ok else ("-" if "N/A" in status else " ")
        print(f"    [{icon}] {name}: {status}")
    print("=" * w)


def find_lib(build_dir, lib_name):
    candidates = [
        build_dir / "bin" / lib_name,
        build_dir / "src" / lib_name,
        build_dir / "lib" / lib_name,
        build_dir / lib_name,
    ]
    if IS_LINUX:
        for v in ["libwhisper.so.1", "libwhisper.so.1.9.1"]:
            candidates.append(build_dir / "bin" / v)
            candidates.append(build_dir / "src" / v)
            candidates.append(build_dir / "lib" / v)
    if IS_MAC:
        for v in ["libwhisper.1.dylib", "libwhisper.1.9.1.dylib"]:
            candidates.append(build_dir / "bin" / v)
            candidates.append(build_dir / "src" / v)
            candidates.append(build_dir / "lib" / v)
    if IS_WIN:
        for cfg in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel", ""]:
            for base in [build_dir / "bin", build_dir / "src", build_dir / "lib"]:
                candidates.append(base / cfg / lib_name if cfg else base / lib_name)
    return next((p for p in candidates if p.is_file()), None)


def verify_lib(lib_path):
    try:
        import ctypes
        lib = ctypes.cdll.LoadLibrary(str(lib_path))
        fn = lib.whisper_version; fn.restype = ctypes.c_char_p
        ver = fn().decode()
        # Cek compile info
        try:
            sysinfo = lib.whisper_print_system_info
            sysinfo.restype = ctypes.c_char_p
            info_str = sysinfo().decode()
            # Parse GPU info
            gpu_lines = [l for l in info_str.split("\n") if any(x in l for x in ["GPU", "CUDA", "Vulkan", "OpenCL", "Metal", "BLAS"])]
            print(f"  Verify: whisper.cpp v{ver}")
            for l in gpu_lines[:5]:
                print(f"    {l.strip()}")
        except:
            print(f"  Verify: whisper.cpp v{ver}")
        return True
    except Exception as e:
        print(f"  Verify FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build whisper.cpp with ALL GPU detection")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--gpu", choices=["auto","all","cuda","vulkan","metal","hip","rocm",
                                          "sycl","opencl","openvino","blas","none","cpu"],
                        default="all", help="GPU backend (default: all available)")
    parser.add_argument("--build-type", choices=["Release","Debug"], default="Release")
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--native", choices=["ON","OFF"], default="ON",
                        help="CPU-native optimization (ON=fast but machine-specific, OFF=portable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    info = detect_all()
    print(); print_report(info)
    if args.dry_run:
        return

    # Sanity checks
    if not (WHISPER_DIR / "CMakeLists.txt").exists():
        print(f"\nERROR: whisper.cpp submodule not found.\n  Run: git submodule update --init --recursive")
        sys.exit(1)
    if info["cmake"] == "NOT FOUND":
        print("\nERROR: CMake required. Install CMake first.")
        sys.exit(1)

    if args.clean and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR); print("  Cleaned build dir")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # === Determine backends ===
    backends = {"cuda": False, "vulkan": False, "metal": False, "hip": False,
                "sycl": False, "opencl": False, "openvino": False, "blas": False}
    mode = args.gpu

    def _ov_detected():
        return bool(os.environ.get("OpenVINO_DIR") or os.environ.get("INTEL_OPENVINO_DIR"))

    def _nvcc_present():
        return bool(os.environ.get("CUDA_PATH")) or _sh("nvcc", "--version")[0] == 0
    def _glslc_present():
        return _sh("glslc", "--version")[0] == 0

    if mode == "all":
        gpu = info["gpu"]
        # FULL backend — semua diwajibkan KALAU toolchain ada. Kalau gak ada
        # (mis. CI tanpa CUDA SDK), backend tersebut di-skip dengan warning agar
        # build tetap hijau. Runtime tetap aman: device gak ketemu -> fallback CPU.
        backends["cuda"]    = _nvcc_present()   # butuh nvcc/CUDA_PATH
        backends["vulkan"]  = _glslc_present()  # butuh glslc (Vulkan SDK)
        backends["metal"]   = IS_MAC
        backends["hip"]     = (not IS_WIN and not IS_MAC) and "NOT" not in gpu.get("HIP/ROCm", "NOT")
        backends["sycl"]    = False  # Intel oneAPI SDK terlalu berat untuk CI
        backends["opencl"]  = not IS_MAC  # WinSDK / Linux libOpenCL sudah ada
        backends["openvino"] = _ov_detected()  # hanya kalau OpenVINO SDK terinstall
        backends["blas"]    = IS_MAC
        print("  Mode: ALL backends (FULL — CUDA/Vulkan/OpenCL/Metal/OpenVINO + CPU SIMD)")
    elif mode == "auto":
        print("  Mode: Auto-detect available backends")
        gpu = info["gpu"]
        backends["cuda"]    = _nvcc_present()
        backends["vulkan"]  = "NOT" not in gpu["Vulkan"]
        backends["metal"]   = "NOT" not in gpu["Metal"] and "N/A" not in gpu["Metal"]
        backends["hip"]     = "NOT" not in gpu["HIP/ROCm"] and "N/A" not in gpu["HIP/ROCm"]
        backends["sycl"]    = "NOT" not in gpu["SYCL (Intel)"]
        backends["opencl"]  = not IS_MAC and "NOT" not in gpu["OpenCL"]
        backends["openvino"] = _ov_detected()
        backends["blas"]    = IS_MAC
    elif mode == "none" or mode == "cpu":
        pass  # CPU only
    else:
        # Map single backend
        single_map = {"cuda":"cuda","vulkan":"vulkan","metal":"metal",
                      "hip":"hip","rocm":"hip","sycl":"sycl",
                      "opencl":"opencl","openvino":"openvino","blas":"blas"}
        if mode in single_map:
            backends[single_map[mode]] = True

    # === Build CMake args ===
    flag_map = {
        "cuda":   ("GGML_CUDA", "CUDA"),
        "vulkan": ("GGML_VULKAN", "Vulkan"),
        "metal":  ("GGML_METAL", "Metal"),
        "hip":    ("GGML_HIP", "HIP/ROCm"),
        "sycl":   ("GGML_SYCL", "SYCL"),
        "opencl": ("GGML_OPENCL", "OpenCL"),
        "openvino": ("GGML_OPENVINO", "OpenVINO"),
        "blas":   ("GGML_BLAS", "BLAS"),
    }

    # CUDA arch: auto-detect dari GPU atau fallback ke env var
    cuda_arch_opt = os.environ.get("CMAKE_CUDA_ARCHITECTURES", "89")  # RTX 4070 SUPER default
    cuda_arch = f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch_opt}"
    cmake_args = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(WHISPER_DIR),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
        f"-DGGML_NATIVE={args.native}",
        "-DGGML_OPENMP=ON",
        "-DGGML_LLAMAFILE=ON",
        # === FULL CPU SIMD (wajib, bukan opsional) ===
        "-DGGML_AVX=ON",
        "-DGGML_AVX2=ON",
        "-DGGML_AVX512=ON",
        "-DGGML_AVX512_VBMI=ON",
        "-DGGML_AVX512_VNNI=ON",
        "-DGGML_AVX512_BF16=ON",
        "-DGGML_FMA=ON",
        "-DGGML_F16C=ON",
        "-DGGML_AMX_TILE=ON",
        "-DGGML_AMX_INT8=ON",
        "-DGGML_AMX_BF16=ON",
    ]
    # CUDA-specific flags
    if backends.get("cuda"):
        cmake_args.append(cuda_arch)
        # Batasi thread nvcc agar gak OOM di CI (kompilasi kernel CUDA makan RAM besar)
        cmake_args.append("-DCMAKE_CUDA_FLAGS=--threads 4")
        # Force VS2022 (v143) toolset di Windows -> kompatibel dengan CUDA 13.x.
        # (VS2026/v144 belum didukung nvcc, build akan gagal kalau dipakai.)
        if IS_WIN:
            cmake_args.append("-T")
            cmake_args.append("v143")
            if "CUDA_PATH" in os.environ:
                cuda_root = os.environ["CUDA_PATH"]
                cmake_args.append(f"-DCMAKE_CUDA_COMPILER:FILEPATH={cuda_root}/bin/nvcc.exe")
                cmake_args.append(f"-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES={cuda_root}/include")
        elif IS_LINUX and os.path.isfile("/usr/local/cuda/bin/nvcc"):
            cmake_args.append("-DCMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc")
            cmake_args.append("-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=/usr/local/cuda/include")
        elif "CUDA_HOME" in os.environ:
            cuda_root = os.environ["CUDA_HOME"]
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER:FILEPATH={cuda_root}/bin/nvcc")
            cmake_args.append(f"-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES={cuda_root}/include")
    # OpenCL paths
    if backends.get("opencl"):
        if IS_WIN and "CUDA_PATH" in os.environ:
            cuda_root = os.environ["CUDA_PATH"]
            cmake_args.append(f"-DOpenCL_INCLUDE_DIR={cuda_root}/include")
            cmake_args.append(f"-DOpenCL_LIBRARY={cuda_root}/lib/x64/OpenCL.lib")
        elif IS_WIN:
            # Find OpenCL.lib in Windows SDK (cmake's FindOpenCL doesn't search there)
            import glob
            sdk_lib_paths = glob.glob("C:/Program Files (x86)/Windows Kits/10/Lib/*/um/x64/OpenCL.lib")
            sdk_include_paths = glob.glob("C:/Program Files (x86)/Windows Kits/10/Include/*/um/CL/cl.h")
            if sdk_lib_paths and sdk_include_paths:
                cmake_args.append(f"-DOpenCL_LIBRARY={sdk_lib_paths[0]}")
                inc_dir = sdk_include_paths[0].rsplit("CL", 1)[0]
                cmake_args.append(f"-DOpenCL_INCLUDE_DIR={inc_dir}")
                print(f"Found OpenCL in WinSDK: lib={sdk_lib_paths[0]}, inc={inc_dir}")
            else:
                backends["opencl"] = False
                print("OpenCL not found in WinSDK, disabling")
        elif IS_LINUX:
            # OpenCL usually via system package: libOpenCL.so
            cmake_args.append("-DOpenCL_INCLUDE_DIR=/usr/include/CL")
            cmake_args.append("-DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so")
        elif "OpenCL_INCLUDE_DIR" in os.environ and "OpenCL_LIBRARY" in os.environ:
            cmake_args.append(f"-DOpenCL_INCLUDE_DIR={os.environ['OpenCL_INCLUDE_DIR']}")
            cmake_args.append(f"-DOpenCL_LIBRARY={os.environ['OpenCL_LIBRARY']}")

    any_gpu = False
    for bk, enabled in backends.items():
        if enabled:
            flag, label = flag_map[bk]
            cmake_args.append(f"-D{flag}=ON")
            print(f"  + {flag} ({label})")
            any_gpu = True

    if not any_gpu:
        print("  CPU only (no GPU backends enabled)")
        print("  CPU optimizations: AVX/AVX2/AMX/OpenMP/LLAMAFILE")

    # Configure & Build
    print(f"\n  Configuring ({args.build_type})...")
    if subprocess.run(cmake_args).returncode != 0:
        print("  CMake configuration FAILED. Check errors above.")
        sys.exit(1)

    print(f"  Building ({args.build_type})...")
    # Batasi paralelisme build agar gak OOM di CI (CUDA kernel compile butuh RAM besar)
    import multiprocessing
    _ncpu = max(1, min(multiprocessing.cpu_count() or 2, 4))
    if IS_LINUX:
        _ncpu = 2  # CI Linux sering OOM kalau full parallel + CUDA
    ret = subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", args.build_type, "-j", str(_ncpu)])
    if ret.returncode != 0:
        print("  Build FAILED"); sys.exit(1)

    src_path = find_lib(BUILD_DIR, LIB_NAME)
    if not src_path:
        print(f"  Build done but {LIB_NAME} not located in {BUILD_DIR}")
        sys.exit(1)

    if not args.no_copy:
        dst = NODE_DIR / LIB_NAME
        shutil.copy2(str(src_path.resolve()), str(dst))
        mb = os.path.getsize(dst) / (1024*1024)
        print(f"\n  Copied {dst.name} ({mb:.1f} MB)")
        verify_lib(dst)

    print("\n  BUILD COMPLETE")


if __name__ == "__main__":
    main()
