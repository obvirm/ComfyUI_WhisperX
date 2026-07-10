#!/usr/bin/env python3
"""Build cpp-annote shared library (DLL/so/dylib).
Supports CPU and CUDA backends.
"""
import argparse, io, os, platform, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
ANNOTE_DIR = NODE_DIR / "cpp-annote"
BUILD_DIR = ANNOTE_DIR / "build"
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

ORT_VERSION = "1.27.0"

def _detect_cuda_major():
    """Return CUDA major version (int) or 0 if not found."""
    cp = os.environ.get("CUDA_PATH", "")
    import re
    m = re.search(r"v?(\d+)", cp)
    if m:
        return int(m.group(1))
    return 0

def download_ort(cuda=False, directml=False, coreml=False, rocm=False, openvino=False):
    """Download ONNX Runtime with specific provider support.
    v1.27.0 packages:
      Windows: onnxruntime-win-x64-{VERSION}.zip (CPU)
               onnxruntime-win-x64-gpu_cuda12-{VERSION}.zip
               onnxruntime-win-x64-gpu_cuda13-{VERSION}.zip
               onnxruntime-win-x64-openvino-{VERSION}.zip
      Linux:   onnxruntime-linux-x64-{VERSION}.tgz (CPU)
               onnxruntime-linux-x64-gpu_cuda12-{VERSION}.tgz
               onnxruntime-linux-x64-gpu_cuda13-{VERSION}.tgz
      macOS:   onnxruntime-osx-arm64-{VERSION}.tgz (CoreML built-in)
    DirectML: bundled inside the standard win-x64 GPU (CUDA) build.
    """
    cuda_major = _detect_cuda_major()
    cuda_tag = f"gpu_cuda{cuda_major}" if (cuda and cuda_major >= 13) else "gpu_cuda12"
    if IS_WIN:
        if cuda or directml:
            # DML + CUDA + OpenVINO header semua ada di dalam paket CUDA (gpu) build.
            # Paket openvino standalone belum rilis utk 1.27.0, jadi pakai gpu_cuda build.
            ort_name = f"onnxruntime-win-x64-{cuda_tag}-{ORT_VERSION}"
        else:
            ort_name = f"onnxruntime-win-x64-{ORT_VERSION}"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.zip"
    elif platform.system() == "Linux":
        if cuda:
            ort_name = f"onnxruntime-linux-x64-{cuda_tag}-{ORT_VERSION}"
        else:
            ort_name = f"onnxruntime-linux-x64-{ORT_VERSION}"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.tgz"
    elif IS_MAC:
        ort_name = f"onnxruntime-osx-arm64-{ORT_VERSION}" if platform.machine() == "arm64" else f"onnxruntime-osx-x86_64-{ORT_VERSION}"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.tgz"
    else:
        return None

    ort_dir = ANNOTE_DIR / ort_name
    if ort_dir.exists() and (ort_dir / "include").exists():
        print(f"ONNX Runtime already present: {ort_name}")
        return ort_dir
    # Clean up partial/empty directory
    if ort_dir.exists():
        import shutil
        shutil.rmtree(ort_dir, ignore_errors=True)

    print(f"Downloading {ort_name}...")
    try:
        data = urllib.request.urlopen(url, timeout=300).read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(ANNOTE_DIR)
        else:
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                tf.extractall(ANNOTE_DIR)
        print(f"Downloaded to {ort_dir}")
        return ort_dir
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Build cpp-annote shared library")
    parser.add_argument("--clean", action="store_true", help="Clean build dir")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend (NVIDIA)")
    parser.add_argument("--directml", action="store_true", help="Enable DirectML backend (any GPU on Windows)")
    parser.add_argument("--coreml", action="store_true", help="Enable CoreML backend (macOS Metal)")
    parser.add_argument("--rocm", action="store_true", help="Enable ROCm backend (AMD)")
    parser.add_argument("--gpu", action="store_true", help="Auto-detect best GPU backend (DEPRECATED — now default)")
    parser.add_argument("--openvino", action="store_true", help="Enable OpenVINO backend (Intel NPU/CPU)")
    parser.add_argument("--cpu-only", action="store_true", help="Build CPU-only (skip GPU backends)")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy to root")
    args = parser.parse_args()
    # FULL backend: --gpu sekarang default ON kecuali --cpu-only
    if not args.cpu_only:
        args.gpu = True

    # --gpu auto-detect — FULL backend (default ON)
    if args.gpu:
        import shutil as _shutil
        if platform.system() == "Windows":
            args.directml = True  # DirectML = NPU/any-GPU di Windows
            if _shutil.which("nvidia-smi"):
                args.cuda = True
            if _detect_cuda_major() >= 13 or os.environ.get("OpenVINO_DIR"):
                args.openvino = True  # OpenVINO = Intel NPU
        elif platform.system() == "Darwin":
            args.coreml = True  # CoreML = Apple NPU/ANE
        elif _shutil.which("nvidia-smi"):
            args.cuda = True
        # ROCm detection not implemented

    # Download ONNX Runtime with selected provider (full GPU)
    ort_dir = download_ort(cuda=args.cuda, directml=args.directml, coreml=args.coreml, rocm=args.rocm, openvino=args.openvino)
    if not ort_dir:
        print("ERROR: Could not download ONNX Runtime")
        sys.exit(1)

    # Find ONNX Runtime
    ort_include = ort_dir / "include"
    ort_lib = ort_dir / "lib"
    if not ort_include.exists():
        # Try nested path
        for p in ort_dir.iterdir():
            if p.is_dir() and (p / "include").exists():
                ort_include = p / "include"
                ort_lib = p / "lib"
                ort_dir = p  # Use the nested dir as root
                break
    
    # Debug: print directory contents
    print(f"ORT root: {ort_dir}")
    if ort_dir.exists():
        print(f"  root contents: {[p.name for p in ort_dir.iterdir()]}")
    print(f"ORT include: {ort_include} (exists={ort_include.exists()})")
    print(f"ORT lib: {ort_lib} (exists={ort_lib.exists()})")
    if ort_include.exists():
        print(f"  include contents: {list(ort_include.iterdir())[:10]}")
    if ort_lib.exists():
        print(f"  lib contents: {list(ort_lib.iterdir())[:10]}")

    # Generator — force VS2022 (v143) di Windows agar cocok dengan CUDA 13.x
    if IS_WIN:
        generator = "Visual Studio 17 2022"
    elif IS_MAC:
        generator = "Unix Makefiles"
    else:
        generator = "Unix Makefiles"

    # Configure
    providers = []
    if args.cuda: providers.append("CUDA")
    if args.directml: providers.append("DirectML")
    if args.coreml: providers.append("CoreML")
    if args.openvino: providers.append("OpenVINO")
    print(f"Configuring (providers: {','.join(providers) or 'CPU'})...")
    cmd = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(ANNOTE_DIR),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-G", generator,
        f"-DONNXRUNTIME_ROOT={ort_dir}",
        f"-DCPPANNOTE_BUILD_SHARED=ON",
    ]
    if IS_WIN:
        cmd.append("-T")
        cmd.append("v143")  # VS2022 toolset (CUDA 13.x compatible)
        cmd.append("-DCPPANNOTE_BUILD_SHARED=ON")
    # Inject ORT provider compile flags sesuai paket yang di-download
    # macOS selalu pakai CoreML (gak perlu flag khusus, struct di-guard __APPLE__).
    # Windows/Linux: CUDA/DML/OpenVINO flag aktif sesuai download.
    if args.cuda:
        cmd.append("-DCPPANNOTE_ORT_CUDA=ON")
    if args.directml:
        cmd.append("-DCPPANNOTE_ORT_DML=ON")
    if args.openvino:
        # OpenVINO EP header ada di dalam paket CUDA (gpu) ORT. Paket openvino standalone
        # (onnxruntime-win-x64-openvino) belum rilis utk 1.27.0 -> skip flag biar build tetap hijau.
        # Runtime: kalau openvino_provider_factory.h ada, header ter-include via top-level guard.
        if (ort_dir / "include" / "openvino_provider_factory.h").exists():
            cmd.append("-DCPPANNOTE_ORT_OPENVINO=ON")
        else:
            print("  OpenVINO provider header not in ORT package; skipping OpenVINO compile flag")

    r = subprocess.run(cmd, cwd=str(ANNOTE_DIR))
    if r.returncode != 0:
        print("CMake configure failed!")
        sys.exit(1)

    # Build
    target = "cpp_annote_shared" if IS_WIN else "cpp_annote_shared"
    print(f"Building {target}...")
    cmd = ["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", target]
    if IS_WIN:
        cmd.extend(["--", "/m"])
    r = subprocess.run(cmd, cwd=str(ANNOTE_DIR))
    if r.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    # Find output
    lib_ext = "dll" if IS_WIN else "so" if not IS_MAC else "dylib"
    lib_name = f"cpp_annote.{lib_ext}"
    candidates = [
        BUILD_DIR / "Release" / lib_name,
        BUILD_DIR / lib_name,
        BUILD_DIR / "bin" / "Release" / lib_name,
        BUILD_DIR / "bin" / lib_name,
    ]
    if not IS_WIN:
        import glob
        for p in glob.glob(str(BUILD_DIR / f"lib{lib_name}*")):
            candidates.append(Path(p))
        for p in glob.glob(str(BUILD_DIR / f"libcpp_annote*{lib_ext}*")):
            candidates.append(Path(p))

    found = next((p for p in candidates if p.is_file()), None)
    if not found:
        print(f"ERROR: {lib_name} not found in build output")
        sys.exit(1)

    # Copy to root
    if not args.no_copy:
        dest = NODE_DIR / lib_name
        shutil.copy2(str(found), str(dest))
        print(f"Copied to {dest}")

        # Copy ONNX Runtime libs
        if IS_WIN:
            for dll in (ort_lib / "onnxruntime*.dll").glob("*"):
                shutil.copy2(str(dll), str(NODE_DIR / dll.name))
        elif not IS_MAC:
            for so in (ort_lib / "libonnxruntime*.so*").glob("*"):
                shutil.copy2(str(so), str(NODE_DIR / so.name))

    print(f"Build complete: {found}")
    return 0


if __name__ == "__main__":
    import io
    sys.exit(main())
