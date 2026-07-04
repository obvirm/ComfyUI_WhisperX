#!/usr/bin/env python3
"""Build cpp-annote shared library (DLL/so/dylib).
Supports CPU and CUDA backends.
"""
import argparse, os, platform, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
ANNOTE_DIR = NODE_DIR / "cpp-annote"
BUILD_DIR = ANNOTE_DIR / "build"
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

ORT_VERSION = "1.27.0"

def download_ort(cuda=False):
    """Download ONNX Runtime if not present."""
    if IS_WIN:
        if cuda:
            ort_name = f"onnxruntime-win-x64-gpu-{ORT_VERSION}"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.zip"
        else:
            ort_name = f"onnxruntime-win-x64-{ORT_VERSION}"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.zip"
    elif platform.system() == "Linux":
        if cuda:
            ort_name = f"onnxruntime-linux-x64-gpu-{ORT_VERSION}"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.tar.gz"
        else:
            ort_name = f"onnxruntime-linux-x64-{ORT_VERSION}"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.tar.gz"
    elif IS_MAC:
        ort_name = f"onnxruntime-osx-arm64-{ORT_VERSION}"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort_name}.tgz"
    else:
        return None

    ort_dir = ANNOTE_DIR / ort_name
    if ort_dir.exists():
        print(f"ONNX Runtime already present: {ort_name}")
        return ort_dir

    print(f"Downloading {ort_name}...")
    try:
        data = urllib.request.urlopen(url, timeout=300).read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(ANNOTE_DIR)
        else:
            import tarfile, io
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
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy to root")
    args = parser.parse_args()

    if not ANNOTE_DIR.is_dir() or not (ANNOTE_DIR / "CMakeLists.txt").is_file():
        print("ERROR: cpp-annote directory not found.")
        sys.exit(1)

    if args.clean and BUILD_DIR.is_dir():
        shutil.rmtree(BUILD_DIR)
        print("Cleaned build dir")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Download ONNX Runtime
    ort_dir = download_ort(cuda=args.cuda)
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
                break

    # Generator
    if IS_WIN:
        generator = "Visual Studio 18 2026"
    elif IS_MAC:
        generator = "Unix Makefiles"
    else:
        generator = "Unix Makefiles"

    # Configure
    print(f"Configuring (CUDA={'ON' if args.cuda else 'OFF'})...")
    cmd = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(ANNOTE_DIR),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-G", generator,
        f"-DONNXRUNTIME_ROOT={ort_dir}",
        f"-DCPPANNOTE_BUILD_SHARED=ON",
    ]
    if IS_WIN:
        cmd.append("-DCPPANNOTE_BUILD_SHARED=ON")

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
