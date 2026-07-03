#!/usr/bin/env python3
"""Build whisper.cpp shared library with GPU detection."""
import argparse, os, platform, shutil, subprocess, sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
WHISPER_DIR = NODE_DIR / "whisper.cpp"
BUILD_DIR = WHISPER_DIR / "build"


def main():
    parser = argparse.ArgumentParser(description="Build whisper.cpp shared library")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--gpu", choices=["auto", "cuda", "vulkan", "metal", "none"],
                        default="auto", help="GPU backend")
    parser.add_argument("--build-type", choices=["Release", "Debug"],
                        default="Release", help="Build type")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip copying .so/.dll to node root")
    args = parser.parse_args()

    if not (WHISPER_DIR / "CMakeLists.txt").exists():
        print("❌ whisper.cpp submodule not found. Run: git submodule update --init --recursive")
        sys.exit(1)

    try:
        subprocess.run(["cmake", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ CMake not found. Install CMake first.")
        sys.exit(1)

    if args.clean and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print("🧹 Cleaned build directory")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(WHISPER_DIR),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
    ]

    gpu = args.gpu
    if gpu == "auto":
        if platform.system() == "Darwin":
            cmake_args.append("-DWHISPER_METAL=ON")
            print("🔵 Auto-detected macOS → Metal")
        else:
            try:
                subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
                cmake_args.append("-DWHISPER_CUDA=ON")
                print("🟢 Auto-detected CUDA")
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("🟡 No CUDA found, CPU only")
    elif gpu == "cuda":
        cmake_args.append("-DWHISPER_CUDA=ON")
        print("🟢 CUDA enabled")
    elif gpu == "vulkan":
        cmake_args.append("-DWHISPER_VULKAN=ON")
        print("🟣 Vulkan enabled")
    elif gpu == "metal":
        cmake_args.append("-DWHISPER_METAL=ON")
        print("🔵 Metal enabled")
    else:
        print("⚪ CPU only")

    print(f"\n🔧 Configuring CMake ({args.build_type})...")
    subprocess.run(cmake_args, check=True)

    print(f"\n🏗️  Building ({args.build_type})...")
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", args.build_type, "-j"],
                   check=True)

    # Locate built library
    lib_names = {
        "Windows": ["whisper.dll"],
        "Linux": ["libwhisper.so", "libwhisper.so.1", "libwhisper.so.1.9.1"],
        "Darwin": ["libwhisper.dylib"],
    }
    names = lib_names.get(platform.system(), ["libwhisper.so"])

    candidates = []
    for name in names:
        for base in [BUILD_DIR / "bin", BUILD_DIR / "src", BUILD_DIR / "lib"]:
            candidates.append(base / name)
        if args.build_type == "Debug":
            for base in [BUILD_DIR / "bin" / "Debug", BUILD_DIR / "src" / "Debug"]:
                candidates.append(base / name)

    src_path = next((p for p in candidates if p.is_file()), None)

    if not src_path:
        print(f"\n⚠️  Build succeeded but output library not found in expected paths.")
        print(f"   Searched in:")
        for p in candidates:
            print(f"     - {p}")
        print(f"\n   Check {BUILD_DIR / 'bin'} manually.")
        sys.exit(1)

    if not args.no_copy:
        lib_ext = {"Windows": "dll", "Linux": "so", "Darwin": "dylib"}.get(
            platform.system(), "so")
        dst = NODE_DIR / f"whisper.{lib_ext}"
        # resolve symlink to actual file
        real = src_path.resolve()
        shutil.copy2(str(real), str(dst))
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"\n✅ Copied {real.name} ({size_mb:.1f} MB) → {dst}")
    else:
        print(f"\n✅ Build complete (--no-copy, library at {src_path})")


def verify_library(lib_path: str) -> bool:
    """Test that the library loads and returns a version string."""
    try:
        import ctypes
        lib = ctypes.cdll.LoadLibrary(lib_path)
        fn = lib.whisper_version
        fn.restype = ctypes.c_char_p
        ver = fn()
        print(f"   whisper.cpp version: {ver.decode()}")
        return True
    except Exception as e:
        print(f"   ❌ Library verification failed: {e}")
        return False


if __name__ == "__main__":
    main()
    # verify
    lib_ext = {"Windows": "dll", "Linux": "so", "Darwin": "dylib"}.get(
        platform.system(), "so")
    lib_path = Path(__file__).resolve().parent / f"whisper.{lib_ext}"
    if lib_path.exists():
        print("\n🔍 Verifying library...")
        ok = verify_library(str(lib_path))
        if ok:
            print("   ✅ whisper.cpp library loads and works!")
        else:
            print("   ⚠️  Library file exists but verification failed.")
            print("   Node will work once library path is configured.")
