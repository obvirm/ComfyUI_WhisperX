#!/usr/bin/env python3
import argparse, os, platform, shutil, subprocess, sys
from pathlib import Path
NODE_DIR = Path(__file__).resolve().parent
WHISPER_DIR = NODE_DIR / "whisper.cpp"
BUILD_DIR = WHISPER_DIR / "build"
IS_WIN = platform.system() == "Windows"

def main():
    parser = argparse.ArgumentParser(description="Build whisper.cpp shared library")
    parser.add_argument("--clean", action="store_true", help="Clean build")
    parser.add_argument("--gpu", choices=["auto","cuda","vulkan","metal","none"], default="auto", help="GPU backend")
    parser.add_argument("--build-type", choices=["Release","Debug"], default="Release", help="Build type")
    args = parser.parse_args()

    if not (WHISPER_DIR / "CMakeLists.txt").exists():
        print("❌ whisper.cpp submodule not found. Run: git submodule update --init")
        sys.exit(1)

    try: subprocess.run(["cmake","--version"], capture_output=True, check=True)
    except: print("❌ CMake not found"); sys.exit(1)

    if args.clean and BUILD_DIR.exists(): shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_args = ["cmake","-B",str(BUILD_DIR),"-S",str(WHISPER_DIR),f"-DCMAKE_BUILD_TYPE={args.build_type}","-DBUILD_SHARED_LIBS=ON","-DWHISPER_BUILD_TESTS=OFF","-DWHISPER_BUILD_EXAMPLES=OFF"]

    gpu = args.gpu
    if gpu == "auto":
        if platform.system() == "Darwin": cmake_args.append("-DWHISPER_METAL=ON")
        else:
            try: subprocess.run(["nvcc","--version"], capture_output=True, check=True); cmake_args.append("-DWHISPER_CUDA=ON")
            except: pass
    elif gpu == "cuda": cmake_args.append("-DWHISPER_CUDA=ON")
    elif gpu == "vulkan": cmake_args.append("-DWHISPER_VULKAN=ON")
    elif gpu == "metal": cmake_args.append("-DWHISPER_METAL=ON")

    subprocess.run(cmake_args, check=True)
    subprocess.run(["cmake","--build",str(BUILD_DIR),"--config",args.build_type,"-j"], check=True)

    lib_name = {"Windows":"whisper.dll","Linux":"libwhisper.so","Darwin":"libwhisper.dylib"}.get(platform.system(),"libwhisper.so")
    src = BUILD_DIR / "src" / lib_name
    if not src.exists(): src = BUILD_DIR / "src" / args.build_type / lib_name
    if src.exists():
        dst = NODE_DIR / lib_name
        shutil.copy2(str(src), str(dst))
        print(f"✅ Copied to {dst}")
    else:
        print(f"⚠️ Build done but {lib_name} not found in {BUILD_DIR}")

if __name__ == "__main__": main()
