#!/usr/bin/env python3
"""Build bs_roformer.cpp with VS toolchain."""
import os, subprocess, sys, shutil
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
BUILD_DIR = NODE_DIR / "bs_roformer.cpp" / "build"
GGML_DIR  = NODE_DIR / "whisper.cpp" / "ggml"

VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
CMAKE  = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
MSBUILD = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe"

def run(cmd_list):
    cmake_dir = os.path.dirname(CMAKE)
    msbuild_dir = os.path.dirname(MSBUILD)
    chain = " && ".join(cmd_list)
    # Build a clean PATH without depot_tools
    clean_path = f"{cmake_dir};{msbuild_dir};C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem"
    full = f'call "{VCVARS}" x64 >nul 2>&1 && set "PATH={clean_path}" && {chain}'
    r = subprocess.run(full, shell=True, cwd=NODE_DIR)
    return r.returncode == 0

def main():
    cmake_args = ["-DCMAKE_BUILD_TYPE=Release", "-DBSR_BUILD_TESTS=OFF", "-DGGML_CUDA=OFF"]
    if GGML_DIR.exists():
        cmake_args.append(f"-DGGML_DIR={GGML_DIR}")

    # Clean stale build
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Figure out VS generator name
    print("Detecting VS generator...")
    gen = "Visual Studio 18 2026"
    toolset_option = []

    print(f"Configuring with generator={gen}...")
    if not run([
        f'cd /d "{BUILD_DIR}"',
        f'"{CMAKE}" .. -G "{gen}" {" ".join(cmake_args)}',
    ]):
        print("Config FAILED"); return 1

    print("Building...")
    if not run([
        f'cd /d "{BUILD_DIR}"',
        f'"{CMAKE}" --build . --config Release -- /m',
    ]):
        print("Build FAILED"); return 1

    bin_found = list(BUILD_DIR.rglob("bs_roformer-cli.exe"))
    if bin_found:
        shutil.copy2(bin_found[0], NODE_DIR / "bs_roformer-cli.exe")
        print("Copied bs_roformer-cli.exe")
    else:
        print("Binary not found"); return 1

    print("Build OK!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
