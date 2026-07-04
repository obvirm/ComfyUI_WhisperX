"""
GPU Detection — auto-select best backend for whisper.cpp and cpp-annote.

Detection order:
  1. NVIDIA CUDA → use CUDA backend
  2. Vulkan → use Vulkan backend
  3. OpenCL → use OpenCL backend
  4. CPU only → fallback

whisper.cpp: auto-detects at build time, DLLs included in release
cpp-annote: needs CUDA ONNX Runtime for GPU, CPU for fallback
"""
import platform
import subprocess
import logging
import os

logger = logging.getLogger("WhisperCPP")

def detect_gpu() -> dict:
    """Detect available GPUs and return info."""
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "has_nvidia": False,
        "has_amd": False,
        "has_intel": False,
        "gpu_name": None,
        "backend": "cpu",  # default
    }
    
    if platform.system() == "Windows":
        info.update(_detect_windows())
    elif platform.system() == "Linux":
        info.update(_detect_linux())
    elif platform.system() == "Darwin":
        info["backend"] = "metal"  # macOS always has Metal
    
    # Priority: CUDA > Vulkan > OpenCL > CPU
    if info["has_nvidia"]:
        info["backend"] = "cuda"
    elif info.get("has_vulkan"):
        info["backend"] = "vulkan"
    elif info.get("has_opencl"):
        info["backend"] = "opencl"
    
    return info


def _detect_windows() -> dict:
    """Detect GPU on Windows via WMI/nvidia-smi."""
    info = {"has_nvidia": False, "has_amd": False, "has_intel": False}
    
    # Check NVIDIA via nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            info["has_nvidia"] = True
            info["gpu_name"] = r.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check Vulkan
    info["has_vulkan"] = os.path.isfile(
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32", "vulkan-1.dll")
    )
    
    # Check OpenCL
    info["has_opencl"] = os.path.isfile(
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32", "OpenCL.dll")
    )
    
    return info


def _detect_linux() -> dict:
    """Detect GPU on Linux."""
    info = {"has_nvidia": False, "has_amd": False, "has_intel": False}
    
    # Check NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            info["has_nvidia"] = True
            info["gpu_name"] = r.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check Vulkan
    info["has_vulkan"] = os.path.isfile("/usr/lib/x86_64-linux-gnu/libvulkan.so.1") or \
                         os.path.isfile("/usr/lib/libvulkan.so.1")
    
    # Check OpenCL
    info["has_opencl"] = os.path.isfile("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1") or \
                         os.path.isfile("/usr/lib/libOpenCL.so.1")
    
    return info


def get_release_tag() -> str:
    """Get the latest release tag for auto-download."""
    return "v2.0.3"  # update on each release


def get_dll_zip_name() -> str:
    """Get the correct zip filename based on platform."""
    system = platform.system()
    if system == "Windows":
        return "whisper-cpp-win64.zip"
    elif system == "Linux":
        return "whisper-cpp-linux-x64.zip"
    elif system == "Darwin":
        if platform.machine() == "arm64":
            return "whisper-cpp-macos-arm64.zip"
        return "whisper-cpp-macos-x64.zip"
    return ""


if __name__ == "__main__":
    info = detect_gpu()
    print(f"Platform: {info['platform']} {info['arch']}")
    print(f"GPU: {info['gpu_name'] or 'None detected'}")
    print(f"NVIDIA: {info['has_nvidia']}, Vulkan: {info.get('has_vulkan', False)}, OpenCL: {info.get('has_opencl', False)}")
    print(f"Selected backend: {info['backend']}")
