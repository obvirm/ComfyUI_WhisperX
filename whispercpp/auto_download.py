"""
Auto-download DLLs/SOs from GitHub Releases.

All 3 modules (whisper, bs_roformer, cpp_annote) use this.
Downloads individual files based on platform + GPU detection.
"""
import os
import platform
import urllib.request
import logging

logger = logging.getLogger("WhisperCPP")

GITHUB_REPO = "obvirm/ComfyUI-WhisperCPP"
CURRENT_VERSION = "v2.0.7"

IS_WIN = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

# Files needed per module per platform
# fmt: off
ASSETS = {
    "whisper": {
        "Windows": ["whisper.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml.dll"],
        "Linux":   ["libwhisper.so", "libggml-base.so", "libggml-cpu.so", "libggml.so"],
        "Darwin":  ["libwhisper.dylib", "libggml-base.dylib", "libggml-cpu.dylib", "libggml.dylib"],
    },
    "bs_roformer": {
        "Windows": ["bs_roformer.dll"],
        "Linux":   ["libbs_roformer.so"],
        "Darwin":  ["libbs_roformer.dylib"],
    },
    "cpp_annote": {
        "Windows": ["cpp_annote.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"],
        "Linux":   ["libcpp_annote.so", "libonnxruntime.so", "libonnxruntime_providers_shared.so"],
        "Darwin":  ["libcpp_annote.dylib"],
    },
}

# GPU-specific assets (only needed when GPU is available)
GPU_ASSETS = {
    "whisper": {
        "Windows": ["ggml-vulkan.dll"],
        "Linux":   ["libggml-vulkan.so", "libggml-opencl.so"],
        "Darwin":  [],  # Metal is built-in via ggml-metal
    },
}
# fmt: on


def _get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        import tomllib
        pyproject = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pyproject.toml"
        )
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return "v" + data["project"]["version"]
    except Exception:
        return CURRENT_VERSION


def _get_download_url(asset_name: str, version: str = None) -> str:
    """Build GitHub release download URL."""
    v = version or _get_version()
    return f"https://github.com/{GITHUB_REPO}/releases/download/{v}/{asset_name}"


def download_file(url: str, dest: str, timeout: int = 120) -> bool:
    """Download a single file. Returns True on success."""
    try:
        # Skip if already exists
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            return True
        logger.info(f"  Downloading {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
        # Set executable permission on Linux/macOS
        if not IS_WIN:
            os.chmod(dest, 0o755)
        return True
    except Exception as e:
        logger.warning(f"  Failed to download {os.path.basename(dest)}: {e}")
        # Clean up partial download
        try:
            if os.path.isfile(dest):
                os.remove(dest)
        except OSError:
            pass
        return False


def download_module(module: str, target_dir: str, version: str = None,
                    has_gpu: bool = False) -> bool:
    """
    Download all files for a module (whisper/bs_roformer/cpp_annote).
    
    Args:
        module: "whisper", "bs_roformer", or "cpp_annote"
        target_dir: Directory to save files to
        version: Release tag (default: CURRENT_VERSION)
        has_gpu: Whether to include GPU-specific assets
        
    Returns:
        True if all required files downloaded successfully
    """
    system = platform.system()
    if system not in ASSETS.get(module, {}):
        logger.warning(f"No pre-built binaries for {module} on {system}")
        return False

    files = list(ASSETS[module][system])
    if has_gpu:
        gpu_files = GPU_ASSETS.get(module, {}).get(system, [])
        files.extend(gpu_files)

    os.makedirs(target_dir, exist_ok=True)

    success_count = 0
    for fname in files:
        url = _get_download_url(fname, version)
        dest = os.path.join(target_dir, fname)
        if download_file(url, dest):
            success_count += 1

    total = len(files)
    if success_count == total:
        logger.info(f"  {module}: {success_count}/{total} files downloaded")
        return True
    else:
        logger.warning(f"  {module}: only {success_count}/{total} files downloaded")
        return False


def check_module_files(module: str, target_dir: str) -> bool:
    """Check if all required files for a module exist."""
    system = platform.system()
    if system not in ASSETS.get(module, {}):
        return False

    for fname in ASSETS[module][system]:
        if not os.path.isfile(os.path.join(target_dir, fname)):
            return False
    return True


def auto_download_all(target_dir: str, version: str = None,
                      has_gpu: bool = False) -> dict:
    """
    Download all modules at once. Returns status dict.
    
    Args:
        target_dir: Directory to save files to
        version: Release tag (default: CURRENT_VERSION)
        has_gpu: Whether to include GPU-specific assets
        
    Returns:
        {"whisper": True/False, "bs_roformer": True/False, "cpp_annote": True/False}
    """
    results = {}
    for module in ["whisper", "bs_roformer", "cpp_annote"]:
        # Skip if already present
        if check_module_files(module, target_dir):
            logger.info(f"  {module}: already present")
            results[module] = True
            continue
        results[module] = download_module(module, target_dir, version, has_gpu)
    return results
