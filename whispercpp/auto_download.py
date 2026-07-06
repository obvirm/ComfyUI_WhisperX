"""
Auto-download DLLs/SOs from GitHub Releases.

All 3 modules (whisper, bs_roformer, cpp_annote) use this.
Downloads individual files based on platform + GPU detection.

Safety checks:
  - Disk space validation before download
  - Retry with exponential backoff on network errors
  - Skip 404 (file not found) immediately
  - SHA256 checksum verification (if available)
  - GitHub rate limit handling
"""
import hashlib
import os
import platform
import shutil
import urllib.request
import logging

logger = logging.getLogger("WhisperCPP")

GITHUB_REPO = "obvirm/ComfyUI-WhisperCPP"
CURRENT_VERSION = "v2.0.9"

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
# Note: Only include files that actually exist in the release!
GPU_ASSETS = {
    "whisper": {
        "Windows": [],  # No GPU DLLs in release (CPU only)
        "Linux":   ["libggml-opencl.so"],  # OpenCL available
        "Darwin":  [],  # Metal is built-in via libggml-metal.dylib (in ASSETS)
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


def _check_disk_space(dest: str, min_mb: int = 50) -> bool:
    """Check if there's enough disk space for download."""
    try:
        dir_path = os.path.dirname(dest) or "."
        usage = shutil.disk_usage(dir_path)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_mb:
            logger.warning(f"Low disk space: {free_mb:.0f}MB free (need {min_mb}MB)")
            return False
        return True
    except Exception:
        return True  # Can't check, assume OK


def _verify_checksum(filepath: str, expected_sha256: str = None) -> bool:
    """Verify file checksum if provided."""
    if not expected_sha256:
        return True  # No checksum to verify
    try:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256
    except Exception:
        return False


def download_file(url: str, dest: str, timeout: int = 120, retries: int = 2,
                  expected_size: int = None, expected_sha256: str = None) -> bool:
    """Download a single file with safety checks. Returns True on success."""
    for attempt in range(retries + 1):
        try:
            # Skip if already exists and valid
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                if _verify_checksum(dest, expected_sha256):
                    return True
                else:
                    logger.warning(f"  Checksum mismatch, re-downloading {os.path.basename(dest)}...")
                    os.remove(dest)
            
            # Check disk space
            if not _check_disk_space(dest, min_mb=50):
                return False
            
            if attempt > 0:
                logger.info(f"  Retry {attempt}/{retries} for {os.path.basename(dest)}...")
            else:
                logger.info(f"  Downloading {os.path.basename(dest)}...")
            
            # Download with timeout
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-WhisperCPP"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                # Check for rate limiting (403/429)
                if response.status == 403:
                    logger.warning("  GitHub rate limit hit. Wait and retry.")
                    import time
                    time.sleep(60)
                    continue
                
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            
            # Verify file was downloaded
            if os.path.getsize(dest) == 0:
                os.remove(dest)
                raise Exception("Empty file")
            
            # Verify size if provided
            if expected_size and os.path.getsize(dest) != expected_size:
                logger.warning(f"  Size mismatch: expected {expected_size}, got {os.path.getsize(dest)}")
                # Don't fail, just warn (size might differ slightly)
            
            # Verify checksum
            if not _verify_checksum(dest, expected_sha256):
                os.remove(dest)
                raise Exception("Checksum mismatch")
            
            # Set executable permission on Linux/macOS
            if not IS_WIN:
                os.chmod(dest, 0o755)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"  File not found: {os.path.basename(dest)}")
                return False
            elif e.code == 403 or e.code == 429:
                logger.warning(f"  Rate limited ({e.code}). Waiting...")
                import time
                time.sleep(30)
                continue
            else:
                logger.warning(f"  HTTP error {e.code}: {e}")
        except Exception as e:
            logger.warning(f"  Download failed ({os.path.basename(dest)}): {e}")
        
        # Clean up partial download
        try:
            if os.path.isfile(dest):
                os.remove(dest)
        except OSError:
            pass
        
        # Exponential backoff
        if attempt < retries:
            import time
            time.sleep(2 ** attempt)
    
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

    # Core files (non-GPU) must all succeed
    core_files = ASSETS[module][system]
    core_ok = all(
        os.path.isfile(os.path.join(target_dir, f))
        for f in core_files
    )
    
    if core_ok:
        logger.info(f"  {module}: {success_count}/{total} files downloaded")
        return True
    else:
        logger.warning(f"  {module}: core files missing ({success_count}/{total})")
        return False


def check_module_files(module: str, target_dir: str, has_gpu: bool = False) -> bool:
    """Check if all required files for a module exist."""
    system = platform.system()
    if system not in ASSETS.get(module, {}):
        return False

    # Check core files
    for fname in ASSETS[module][system]:
        if not os.path.isfile(os.path.join(target_dir, fname)):
            return False
    
    # Check GPU files if requested
    if has_gpu:
        gpu_files = GPU_ASSETS.get(module, {}).get(system, [])
        for fname in gpu_files:
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
