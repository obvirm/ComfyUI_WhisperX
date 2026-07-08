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
import tempfile
import urllib.request
import logging

logger = logging.getLogger("WhisperCPP")

# Simple file lock for race condition prevention
import threading
_file_locks = {}
_file_locks_lock = threading.Lock()

def _get_file_lock(filepath: str) -> threading.Lock:
    """Get or create a lock for a specific file."""
    with _file_locks_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

GITHUB_REPO = "obvirm/ComfyUI-WhisperCPP"

# Cached latest version from GitHub (1x query per session)
_latest_version_cache = None
_latest_version_checked = False

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
        "Windows": ["bs_roformer.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"],
        "Linux":   ["libbs_roformer.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"],
        "Darwin":  ["libbs_roformer.dylib", "libggml.dylib", "libggml-base.dylib", "libggml-cpu.dylib", "libggml-metal.dylib"],
    },
    "cpp_annote": {
        "Windows": ["cpp_annote.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"],
        "Linux":   ["libcpp_annote.so", "libonnxruntime.so", "libonnxruntime_providers_shared.so"],
        "Darwin":  ["libcpp_annote.dylib", "libonnxruntime.dylib", "libonnxruntime_providers_shared.dylib"],
    },
    # ONNX models (platform-independent)
    "cpp_annote_models": {
        "Windows": ["community1-segmentation.onnx", "community1-embedding.onnx"],
        "Linux":   ["community1-segmentation.onnx", "community1-embedding.onnx"],
        "Darwin":  ["community1-segmentation.onnx", "community1-embedding.onnx"],
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
    """Read version from VERSION env var, fallback pyproject.toml."""
    env_ver = os.environ.get("VERSION")
    if env_ver and env_ver.startswith("v"):
        return env_ver
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
        return "v2.1.5"


def _get_download_url(asset_name: str, version: str = None) -> str:
    """Build GitHub release download URL."""
    v = version or _get_version()
    return f"https://github.com/{GITHUB_REPO}/releases/download/{v}/{asset_name}"


def get_latest_version() -> str:
    """Query GitHub API for latest release version. Cached 1x per session."""
    global _latest_version_cache, _latest_version_checked
    if _latest_version_checked:
        return _latest_version_cache or _get_version()
    _latest_version_checked = True
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ComfyUI-WhisperCPP",
            "Accept": "application/vnd.github.v3+json"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            data = json.loads(resp.read())
            tag = data.get("tag_name", "")
            if tag:
                _latest_version_cache = tag
                logger.info(f"GitHub latest release: {tag}")
                return tag
    except Exception as e:
        logger.debug(f"GitHub API check failed: {e}")
    _latest_version_cache = _get_version()
    return _latest_version_cache


def check_version_and_update(target_dir: str, has_gpu: bool = False) -> bool:
    """
    Check if local DLLs match latest release. Re-download if outdated.
    Returns True if DLLs are up to date (or successfully updated).
    """
    local_ver = _get_version()
    remote_ver = get_latest_version()
    
    if local_ver == remote_ver:
        logger.info(f"DLLs version OK: {local_ver}")
        return True
    
    # Already updated this session
    if _latest_version_cache == remote_ver and remote_ver != _get_version():
        # We already downloaded this version, just sync the local check
        return True
    
    logger.warning(f"DLLs outdated! Local: {local_ver}, Latest: {remote_ver}")
    logger.info(f"Re-downloading DLLs from {remote_ver}...")
    
    # Download all modules with new version
    results = auto_download_all(target_dir, version=remote_ver, has_gpu=has_gpu)
    all_ok = all(results.values())
    
    if all_ok:
        logger.info(f"DLLs updated to {remote_ver} successfully!")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Failed to update: {failed}")
    
    return all_ok


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


_last_download_time = 0


def download_file(url: str, dest: str, timeout: int = 120, retries: int = 2,
                  expected_size: int = None, expected_sha256: str = None) -> bool:
    """Download a single file with safety checks. Returns True on success."""
    global _last_download_time
    # Rate limiting: max 5 downloads per minute
    import time
    now = time.time()
    if now - _last_download_time < 12:  # 12 seconds between downloads
        time.sleep(12 - (now - _last_download_time))
    
    # Prevent race condition: only one thread downloads to same file
    file_lock = _get_file_lock(dest)
    with file_lock:
        result = _download_file_internal(url, dest, timeout, retries, expected_size, expected_sha256)
        if result:
            _last_download_time = time.time()
        return result


def _download_file_internal(url: str, dest: str, timeout: int, retries: int,
                           expected_size: int, expected_sha256: str) -> bool:
    """Internal download function (must be called with file_lock held)."""
    # Safety: Check for symlink attacks
    if os.path.islink(dest):
        logger.warning(f"  Symlink detected: {dest} — removing")
        os.remove(dest)
    
    for attempt in range(retries + 1):
        try:
            # Skip if already exists and valid
            if os.path.isfile(dest) and not os.path.islink(dest) and os.path.getsize(dest) > 0:
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
            
            # Download to TEMP file first (prevents partial file corruption)
            dest_dir = os.path.dirname(dest) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
            os.close(fd)
            
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-WhisperCPP"})
                logger.info(f"  GET {url[:80]}...")
                try:
                    resp = urllib.request.urlopen(req, timeout=timeout)
                except urllib.error.HTTPError as e:
                    logger.warning(f"  HTTPError {e.code}: {e.reason}")
                    # Log response body for debugging
                    try:
                        body = e.read()[:200]
                        logger.warning(f"  Body: {body}")
                    except: pass
                    if e.code == 404:
                        logger.warning(f"  File not found: {os.path.basename(dest)} (retrying)")
                        import time
                        time.sleep(10)
                        os.remove(tmp_path)
                        continue
                    elif e.code == 403 or e.code == 429:
                        logger.warning(f"  Rate limited ({e.code}). Waiting...")
                        import time
                        time.sleep(30)
                        continue
                    else:
                        raise
                
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                
                # Verify file was downloaded
                if os.path.getsize(tmp_path) == 0:
                    raise Exception("Empty file")
                
                # Verify size if provided
                if expected_size and os.path.getsize(tmp_path) != expected_size:
                    logger.warning(f"  Size mismatch: expected {expected_size}, got {os.path.getsize(tmp_path)}")
                
                # Verify checksum
                if not _verify_checksum(tmp_path, expected_sha256):
                    raise Exception("Checksum mismatch")
                
                # Set executable permission on Linux/macOS
                if not IS_WIN:
                    os.chmod(tmp_path, 0o755)
                
                # ATOMIC MOVE: Rename temp → dest (prevents partial file on crash)
                if os.path.exists(dest):
                    os.remove(dest)
                os.rename(tmp_path, dest)
                return True
                
            except Exception as e:
                # Clean up temp file on failure
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                raise  # Re-raise to outer handler
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"  File not found: {os.path.basename(dest)} (CDN propagation?)")
                # CDN might not have propagated yet, retry with delay
                import time
                time.sleep(10)
                continue
            elif e.code == 403 or e.code == 429:
                logger.warning(f"  Rate limited ({e.code}). Waiting...")
                import time
                time.sleep(30)
                continue
            else:
                logger.warning(f"  HTTP error {e.code}: {e}")
        except Exception as e:
            logger.warning(f"  Download failed ({os.path.basename(dest)}): {e}")
        
        # Clean up partial destination file
        try:
            if os.path.isfile(dest) and not os.path.islink(dest):
                # Check if it's a valid file before removing
                if os.path.getsize(dest) == 0:
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
        module: "whisper", "bs_roformer", "cpp_annote", or "cpp_annote_models"
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

    # ONNX models go to cpp-annote/artifacts/
    if module == "cpp_annote_models":
        target_dir = os.path.join(target_dir, "cpp-annote", "artifacts")

    os.makedirs(target_dir, exist_ok=True)

    total = len(files)
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
        # Create symlinks for Linux/macOS versioned naming
        if IS_LINUX:
            _create_linux_symlinks(target_dir)
        elif IS_MAC:
            _create_mac_symlinks(target_dir)
        return True
    else:
        logger.warning(f"  {module}: core files missing ({success_count}/{total})")
        return False


def _create_linux_symlinks(target_dir: str):
    """Rename .so files to versioned names (dlopen uses real filename, not symlink)."""
    rename_map = {
        "libggml.so": "libggml.so.0",
        "libggml-base.so": "libggml-base.so.0",
        "libggml-cpu.so": "libggml-cpu.so.0",
        "libonnxruntime.so": "libonnxruntime.so.1",
        "libonnxruntime_providers_shared.so": "libonnxruntime_providers_shared.so.1",
    }
    for src, dst in rename_map.items():
        src_path = os.path.join(target_dir, src)
        dst_path = os.path.join(target_dir, dst)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            try:
                # Copy then remove original (rename doesn't work across filesystems)
                import shutil
                shutil.copy2(src_path, dst_path)
                logger.info(f"  Copied: {src} -> {dst}")
            except OSError as e:
                logger.warning(f"  Copy failed: {e}")


def _create_mac_symlinks(target_dir: str):
    """Copy .dylib files to versioned names (dlopen uses real filename)."""
    copy_map = {
        "libggml.dylib": "libggml.0.dylib",
        "libggml-base.dylib": "libggml-base.0.dylib",
        "libggml-cpu.dylib": "libggml-cpu.0.dylib",
        "libggml-metal.dylib": "libggml-metal.0.dylib",
        "libonnxruntime.dylib": "libonnxruntime.1.dylib",
        "libonnxruntime_providers_shared.dylib": "libonnxruntime_providers_shared.1.dylib",
    }
    for src, dst in copy_map.items():
        src_path = os.path.join(target_dir, src)
        dst_path = os.path.join(target_dir, dst)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                logger.info(f"  Copied: {src} -> {dst}")
            except OSError as e:
                logger.warning(f"  Copy failed: {e}")


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
