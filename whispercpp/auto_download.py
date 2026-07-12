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
import ssl
import json
import time

def _ssl_context():
    """SSL context using certifi CA bundle (fixes macOS cert issues)."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return ctx

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

# Local fingerprint manifest (SHA256 + timestamp) of downloaded DLLs
# Stored next to the DLLs so we can detect stale/corrupt files across restarts.
MANIFEST_FILE = "dll_state.json"


def _load_manifest(target_dir: str) -> dict:
    """Load local fingerprint manifest (dll_state.json)."""
    try:
        p = os.path.join(target_dir, MANIFEST_FILE)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_manifest(target_dir: str, manifest: dict):
    """Persist local fingerprint manifest (dll_state.json)."""
    try:
        p = os.path.join(target_dir, MANIFEST_FILE)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug(f"Failed to save manifest: {e}")


def refresh_manifest(target_dir: str):
    """Recompute SHA256+size for every file currently recorded in the manifest.

    Used AFTER we intentionally mutate a downloaded file (e.g. macOS
    `install_name_tool` patching of dylibs rewrites the bytes -> SHA256 changes).
    Without this, the next launch would see a spurious 'fingerprint mismatch' and
    re-download a perfectly good (and already-patched) file.
    """
    try:
        manifest = _load_manifest(target_dir)
        if "files" not in manifest or not manifest["files"]:
            return
        changed = False
        for fname in list(manifest["files"].keys()):
            fpath = os.path.join(target_dir, fname)
            if os.path.isfile(fpath):
                new_sha = _file_sha256(fpath)
                new_size = os.path.getsize(fpath)
                rec = manifest["files"][fname]
                if rec.get("sha256") != new_sha or rec.get("size") != new_size:
                    rec["sha256"] = new_sha
                    rec["size"] = new_size
                    rec["updated"] = int(time.time())
                    changed = True
            else:
                del manifest["files"][fname]
                changed = True
        if changed:
            _save_manifest(target_dir, manifest)
    except Exception as e:
        logger.debug(f"refresh_manifest skipped: {e}")


def _apply_pending_files(target_dir: str):
    """Swap any `*.pending` replacement files into place (best-effort).

    A previous launch may have stashed a freshly downloaded DLL as `name.pending`
    because the live file was locked (already loaded into the running process,
    e.g. onnxruntime.dll which we pre-load at import time). On a fresh launch the
    old mapping is gone, so we can now atomically swap the new bytes in. Any
    leftover `*.old` (rename-aside from _replace_existing_file) is also removed.
    """
    try:
        for fn in os.listdir(target_dir):
            full = os.path.join(target_dir, fn)
            if fn.endswith(".pending"):
                dst = full[:-len(".pending")]
                try:
                    os.replace(full, dst)  # atomic on POSIX; on Win dst must be unlocked
                    logger.info(f"  Applied pending update: {fn[:-len('.pending')]}")
                except OSError:
                    pass  # still locked, try again next launch
            elif fn.endswith(".old"):
                try:
                    os.remove(full)
                except OSError:
                    pass
        # Recompute fingerprints so subsequent size/sha checks see the new bytes.
        refresh_manifest(target_dir)
    except Exception:
        pass


def _replace_existing_file(dest: str) -> bool:
    """Remove (or, on Windows, rename-aside) an existing destination file so a
    fresh copy can be written.

    On Windows a DLL that is already loaded into this process is locked
    (PermissionError / WinError 5) and cannot be deleted. We rename it aside
    (file.dll -> file.dll.old); the old mapping stays valid until process exit,
    and the new content is written under the original name for the next launch.
    Returns True if the original path is now free to write, False otherwise.
    """
    try:
        if os.path.exists(dest):
            os.remove(dest)
        return True
    except PermissionError:
        try:
            old = dest + ".old"
            if os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            os.rename(dest, old)
            return True
        except OSError:
            return False
    except OSError:
        return False


def _file_sha256(filepath: str) -> str:
    """Compute SHA256 of a file."""
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _update_manifest_for_file(target_dir: str, fname: str, manifest: dict):
    """Record/update fingerprint for one successfully downloaded file."""
    fpath = os.path.join(target_dir, fname)
    if not os.path.isfile(fpath):
        return
    if "files" not in manifest:
        manifest["files"] = {}
    manifest["files"][fname] = {
        "sha256": _file_sha256(fpath),
        "size": os.path.getsize(fpath),
        "updated": int(time.time()),
    }


# All modules that ship prebuilt DLLs
MODULES = ["whisper", "bs_roformer", "cpp_annote"]

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
        # Windows/Linux: FULL build, backends (CUDA/Vulkan/OpenCL) di-build sbg MODULE
        # terpisah & di-dlopen malas oleh ggml (GGML_BACKEND_DL=ON). libggml.so TIDAK
        # NEEDED backend .so — di host tanpa GPU, backend CUDA gagal load silently & CPU
        # dipakai. Di host GPU, backend aktif otomatis.
        # macOS: GGML_BACKEND_DL=OFF (ggml DL loader hardcode .so tapi CMake MODULE di
        # macOS jadi .dylib -> 0 CPU device -> crash). Backends (CPU/Metal/BLAS) di-static
        # link ke libggml.dylib & auto-register saat load. Jadi macOS TIDAK butuh file
        # backend terpisah.
        "Windows": ["whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"],
        "Linux":   ["libwhisper.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                    "libggml-cuda.so", "libggml-opencl.so"],
        # macOS: GGML_BACKEND_DL=OFF + BUILD_SHARED_LIBS=ON -> backends (cpu/metal/blas)
        # di-build sbg SHARED .dylib TERPISAH & di-LINK ke libggml.dylib (libggml.dylib
        # NEEDED libggml-cpu.dylib dkk). Mereka auto-register saat libggml.dylib load.
        # Jadi HARUS di-shipped. (ggml DL loader di macOS hardcoded .so, jadi
        # GGML_BACKEND_DL=ON tidak bisa dipakai di macOS.)
        "Darwin":  ["libwhisper.dylib", "libggml.dylib", "libggml-base.dylib",
                    "libggml-cpu.dylib", "libggml-blas.dylib", "libggml-metal.dylib"],
    },
    "bs_roformer": {
        "Windows": ["bs_roformer.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"],
        "Linux":   ["libbs_roformer.so", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                    "libggml-cuda.so", "libggml-opencl.so"],
        "Darwin":  ["libbs_roformer.dylib", "libggml.dylib", "libggml-base.dylib",
                    "libggml-cpu.dylib", "libggml-blas.dylib", "libggml-metal.dylib"],
    },
    "cpp_annote": {
        "Windows": ["cpp_annote.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"],
        "Linux":   ["libcpp_annote.so", "libonnxruntime.so", "libonnxruntime_providers_shared.so"],
        "Darwin":  ["libcpp_annote.dylib", "libonnxruntime.dylib"],
    },
    # ONNX models (platform-independent)
    "cpp_annote_models": {
        "Windows": ["community1-segmentation.onnx", "community1-embedding.onnx"],
        "Linux":   ["community1-segmentation.onnx", "community1-embedding.onnx"],
        "Darwin":  ["community1-segmentation.onnx", "community1-embedding.onnx"],
    },
}

# GPU-specific assets — untuk FULL build (semua backend di-bake ke 1 DLL):
# whisper.dll / bs_roformer.dll sudah berisi CUDA+Vulkan+OpenCL+OpenVINO statically,
# jadi gak ada plugin terpisah yang perlu di-download. GPU_ASSETS kosong.
# cpp-annote: GPU via ONNX Runtime — pakai build GPU (CUDA/DirectML/OpenVINO),
# file onnxruntime*.dll/.so sudah di-ASSETS inti (sama nama, beda isi).
GPU_ASSETS = {
    "whisper":    { "Windows": [], "Linux": [], "Darwin": [] },
    "bs_roformer":{ "Windows": [], "Linux": [], "Darwin": [] },
    "cpp_annote": { "Windows": [], "Linux": [], "Darwin": [] },
}

# OPTIONAL backend/provider libs — shipped ONLY when the build actually emits them.
# With the dynamic-load FULL build (GGML_BACKEND_DL=ON) whisper emits separate
# backend .so/.dll (libggml-cuda, libggml-opencl, libggml-vulkan) that are
# dlopen'd lazily; they're optional (skip_if_missing) so a GPU-less host still
# works (CPU backend). cpp-annote's GPU ORT packages additionally ship CUDA/
# TensorRT provider libs; the base libonnxruntime (in ASSETS) loads them on demand.
OPTIONAL_ASSETS = {
    # NOTE: Vulkan backend (ggml-vulkan.*) is intentionally NOT listed — our CI build
    # only ships CUDA + OpenCL backends, so a ggml-vulkan.* release asset never exists.
    # Listing it would produce a harmless-but-noisy 404 "optional asset" warning on every
    # install. Users who build Vulkan locally can drop the file in manually.
    "whisper": {
        "Windows": ["ggml-cuda.dll", "ggml-opencl.dll"],
        "Linux":   ["libggml-cuda.so", "libggml-opencl.so"],
        "Darwin":  [],
    },
    "bs_roformer": {
        "Windows": ["ggml-cuda.dll", "ggml-opencl.dll"],
        "Linux":   ["libggml-cuda.so", "libggml-opencl.so"],
        "Darwin":  [],
    },
    "cpp_annote": {
        "Windows": ["onnxruntime_providers_cuda.dll", "onnxruntime_providers_tensorrt.dll"],
        "Linux":   ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_tensorrt.so"],
        "Darwin":  [],
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
    v = version or get_latest_version()
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
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
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


def get_release_asset_sizes(version: str) -> dict:
    """Query GitHub API for release asset sizes. Returns {asset_name: size_in_bytes}."""
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{version}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ComfyUI-WhisperCPP",
            "Accept": "application/vnd.github.v3+json"
        })
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
            import json
            data = json.loads(resp.read())
            return {a["name"]: a["size"] for a in data.get("assets", [])}
    except Exception as e:
        logger.debug(f"GitHub asset size check failed: {e}")
        return {}


def check_version_and_update(target_dir: str, has_gpu: bool = False) -> bool:
    """
    Check if local DLLs are present and match latest release.
    Re-download if outdated OR any file is missing (e.g. user deleted them).
    Returns True if DLLs are up to date (or successfully updated).
    """
    try:
        remote_ver = get_latest_version()
    except Exception:
        remote_ver = _get_version()

    local_ver = _get_version()

    # On a fresh launch, swap in any `*.pending` updates stashed by a previous
    # run whose live DLL was locked (e.g. onnxruntime.dll pre-loaded at import).
    _apply_pending_files(target_dir)

    # Check if ALL module files are present on disk (and fingerprint-valid)
    manifest = _load_manifest(target_dir)
    all_present = all(
        check_module_files(m, target_dir, has_gpu=has_gpu, manifest=manifest)
        for m in MODULES
    )

    # Fetch release asset sizes to detect stale/corrupt DLLs (same version, bad content)
    asset_sizes = {}
    try:
        asset_sizes = get_release_asset_sizes(remote_ver)
    except Exception:
        pass

    if all_present and local_ver == remote_ver:
        # Also verify file sizes match release (catches stale DLLs with same version string)
        if asset_sizes:
            size_ok = all(
                check_module_files(m, target_dir, has_gpu=has_gpu,
                                   verify_size=True, asset_sizes=asset_sizes,
                                   manifest=manifest)
                for m in MODULES
            )
            if size_ok:
                # Ensure fingerprint manifest exists so future restarts can
                # detect silent corruption (e.g. manual DLL swap).
                _ensure_manifest(target_dir, manifest)
                logger.info(f"DLLs version OK: {local_ver}")
                return True
            logger.warning("DLLs present but fingerprint/size mismatch (stale/corrupt) — re-downloading from latest release...")
        else:
            _ensure_manifest(target_dir, manifest)
            logger.info(f"DLLs version OK: {local_ver}")
            return True

    if not all_present:
        logger.warning("Some DLLs missing on disk — re-downloading from latest release...")
    elif local_ver != remote_ver:
        logger.warning(f"DLLs outdated! Local: {local_ver}, Latest: {remote_ver}")
        logger.info(f"Re-downloading DLLs from {remote_ver}...")

    # Download all modules with new version
    results = auto_download_all(target_dir, version=remote_ver, has_gpu=has_gpu,
                                verify_size=bool(asset_sizes), asset_sizes=asset_sizes,
                                manifest=manifest)
    all_ok = all(results.values())

    if all_ok:
        logger.info(f"DLLs updated to {remote_ver} successfully!")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Failed to update: {failed}")

    return all_ok


def _ensure_manifest(target_dir: str, manifest: dict):
    """Make sure the manifest records fingerprints for all currently-present
    module files. Saves only if something changed."""
    try:
        changed = False
        if "files" not in manifest:
            manifest["files"] = {}
        for m in MODULES:
            system = platform.system()
            if system not in ASSETS.get(m, {}):
                continue
            for fname in ASSETS[m][system]:
                fpath = os.path.join(target_dir, fname)
                if os.path.isfile(fpath) and fname not in manifest["files"]:
                    _update_manifest_for_file(target_dir, fname, manifest)
                    changed = True
        if changed:
            _save_manifest(target_dir, manifest)
    except Exception as e:
        logger.debug(f"_ensure_manifest skipped: {e}")


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
                  expected_size: int = None, expected_sha256: str = None,
                  manifest: dict = None, skip_if_missing: bool = False) -> bool:
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
        result = _download_file_internal(url, dest, timeout, retries, expected_size, expected_sha256, skip_if_missing)
        if result and manifest is not None:
            # Record fingerprint for this successfully downloaded file
            target_dir = os.path.dirname(dest) or "."
            _update_manifest_for_file(target_dir, os.path.basename(dest), manifest)
            _save_manifest(target_dir, manifest)
        if result:
            _last_download_time = time.time()
        return result


def _download_file_internal(url: str, dest: str, timeout: int, retries: int,
                           expected_size: int, expected_sha256: str,
                           skip_if_missing: bool = False) -> bool:
    """Internal download function (must be called with file_lock held)."""
    # Safety: Check for symlink attacks
    if os.path.islink(dest):
        logger.warning(f"  Symlink detected: {dest} — removing")
        os.remove(dest)
    
    for attempt in range(retries + 1):
        try:
            # Skip if already exists and valid
            if os.path.isfile(dest) and not os.path.islink(dest) and os.path.getsize(dest) > 0:
                # Re-download if size mismatch (stale/corrupt same-version file)
                if expected_size and os.path.getsize(dest) != expected_size:
                    logger.warning(f"  Size mismatch on disk ({os.path.basename(dest)}), re-downloading...")
                    # On Windows a loaded DLL can't be deleted -> rename-aside so the
                    # running process keeps its mapping; the new content lands under
                    # the canonical name via the atomic-move step below.
                    _replace_existing_file(dest)
                elif _verify_checksum(dest, expected_sha256):
                    return True
                else:
                    logger.warning(f"  Checksum mismatch, re-downloading {os.path.basename(dest)}...")
                    _replace_existing_file(dest)
            
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
                    resp = urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())
                except urllib.error.HTTPError as e:
                    logger.warning(f"  HTTPError {e.code}: {e.reason}")
                    # Log response body for debugging
                    try:
                        body = e.read()[:200]
                        logger.warning(f"  Body: {body}")
                    except: pass
                    if e.code == 404:
                        if skip_if_missing:
                            # OPTIONAL asset absent in this release — not fatal.
                            logger.info(f"  Optional asset not in release, skipping: {os.path.basename(dest)}")
                            os.remove(tmp_path)
                            return True
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
                
                # ATOMIC MOVE: Rename temp → dest (prevents partial file on crash).
                # If dest is locked (e.g. a DLL already loaded into this process on
                # Windows -> PermissionError / WinError 5), rename the old file aside
                # (.old) so the in-memory mapping keeps working, then place the new
                # content under the canonical name for the NEXT launch. If even the
                # rename-aside fails, stash the new bytes as dest + ".pending" and let
                # the next launch swap it in.
                if os.path.exists(dest):
                    if not _replace_existing_file(dest):
                        try:
                            os.replace(tmp_path, dest + ".pending")
                            logger.warning(f"  Locked file, stashed update as {os.path.basename(dest)}.pending (used next launch)")
                        except OSError:
                            pass
                        return True
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
                if skip_if_missing:
                    logger.info(f"  Optional asset not in release, skipping: {os.path.basename(dest)}")
                    return True
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
                    has_gpu: bool = False, asset_sizes: dict = None,
                    manifest: dict = None) -> bool:
    """
    Download all files for a module (whisper/bs_roformer/cpp_annote).
    
    Args:
        module: "whisper", "bs_roformer", "cpp_annote", or "cpp_annote_models"
        target_dir: Directory to save files to
        version: Release tag (default: CURRENT_VERSION)
        has_gpu: Whether to include GPU-specific assets
        asset_sizes: {asset_name: size} from release (for size verification)
        
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
        # OPTIONAL backend libs (CUDA/OpenCL/Vulkan/ORT providers) — only ship
        # when present in this release. We resolve the actual asset list via the
        # GitHub release listing so a missing one (different toolchain) isn't fatal.
        for opt in OPTIONAL_ASSETS.get(module, {}).get(system, []):
            files.append(opt)
        # De-dupe while preserving order
        files = list(dict.fromkeys(files))

    # ONNX models go to cpp-annote/artifacts/
    if module == "cpp_annote_models":
        target_dir = os.path.join(target_dir, "cpp-annote", "artifacts")

    os.makedirs(target_dir, exist_ok=True)

    total = len(files)
    success_count = 0
    for fname in files:
        url = _get_download_url(fname, version)
        dest = os.path.join(target_dir, fname)
        expected_size = asset_sizes.get(fname) if asset_sizes else None
        # OPTIONAL assets: skip silently if they don't exist in this release.
        is_optional = fname in OPTIONAL_ASSETS.get(module, {}).get(system, [])
        if download_file(url, dest, expected_size=expected_size, manifest=manifest,
                         skip_if_missing=is_optional):
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
        "libggml-cuda.so": "libggml-cuda.so.0",
        "libggml-opencl.so": "libggml-opencl.so.0",
        "libggml-vulkan.so": "libggml-vulkan.so.0",
        "libwhisper.so": "libwhisper.so.1",
        "libbs_roformer.so": "libbs_roformer.so.1",
        "libonnxruntime.so": "libonnxruntime.so.1",
        "libonnxruntime_providers_shared.so": "libonnxruntime_providers_shared.so.1",
    }
    for src, dst in rename_map.items():
        src_path = os.path.join(target_dir, src)
        dst_path = os.path.join(target_dir, dst)
        # Also: if the versioned file (e.g. libggml.so.0) exists in the release
        # but the unversioned (libggml.so) does NOT, create the unversioned too,
        # since some loaders/dlopen calls reference the bare name.
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                logger.info(f"  Copied: {src} -> {dst}")
            except OSError as e:
                logger.warning(f"  Copy failed: {e}")
        elif os.path.isfile(dst_path) and not os.path.exists(src_path):
            try:
                import shutil
                shutil.copy2(dst_path, src_path)
                logger.info(f"  Copied: {dst} -> {src}")
            except OSError as e:
                logger.warning(f"  Copy failed: {e}")


def _create_mac_symlinks(target_dir: str):
    """Copy .dylib files to versioned names (dlopen uses real filename)."""
    copy_map = {
        "libggml.dylib": "libggml.0.dylib",
        "libggml-base.dylib": "libggml-base.0.dylib",
        "libggml-cpu.dylib": "libggml-cpu.0.dylib",
        "libggml-blas.dylib": "libggml-blas.0.dylib",
        "libggml-metal.dylib": "libggml-metal.0.dylib",
        "libwhisper.dylib": "libwhisper.1.dylib",
        "libbs_roformer.dylib": "libbs_roformer.1.dylib",
        "libonnxruntime.dylib": "libonnxruntime.1.dylib",
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


def check_module_files(module: str, target_dir: str, has_gpu: bool = False,
                       verify_size: bool = False, asset_sizes: dict = None,
                       manifest: dict = None) -> bool:
    """Check if all required files for a module exist.

    If verify_size is True and asset_sizes is provided, also confirms each file's
    byte size matches the release asset (catches stale/corrupt DLLs that are
    present but have the same version string).

    If manifest is provided and contains a fingerprint for a file, also confirms
    the file's current SHA256 matches the recorded fingerprint (catches
    manual replacement / silent corruption across restarts).
    """
    system = platform.system()
    if system not in ASSETS.get(module, {}):
        return False

    # Check core files
    for fname in ASSETS[module][system]:
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            return False
        if verify_size and asset_sizes:
            expected = asset_sizes.get(fname)
            if expected is not None and os.path.getsize(fpath) != expected:
                logger.warning(f"  Stale/corrupt file: {fname} "
                               f"(size {os.path.getsize(fpath)} != release {expected})")
                return False
        if manifest and manifest.get("files", {}).get(fname):
            rec = manifest["files"][fname]
            # If manifest recorded a sha256, verify it (unless size already mismatched)
            if rec.get("sha256"):
                cur = _file_sha256(fpath)
                if cur and cur != rec["sha256"]:
                    logger.warning(f"  Fingerprint mismatch: {fname} "
                                   f"(local sha256 changed since last download)")
                    return False

    # Check GPU files if requested
    if has_gpu:
        gpu_files = GPU_ASSETS.get(module, {}).get(system, [])
        for fname in gpu_files:
            fpath = os.path.join(target_dir, fname)
            if not os.path.isfile(fpath):
                return False
            if verify_size and asset_sizes:
                expected = asset_sizes.get(fname)
                if expected is not None and os.path.getsize(fpath) != expected:
                    return False
            if manifest and manifest.get("files", {}).get(fname):
                rec = manifest["files"][fname]
                if rec.get("sha256"):
                    cur = _file_sha256(fpath)
                    if cur and cur != rec["sha256"]:
                        return False

    return True


def auto_download_all(target_dir: str, version: str = None,
                      has_gpu: bool = False, verify_size: bool = False,
                      asset_sizes: dict = None, manifest: dict = None) -> dict:
    """
    Download all modules at once. Returns status dict.
    
    Args:
        target_dir: Directory to save files to
        version: Release tag (default: CURRENT_VERSION)
        has_gpu: Whether to include GPU-specific assets
        verify_size: If True, re-download modules whose files have wrong size
        asset_sizes: {asset_name: size} from release (for verify_size)
        
    Returns:
        {"whisper": True/False, "bs_roformer": True/False, "cpp_annote": True/False}
    """
    results = {}
    for module in ["whisper", "bs_roformer", "cpp_annote"]:
        # Skip if already present (and size/fingerprint-valid when verify is on)
        if check_module_files(module, target_dir, has_gpu=has_gpu,
                              verify_size=verify_size, asset_sizes=asset_sizes,
                              manifest=manifest):
            logger.info(f"  {module}: already present")
            results[module] = True
            continue
        results[module] = download_module(module, target_dir, version, has_gpu,
                                           asset_sizes=asset_sizes, manifest=manifest)
    # ALWAYS ensure versioned symlinks/copies exist (even on cached runs where
    # download_module was skipped) — dlopen needs libggml.so.0 etc present.
    if IS_LINUX:
        _create_linux_symlinks(target_dir)
    elif IS_MAC:
        _create_mac_symlinks(target_dir)
    return results
