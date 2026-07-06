"""
ComfyUI-WhisperCPP - whisper.cpp with full params, alignment, diarization.

Auto-download: DLLs + ONNX models from GitHub Releases
"""

import os
import sys

# ── Version Check ──────────────────────────────────────────────────────────
# Force users on old versions to update to latest

def _check_version():
    """Check if current version is up-to-date. Shows warning if outdated."""
    try:
        import tomllib
        pyproject = os.path.join(os.path.dirname(__file__), "pyproject.toml")
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        current_version = data["project"]["version"]
    except Exception:
        return True  # Skip check on error
    
    # Try to get latest version from GitHub (async, don't block)
    try:
        import urllib.request
        import json
        
        def get_latest_version():
            """Get latest release tag from GitHub."""
            url = "https://api.github.com/repos/obvirm/ComfyUI-WhisperCPP/releases/latest"
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-WhisperCPP"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
                return data.get("tag_name", "").lstrip("v")
        
        latest_version = get_latest_version()
        
        if current_version != latest_version:
            print(f"\n{'='*60}")
            print(f"⚠️  ComfyUI-WhisperCPP v{current_version} is OUTDATED!")
            print(f"")
            print(f"Please update to v{latest_version}:")
            print(f"  cd ComfyUI/custom_nodes/ComfyUI-WhisperCPP")
            print(f"  git pull")
            print(f"  pip install -r requirements.txt")
            print(f"")
            print(f"Current:  v{current_version}")
            print(f"Latest:   v{latest_version}")
            print(f"")
            print(f"❌ DO NOT open GitHub issues if you're on an outdated version!")
            print(f"❌ Update first, then report if the problem persists.")
            print(f"{'='*60}\n")
            return False
        
        return True
    except Exception:
        # Can't reach GitHub — skip check (don't block user)
        return True

# Run version check
_check_version()

# ── Module Imports ─────────────────────────────────────────────────────────
from .whispercpp_node import WhisperCPPNode

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "WhisperCPPNode": WhisperCPPNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperCPPNode": "WhisperCPP Transcription",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
