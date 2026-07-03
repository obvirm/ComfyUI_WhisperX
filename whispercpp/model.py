import json, logging, os
from typing import Dict, List, Optional
from urllib.parse import urlparse
import requests
from tqdm.auto import tqdm

logger = logging.getLogger("WhisperCPP")

GGML_MODELS = {
    "tiny": {"name": "tiny", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", "size_mb": 75, "multilingual": True, "description": "Tiny multilingual (75MB)"},
    "tiny.en": {"name": "tiny.en", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin", "size_mb": 75, "multilingual": False, "description": "Tiny English-only (75MB)"},
    "base": {"name": "base", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "size_mb": 142, "multilingual": True, "description": "Base multilingual (142MB)"},
    "base.en": {"name": "base.en", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin", "size_mb": 142, "multilingual": False, "description": "Base English-only (142MB)"},
    "small": {"name": "small", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin", "size_mb": 466, "multilingual": True, "description": "Small multilingual (466MB)"},
    "small.en": {"name": "small.en", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin", "size_mb": 466, "multilingual": False, "description": "Small English-only (466MB)"},
    "medium": {"name": "medium", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin", "size_mb": 1.5e3, "multilingual": True, "description": "Medium multilingual (1.5GB)"},
    "medium.en": {"name": "medium.en", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin", "size_mb": 1.5e3, "multilingual": False, "description": "Medium English-only (1.5GB)"},
    "large-v2": {"name": "large-v2", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin", "size_mb": 2.9e3, "multilingual": True, "description": "Large v2 multilingual (2.9GB)"},
    "large-v3": {"name": "large-v3", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin", "size_mb": 2.9e3, "multilingual": True, "description": "Large v3 multilingual (2.9GB)"},
    "large-v3-turbo": {"name": "large-v3-turbo", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin", "size_mb": 1.6e3, "multilingual": True, "description": "Large v3 Turbo multilingual (1.6GB)"},
}

class ModelManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.environ.get("WHISPERCPP_MODELS_DIR", "")
        if not self.cache_dir:
            from folder_paths import models_dir
            self.cache_dir = os.path.join(models_dir, "whispercpp")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.custom_models = {}

    def ensure_custom_config(self, config_path):
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    self.custom_models = json.load(f).get("whisper_models", {})
        except:
            pass

    def get_model_path(self, key):
        if key in GGML_MODELS:
            url = GGML_MODELS[key]["url"]
            fname = os.path.basename(urlparse(url).path)
            path = os.path.join(self.cache_dir, fname)
            if os.path.isfile(path):
                return path
        if key in self.custom_models:
            path = self.custom_models[key]
            if os.path.isfile(path):
                return path
        return None

    def download_model(self, key):
        if key not in GGML_MODELS:
            return None
        info = GGML_MODELS[key]
        fname = os.path.basename(urlparse(info["url"]).path)
        path = os.path.join(self.cache_dir, fname)
        if os.path.isfile(path):
            logger.info(f"Model cached at {path}")
            return path
        logger.info(f"Downloading {key} ({info['size_mb']/1000:.1f}GB)...")
        try:
            resp = requests.get(info["url"], stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=key) as pbar:
                for chunk in resp.iter_content(1024*1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
            logger.info(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(path):
                os.unlink(path)
            return None

def load_custom_models(config_path):
    mgr = ModelManager()
    mgr.ensure_custom_config(config_path)

def get_model_keys():
    return list(GGML_MODELS.keys())
