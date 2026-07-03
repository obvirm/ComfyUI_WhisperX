import json
import logging
import os
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
    "medium": {"name": "medium", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin", "size_mb": 1525, "multilingual": True, "description": "Medium multilingual (1.5GB)"},
    "medium.en": {"name": "medium.en", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin", "size_mb": 1525, "multilingual": False, "description": "Medium English-only (1.5GB)"},
    "large-v1": {"name": "large-v1", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin", "size_mb": 2950, "multilingual": True, "description": "Large V1 multilingual (2.95GB)"},
    "large-v2": {"name": "large-v2", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin", "size_mb": 2950, "multilingual": True, "description": "Large V2 multilingual (2.95GB)"},
    "large-v3": {"name": "large-v3", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin", "size_mb": 2950, "multilingual": True, "description": "Large V3 multilingual (2.95GB) - BEST QUALITY"},
    "large-v3-turbo": {"name": "large-v3-turbo", "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin", "size_mb": 1550, "multilingual": True, "description": "Large V3 Turbo multilingual (1.55GB) - FAST + QUALITY"},
}

MODEL_ORDER = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
CUSTOM_MODELS_CONFIG = {}

def load_custom_models(config_path: str):
    global CUSTOM_MODELS_CONFIG
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        custom = config.get("whisper_models", {})
        if isinstance(custom, dict):
            for key, val in custom.items():
                if isinstance(val, str) and val.startswith("http"):
                    CUSTOM_MODELS_CONFIG[key] = {"name": key, "url": val, "size_mb": 0, "multilingual": True, "description": f"Custom: {val.split('/')[-1]}"}
                elif isinstance(val, dict):
                    CUSTOM_MODELS_CONFIG[key] = val
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load custom models: {e}")

def get_all_models() -> Dict:
    models = {}
    for key in MODEL_ORDER:
        if key in GGML_MODELS:
            models[key] = dict(GGML_MODELS[key])
    models.update(CUSTOM_MODELS_CONFIG)
    return models

def get_model_keys() -> List[str]:
    keys = [k for k in MODEL_ORDER if k in GGML_MODELS]
    keys.extend(k for k in CUSTOM_MODELS_CONFIG if k not in keys)
    return keys

class ModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            try:
                import folder_paths
                self.cache_dir = os.path.join(folder_paths.models_dir, "whispercpp")
            except ImportError:
                self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.models = get_all_models()
        self._custom_loaded = False

    def ensure_custom_config(self, config_path: str):
        if not self._custom_loaded:
            load_custom_models(config_path)
            self.models = get_all_models()
            self._custom_loaded = True

    def get_model_path(self, model_key: str) -> Optional[str]:
        if model_key not in self.models:
            return None
        info = self.models[model_key]
        model_file = os.path.basename(urlparse(info["url"]).path)
        local_path = os.path.join(self.cache_dir, model_file)
        if os.path.isfile(local_path):
            return local_path
        return None

    def download_model(self, model_key: str, force: bool = False) -> Optional[str]:
        if model_key not in self.models:
            return None
        info = self.models[model_key]
        url = info["url"]
        model_file = os.path.basename(urlparse(url).path)
        local_path = os.path.join(self.cache_dir, model_file)
        if os.path.isfile(local_path) and not force:
            return local_path
        logger.info(f"Downloading {model_key}...")
        temp_path = local_path + ".incomplete"
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=model_key,
                      bar_format="\033[96m{l_bar}\033[0m\033[92m{bar:15}\033[0m\033[93m{r_bar}\033[0m") as pbar:
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8*1024*1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            os.rename(temp_path, local_path)
            return local_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def list_available(self) -> List[Dict]:
        results = []
        for key in get_model_keys():
            info = self.models.get(key, {})
            local_path = self.get_model_path(key)
            results.append({"key": key, "name": info.get("name", key), "description": info.get("description", ""), "size_mb": info.get("size_mb", 0), "downloaded": local_path is not None, "local_path": local_path})
        return results
