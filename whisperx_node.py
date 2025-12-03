import torch
import os
import folder_paths
import comfy.utils
import torchaudio
import numpy as np
import gc
from typing import Optional, Tuple
import json
import io
import requests
import shutil
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_url, list_repo_files
import sys
import re
import time
import importlib.metadata
import warnings
import logging
from contextlib import contextmanager

from whisperx.alignment import (
    align,
    load_align_model,
    DEFAULT_ALIGN_MODELS_TORCH,
    DEFAULT_ALIGN_MODELS_HF,
)
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer, ResultWriter
from whisperx.SubtitlesProcessor import SubtitlesProcessor

os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "copy"


# --- Colored Logger ---
class ColoredLogger:
    def __init__(self, name="WhisperXNode"):
        self.name = name
        self.COLORS = {
            "reset": "\033[0m",
            "info": "\033[94m",  # Blue
            "success": "\033[92m",  # Green
            "warning": "\033[93m",  # Yellow
            "error": "\033[91m",  # Red
        }

    def _log(self, message, level):
        color = self.COLORS.get(level, self.COLORS["reset"])
        print(f"{color}[{self.name}] {message}{self.COLORS['reset']}")

    def info(self, message):
        self._log(message, "info")

    def success(self, message):
        self._log(message, "success")

    def warning(self, message):
        self._log(message, "warning")

    def error(self, message):
        self._log(message, "error")


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "WARNING": "\033[93m",  # Yellow
        "INFO": "\033[94m",  # Blue
        "ERROR": "\033[91m",  # Red
        "reset": "\033[0m",
    }

    def format(self, record):
        if record.levelname == "INFO" and record.name in [
            "whisperx.asr",
            "whisperx.vads.pyannote",
            "whisperx.vads.silero",
        ]:
            if (
                "No language specified" in record.msg
                or "Performing voice activity detection" in record.msg
                or "Detected language:" in record.msg
            ):
                record.levelname = "WARNING"

        color = self.COLORS.get(record.levelname, self.COLORS["reset"])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['reset']}"


# --- Custom Log Filter ---
# (Removed)

# --- Model Mappings & Configuration Loading ---
logger = ColoredLogger()

# Filter out pyannote's ReproducibilityWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TensorFloat-32 (TF32) has been disabled",
    module="pyannote.audio.utils.reproducibility",
)


DEFAULT_WHISPER_MODELS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v1": "Systran/faster-whisper-large-v1",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium-en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small-en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base-en",
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny-en",
}
WHISPER_MODELS = DEFAULT_WHISPER_MODELS.copy()
ALIGN_MODELS_HF = {**DEFAULT_ALIGN_MODELS_HF}
align_models_list = sorted(
    list(DEFAULT_ALIGN_MODELS_TORCH.values()) + list(DEFAULT_ALIGN_MODELS_HF.values())
)
diarization_models_list = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-2.1",
]
CUSTOM_ALIGN_MODELS_MAP = {}

try:
    config_path = os.path.join(os.path.dirname(__file__), "whisperx.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            WHISPER_MODELS = config.get("whisper_models", DEFAULT_WHISPER_MODELS)
            diarization_models_list = config.get(
                "diarization_models", diarization_models_list
            )
            custom_models = config.get("custom_align_models", {})
            if isinstance(custom_models, dict):
                CUSTOM_ALIGN_MODELS_MAP = custom_models
                for lang_code, model_name in custom_models.items():
                    if model_name not in align_models_list:
                        align_models_list.append(model_name)
                align_models_list.sort()
except (IOError, json.JSONDecodeError) as e:
    logger.warning(
        f"Could not load or parse whisperx.json, falling back to default models. Error: {e}"
    )


# --- Progress Bar Handling ---
@contextmanager
def _capture_progress(pbar: tqdm):
    """Context manager to capture stdout/stderr and update a tqdm progress bar, avoiding recursion."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class TqdmRedirect(io.StringIO):
        def __init__(self, pbar_instance, original_stream):
            super().__init__()
            self.pbar = pbar_instance
            self.original_stream = original_stream
            self.progress_regex = re.compile(r"Progress:\s*(\d+\.\d+)%")
            self.tqdm_regex = re.compile(r"^\s*(\d+)%")

        def write(self, s):
            tqdm_match = self.tqdm_regex.search(s)
            if tqdm_match:
                progress = int(tqdm_match.group(1))
                update_value = progress - self.pbar.n
                if update_value > 0:
                    self.pbar.update(update_value)
                return

            progress_match = self.progress_regex.search(s)
            if progress_match:
                progress = float(progress_match.group(1))
                update_value = int(progress) - self.pbar.n
                if update_value > 0:
                    self.pbar.update(update_value)
                return

            # Filter out torchaudio's direct download message to avoid interrupting tqdm bar
            torchaudio_download_regex = re.compile(
                r"^Downloading: \"https:\/\/download\.pytorch\.org\/torchaudio\/models\/.*\" to .*"
            )
            if torchaudio_download_regex.search(s):
                return  # Suppress this line

            lines = s.splitlines()
            for line in lines:
                line_stripped = line.strip()
                if line_stripped:
                    tqdm.write(line, file=self.original_stream)

        def flush(self):
            self.original_stream.flush()

    sys.stdout = TqdmRedirect(pbar, original_stdout)
    sys.stderr = TqdmRedirect(pbar, original_stderr)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class WhisperXNode:
    PBAR_FORMAT = "\033[96m{l_bar}\033[0m\033[92m{bar:15}\033[0m\033[93m{r_bar}\033[0m"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (list(WHISPER_MODELS.keys()),),
                "language": (
                    ["None"]
                    + sorted(LANGUAGES.keys())
                    + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                    {"default": "None"},
                ),
                "task": (["transcribe", "translate"],),
                "batch_size": ("INT", {"default": 8, "min": 1}),
                "compute_type": (["float16", "float32", "int8"],),
                "device": (["cuda", "cpu"],),
            },
            "optional": {
                "show_advance_settings": ("BOOLEAN", {"default": False}),
                "align_model_optional": (
                    ["auto"] + align_models_list,
                    {"default": "auto"},
                ),
                "diarize_optional": ("BOOLEAN", {"default": False}),
                "diarize_model_optional": (
                    diarization_models_list,
                    {"default": diarization_models_list[0]},
                ),
                "min_speakers_optional": ("INT", {"default": -1, "min": -1}),
                "max_speakers_optional": ("INT", {"default": -1, "min": -1}),
                "speaker_embeddings_optional": ("BOOLEAN", {"default": False}),
                "hf_token_optional": ("STRING", {"default": ""}),
                "filename_prefix_optional": ("STRING", {"default": "whisperx/output"}),
                "output_format_optional": (
                    ["all", "srt", "vtt", "txt", "tsv", "json", "aud"],
                ),
                "vad_method_optional": (
                    ["pyannote", "silero"],
                    {"default": "pyannote"},
                ),
                "vad_onset_optional": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "vad_offset_optional": (
                    "FLOAT",
                    {"default": 0.363, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "chunk_size_optional": ("INT", {"default": 30, "min": 1}),
                "temperature_optional": ("STRING", {"default": "0.0"}),
                "temperature_increment_on_fallback_optional": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "step": 0.1},
                ),
                "initial_prompt_optional": ("STRING", {"default": ""}),
                "suppress_numerals_optional": ("BOOLEAN", {"default": False}),
                "suppress_tokens_optional": ("STRING", {"default": "-1"}),
                "condition_on_previous_text_optional": ("BOOLEAN", {"default": False}),
                "beam_size_optional": ("INT", {"default": 5, "min": 1}),
                "best_of_optional": ("INT", {"default": 5, "min": 1}),
                "patience_optional": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "step": 0.1},
                ),
                "length_penalty_optional": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "step": 0.1},
                ),
                "logprob_threshold_optional": ("FLOAT", {"default": -1.0, "step": 0.1}),
                "no_speech_threshold_optional": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "compression_ratio_threshold_optional": (
                    "FLOAT",
                    {"default": 3.0, "step": 0.1},
                ),
                "no_align_optional": ("BOOLEAN", {"default": False}),
                "return_char_alignments_optional": ("BOOLEAN", {"default": False}),
                "interpolate_method_optional": (
                    ["nearest", "linear", "ignore"],
                    {"default": "nearest"},
                ),
                "max_line_width_optional": ("INT", {"default": -1, "min": -1}),
                "max_line_count_optional": ("INT", {"default": -1, "min": -1}),
                "highlight_words_optional": ("BOOLEAN", {"default": False}),
                "threads_optional": ("INT", {"default": 0, "min": 0}),
                "hotwords_optional": ("STRING", {"default": ""}),
                "allow_tf32_optional": ("BOOLEAN", {"default": False}),
                "propagate_log_optional": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "text",
        "segments_json",
        "srt",
        "vtt",
        "tsv",
        "aud",
        "json_result",
    )
    FUNCTION = "transcribe"
    CATEGORY = "WhisperX"

    def _get_optional_arg(self, kwargs, key, default_value):
        value = kwargs.get(key)
        if value is None:
            return default_value
        return value

    def _download_with_tqdm(
        self,
        repo_id: str,
        filename: str,
        target_dir: str,
        logger,
        token: Optional[str] = None,
    ):
        target_path = os.path.join(target_dir, filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if os.path.exists(target_path):
            logger.warning(f"File '{filename}' already exists. Skipping.")
            return

        temp_path = f"{target_path}.incomplete"

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Setup headers and initial position for each attempt
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                initial_pos = 0
                if os.path.exists(temp_path):
                    initial_pos = os.path.getsize(temp_path)
                    # Do not log resume message on every retry, only on the first attempt
                    if attempt == 0:
                        logger.info(
                            f"Resuming download for '{filename}' from {initial_pos} bytes."
                        )
                    headers["Range"] = f"bytes={initial_pos}-"

                url = hf_hub_url(repo_id=repo_id, filename=filename)

                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    is_resuming = r.status_code == 206 and initial_pos > 0
                    if r.status_code == 200 and initial_pos > 0:
                        logger.warning(
                            "Server does not support resume. Starting download from scratch."
                        )
                        initial_pos = 0
                    elif r.status_code not in [200, 206]:
                        r.raise_for_status()

                    total_size_str = r.headers.get(
                        "Content-Range", r.headers.get("Content-Length", "0")
                    ).split("/")[-1]
                    total_size = int(total_size_str)
                    file_mode = "ab" if is_resuming else "wb"

                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {filename}",
                        bar_format=self.PBAR_FORMAT,
                        initial=initial_pos,
                    ) as pbar:
                        with open(temp_path, file_mode) as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                shutil.move(temp_path, target_path)
                logger.success(f"Finished downloading '{filename}'.")
                return  # Exit successfully

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} for '{filename}' failed: {e}"
                )
                if attempt + 1 < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All download attempts failed for '{filename}'.")
                    raise  # Re-raise the final exception

    def _ensure_model_downloaded(
        self,
        model_name: str,
        repo_map: dict,
        cache_dir: str,
        token: Optional[str],
        message: str,
        logger,
    ):
        repo_id = repo_map.get(model_name)
        if not repo_id and ("/" in model_name or "pyannote" in model_name):
            repo_id = model_name

        if not repo_id:
            logger.warning(
                f"{message.capitalize()} '{model_name}' not found in predefined list. Cannot download."
            )
            return

        model_path = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")

        logger.info(
            f"Ensuring {message} '{model_name}' ({repo_id}) is available in {model_path}..."
        )

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                repo_files = list(list_repo_files(repo_id=repo_id, token=token))
                logger.info(
                    f"Found {len(repo_files)} files in repo. Checking local files..."
                )

                os.makedirs(model_path, exist_ok=True)

                for file in repo_files:
                    self._download_with_tqdm(repo_id, file, model_path, logger, token)

                logger.success(
                    f"{message.capitalize()} '{model_name}' is fully available locally."
                )
                return model_path  # Success

            except Exception as e:
                logger.warning(
                    f"Model check/download attempt {attempt + 1}/{max_retries} for '{model_name}' failed: {e}"
                )
                if attempt + 1 < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"All model check/download attempts failed for '{model_name}'."
                    )
                    return None  # Final failure

        return None  # Should not be reached, but for safety

    def _prepare_args(self, **kwargs):
        """Prepares the arguments for the transcription process based on node inputs."""
        logger = ColoredLogger()
        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            os.path.dirname(self._get_optional_arg(kwargs, "filename_prefix_optional", "whisperx/output")),
        )
        hf_cache_dir = os.path.join(folder_paths.models_dir, "whisperx")
        os.makedirs(hf_cache_dir, exist_ok=True)

        os.environ["TORCH_HOME"] = hf_cache_dir
        os.environ["PYANNOTE_CACHE"] = hf_cache_dir
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HF_HUB_CACHE"] = hf_cache_dir
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        try:
            temp_values = [
                float(t.strip()) for t in self._get_optional_arg(kwargs, "temperature_optional", "0.0").split(",")
            ]
        except (ValueError, AttributeError):
            logger.warning("Invalid temperature format. Using default value [0.0].")
            temp_values = [0.0]

        hf_token = self._get_optional_arg(kwargs, "hf_token_optional", "") or None

        # Ensure models are downloaded and get their local paths
        asr_model_path = self._ensure_model_downloaded(
            kwargs.get("model"),
            WHISPER_MODELS,
            hf_cache_dir,
            hf_token,
            "ASR model",
            logger,
        )
        if not asr_model_path:
            logger.error("Failed to prepare ASR model. Cannot continue.")
            return None

        diarize_enabled = self._get_optional_arg(kwargs, "diarize_optional", False)
        diarize_model_path = None
        if diarize_enabled:
            diarize_model_dir = self._ensure_model_downloaded(
                self._get_optional_arg(kwargs, "diarize_model_optional", diarization_models_list[0]),
                {},
                hf_cache_dir,
                hf_token,
                "diarization model",
                logger,
            )
            if diarize_model_dir and os.path.isdir(diarize_model_dir):
                diarize_model_path = os.path.join(diarize_model_dir, "config.yaml")
                if not os.path.exists(diarize_model_path):
                    logger.error(
                        f"config.yaml not found in {diarize_model_dir}. Cannot continue."
                    )
                    return None
            else:
                logger.error("Failed to prepare diarization model. Cannot continue.")
                return None

        align_model_val = self._get_optional_arg(kwargs, "align_model_optional", "auto")
        min_speakers_val = self._get_optional_arg(kwargs, "min_speakers_optional", -1)
        max_speakers_val = self._get_optional_arg(kwargs, "max_speakers_optional", -1)
        max_line_width_val = self._get_optional_arg(kwargs, "max_line_width_optional", -1)
        max_line_count_val = self._get_optional_arg(kwargs, "max_line_count_optional", -1)

        return {
            "model": asr_model_path,
            "language": kwargs.get("language")
            if kwargs.get("language") != "None"
            else None,
            "task": kwargs.get("task"),
            "batch_size": kwargs.get("batch_size"),
            "compute_type": kwargs.get("compute_type"),
            "device": kwargs.get("device"),
            "align_model": align_model_val if align_model_val != "auto" else None,
            "diarize": self._get_optional_arg(kwargs, "diarize_optional", False),
            "min_speakers": min_speakers_val if min_speakers_val > -1 else None,
            "max_speakers": max_speakers_val if max_speakers_val > -1 else None,
            "hf_token": hf_token,
            "output_dir": output_dir,
            "output_format": self._get_optional_arg(kwargs, "output_format_optional", "all"),
            "model_dir": hf_cache_dir,
            "device_index": 0,
            "verbose": True,
            "interpolate_method": self._get_optional_arg(kwargs, "interpolate_method_optional", "nearest"),
            "no_align": self._get_optional_arg(kwargs, "no_align_optional", False),
            "return_char_alignments": self._get_optional_arg(kwargs, "return_char_alignments_optional", False),
            "vad_method": self._get_optional_arg(kwargs, "vad_method_optional", "pyannote"),
            "vad_onset": self._get_optional_arg(kwargs, "vad_onset_optional", 0.5),
            "vad_offset": self._get_optional_arg(kwargs, "vad_offset_optional", 0.363),
            "chunk_size": self._get_optional_arg(kwargs, "chunk_size_optional", 30),
            "diarize_model": diarize_model_path,
            "speaker_embeddings": self._get_optional_arg(kwargs, "speaker_embeddings_optional", False),
            "temperature": tuple(temp_values),
            "best_of": self._get_optional_arg(kwargs, "best_of_optional", 5),
            "beam_size": self._get_optional_arg(kwargs, "beam_size_optional", 5),
            "patience": self._get_optional_arg(kwargs, "patience_optional", 1.0),
            "length_penalty": self._get_optional_arg(kwargs, "length_penalty_optional", 1.0),
            "suppress_tokens": self._get_optional_arg(kwargs, "suppress_tokens_optional", "-1"),
            "suppress_numerals": self._get_optional_arg(kwargs, "suppress_numerals_optional", False),
            "initial_prompt": self._get_optional_arg(kwargs, "initial_prompt_optional", "") or None,
            "condition_on_previous_text": self._get_optional_arg(
                kwargs, "condition_on_previous_text_optional", False
            ),
            "fp16": True,
            "temperature_increment_on_fallback": self._get_optional_arg(
                kwargs, "temperature_increment_on_fallback_optional", 0.2
            ),
            "compression_ratio_threshold": self._get_optional_arg(
                kwargs, "compression_ratio_threshold_optional", 3.0
            ),
            "logprob_threshold": self._get_optional_arg(kwargs, "logprob_threshold_optional", -1.0),
            "no_speech_threshold": self._get_optional_arg(kwargs, "no_speech_threshold_optional", 0.2),
            "max_line_width": max_line_width_val if max_line_width_val > -1 else None,
            "max_line_count": max_line_count_val if max_line_count_val > -1 else None,
            "highlight_words": self._get_optional_arg(kwargs, "highlight_words_optional", False),
            "threads": self._get_optional_arg(kwargs, "threads_optional", 0),
            "hotwords": self._get_optional_arg(kwargs, "hotwords_optional", "") or None,
            "allow_tf32": self._get_optional_arg(kwargs, "allow_tf32_optional", False),
            "print_progress": True,
        }

    def _process_audio(self, audio: dict, logger: ColoredLogger) -> np.ndarray:
        logger.info("Processing audio...")
        waveform = audio["waveform"].cpu()
        if audio["sample_rate"] != 16000:
            logger.info(f"Resampling audio from {audio['sample_rate']}Hz to 16000Hz...")
            waveform = torchaudio.functional.resample(
                waveform, audio["sample_rate"], 16000
            )

        audio_data = torch.mean(waveform[0], dim=0).numpy().astype(np.float32)
        logger.success("Audio processing complete.")
        return audio_data

    def _run_transcription(
        self, audio_data: np.ndarray, args: dict, logger: ColoredLogger
    ) -> dict:
        logger.info(f"Loading ASR model from: {args['model']}.")

        asr_options = {
            "beam_size": args["beam_size"],
            "best_of": args["best_of"],
            "patience": args["patience"],
            "length_penalty": args["length_penalty"],
            "temperatures": args["temperature"],
            "compression_ratio_threshold": args["compression_ratio_threshold"],
            "log_prob_threshold": args["logprob_threshold"],
            "no_speech_threshold": args["no_speech_threshold"],
            "condition_on_previous_text": args["condition_on_previous_text"],
            "initial_prompt": args["initial_prompt"],
            "suppress_tokens": [int(x) for x in args["suppress_tokens"].split(",")],
            "hotwords": args["hotwords"],
            "suppress_numerals": args["suppress_numerals"],
        }

        model_instance = None
        with tqdm(
            total=100,
            desc="Loading ASR Model",
            unit="%",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            with _capture_progress(pbar):
                model_instance = load_model(
                    args["model"],
                    device=args["device"],
                    device_index=args["device_index"],
                    compute_type=args["compute_type"],
                    language=args["language"],
                    asr_options=asr_options,
                    vad_method=args["vad_method"],
                    vad_options={
                        "chunk_size": args["chunk_size"],
                        "vad_onset": args["vad_onset"],
                        "vad_offset": args["vad_offset"],
                    },
                    task=args["task"],
                    threads=args["threads"],
                    local_files_only=True,
                )
                pbar.update(100 - pbar.n)

        # Custom message for Silero VAD loading if applicable
        if args["vad_method"] == "silero":
            hub_dir = torch.hub.get_dir()
            model_path = os.path.join(hub_dir, "snakers4_silero-vad_master")
            logger.info(f"Loading Silero VAD model from cache: {model_path}")

        logger.info("Starting transcription...")
        result = {}
        with tqdm(
            total=100,
            desc="Transcription",
            unit="%",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            with _capture_progress(pbar):
                result = model_instance.transcribe(
                    audio_data,
                    batch_size=args["batch_size"],
                    print_progress=args["print_progress"],
                )
        logger.success("Transcription complete.")

        del model_instance
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _run_alignment(
        self, result: dict, audio_data: np.ndarray, args: dict, logger: ColoredLogger
    ) -> dict:
        if args["no_align"] or args["task"] != "transcribe":
            return result

        logger.info("Starting alignment...")
        align_language = result.get("language", "en")

        # Determine the alignment model name if set to "auto"
        align_model_name = args["align_model"]
        if align_model_name is None:
            if align_language in CUSTOM_ALIGN_MODELS_MAP:
                align_model_name = CUSTOM_ALIGN_MODELS_MAP[align_language]
                logger.info(
                    f"Auto-detecting alignment model for '{align_language}': Found custom model -> {align_model_name}"
                )
            elif align_language in DEFAULT_ALIGN_MODELS_TORCH:
                align_model_name = DEFAULT_ALIGN_MODELS_TORCH[align_language]
                logger.info(
                    f"Auto-detecting alignment model for '{align_language}': Found default torchaudio model -> {align_model_name}"
                )
            elif align_language in DEFAULT_ALIGN_MODELS_HF:
                align_model_name = DEFAULT_ALIGN_MODELS_HF[align_language]
                logger.info(
                    f"Auto-detecting alignment model for '{align_language}': Found default HuggingFace model -> {align_model_name}"
                )
            else:
                logger.warning(
                    f"Could not auto-detect an alignment model for language '{align_language}'. Alignment may fail."
                )

        # Determine the local path for the alignment model.
        # For HF/custom models, this ensures they are downloaded and returns the local path.
        # For default torchaudio models, it remains the model name.
        align_model_path = align_model_name
        if (
            align_model_name is not None
            and align_model_name not in DEFAULT_ALIGN_MODELS_TORCH.values()
        ):
            align_model_path = self._ensure_model_downloaded(
                align_model_name,
                ALIGN_MODELS_HF,
                args["model_dir"],
                args["hf_token"],
                "alignment model",
                logger,
            )

        # Log the path or name that will be used
        logger.info(f"Loading alignment model from: {align_model_path or 'N/A'}")

        # Determine dynamic description for the progress bar
        tqdm_desc = "Loading Alignment Model"
        if align_model_name in torchaudio.pipelines.__all__:
            pipeline_obj = getattr(torchaudio.pipelines, align_model_name)
            if hasattr(pipeline_obj, "_path"):
                download_url = pipeline_obj._path
                filename = os.path.basename(download_url)
                expected_local_path = os.path.join(args["model_dir"], filename)
                if not os.path.exists(expected_local_path):
                    tqdm_desc = "Downloading Alignment Model"

        # Load the model using its name
        align_model_instance, align_metadata = (None, None)
        with tqdm(
            total=100,
            desc=tqdm_desc,  # Use the dynamic description
            unit="%",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            with _capture_progress(pbar):
                align_model_instance, align_metadata = load_align_model(
                    align_language,
                    args["device"],
                    model_name=align_model_path,
                    model_dir=args["model_dir"],
                )
                pbar.update(100 - pbar.n)

        aligned_result = {}
        with tqdm(
            total=100,
            desc="Alignment",
            unit="%",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            with _capture_progress(pbar):
                aligned_result = align(
                    result["segments"],
                    align_model_instance,
                    align_metadata,
                    audio_data,
                    args["device"],
                    interpolate_method=args["interpolate_method"],
                    return_char_alignments=args["return_char_alignments"],
                    print_progress=args["print_progress"],
                )
        logger.success("Alignment complete.")

        del align_model_instance
        gc.collect()
        torch.cuda.empty_cache()

        result.update(aligned_result)
        return result

    def _run_diarization(
        self, result: dict, audio_data: np.ndarray, args: dict, logger: ColoredLogger
    ) -> dict:
        if not args["diarize"]:
            return result

        logger.info("Starting diarization...")
        logger.info(f"Loading diarization model from: {args['diarize_model']}.")

        diarize_model_instance = None
        with tqdm(
            total=100,
            desc="Loading Diarization Model",
            unit="%",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            with _capture_progress(pbar):
                try:
                    diarize_model_instance = DiarizationPipeline(
                        model_name=args["diarize_model"],
                        use_auth_token=args["hf_token"],
                        device=args["device"],
                    )
                    pbar.update(100 - pbar.n)  # Update to 100% if successful
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'to'" in str(e):
                        logger.error(
                            "Failed to load diarization model. The model name may be incorrect or it might require a Hugging Face authentication token. Please check your inputs."
                        )
                        pbar.update(
                            100 - pbar.n
                        )  # Update to 100% even on error to finish bar
                        return result  # Return original result without diarization
                    else:
                        pbar.update(
                            100 - pbar.n
                        )  # Update to 100% even on error to finish bar
                        raise e  # Re-raise other unexpected AttributeErrors

        logger.info("Running diarization...")
        diarize_result = None
        with tqdm(
            total=1,
            desc="Diarization",
            unit="step",
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            diarize_result = diarize_model_instance(
                audio_data,
                min_speakers=args["min_speakers"],
                max_speakers=args["max_speakers"],
                return_embeddings=args["speaker_embeddings"],
            )
            pbar.update(1)

        speaker_embeddings = diarize_result[1] if args["speaker_embeddings"] else None
        diarize_segments = (
            diarize_result[0] if args["speaker_embeddings"] else diarize_result
        )

        final_result = assign_word_speakers(
            diarize_segments, result, speaker_embeddings
        )
        logger.success("Diarization complete.")

        del diarize_model_instance
        del diarize_result
        gc.collect()
        torch.cuda.empty_cache()

        return final_result

    def _write_outputs(
        self, result: dict, audio_basename: str, args: dict, logger: ColoredLogger
    ) -> tuple:
        logger.info("Writing output files...")
        os.makedirs(args["output_dir"], exist_ok=True)
        full_text = " ".join([seg["text"].strip() for seg in result["segments"]])
        segments_json = json.dumps(result["segments"], indent=2, ensure_ascii=False)

        outputs = {}
        formats_to_generate = (
            ["srt", "vtt", "tsv", "aud", "json", "txt"]
            if args["output_format"] == "all"
            else [args["output_format"]]
        )

        for fmt in ["srt", "vtt", "tsv", "aud", "json", "txt"]:
            outputs[fmt] = ""
            if fmt in formats_to_generate:
                writer = get_writer(fmt, args["output_dir"])
                output_path = os.path.join(
                    args["output_dir"], f"{audio_basename}.{fmt}"
                )

                options = {
                    "max_line_width": args["max_line_width"],
                    "max_line_count": args["max_line_count"],
                    "highlight_words": args["highlight_words"],
                }

                if fmt == "json":
                    outputs[fmt] = json.dumps(result, indent=2, ensure_ascii=False)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(outputs[fmt])
                else:
                    string_io = io.StringIO()
                    writer.write_result(result, file=string_io, options=options)
                    outputs[fmt] = string_io.getvalue()
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(outputs[fmt])

        logger.success("Process finished.")
        return (
            full_text,
            segments_json,
            outputs.get("srt"),
            outputs.get("vtt"),
            outputs.get("tsv"),
            outputs.get("aud"),
            outputs.get("json"),
        )

    def transcribe(self, **kwargs):
        print("--- WHISPERXNODE CALLED ---")
        logger.info("Starting transcription process...")
        pbar = comfy.utils.ProgressBar(6)
        whisperx_version = importlib.metadata.version("whisperx")
        logger.info(f"WhisperX library version: {whisperx_version}")

        # Set logger propagation based on user input
        propagate_log = kwargs.get("propagate_log_optional", False)
        logging.getLogger("whisperx").propagate = propagate_log
        if propagate_log:
            logger.info("WhisperX logger propagation has been enabled.")
        else:
            logger.info("WhisperX logger propagation has been disabled.")

        # Unify whisperx logger format to match the node's logger
        whisperx_root_logger = logging.getLogger("whisperx")
        new_formatter = ColoredFormatter("[WhisperXNode] %(message)s")
        for handler in whisperx_root_logger.handlers:
            handler.setFormatter(new_formatter)
        logger.info(
            "WhisperX logger format has been unified to match this node's format."
        )

        args = self._prepare_args(**kwargs)
        if not args:
            return ("", "", "", "", "", "", "")
        pbar.update(1)

        # Set TF32 based on user input
        if args["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TensorFloat-32 (TF32) has been enabled as requested.")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info(
                "TensorFloat-32 (TF32) has been disabled as requested (default). "
            )

        logger.warning(
            "Attention: TensorFloat-32 (TF32) is disabled by default to maintain reproducibility and accuracy. "
            "TF32 can accelerate computation on modern NVIDIA GPUs (Ampere and above) with a slight trade-off in numerical precision. "
            "If you prioritize speed and are not overly concerned about very small differences in results between sessions, "
            "you can enable it through the 'allow_tf32_optional' option in this node."
        )

        audio_basename = os.path.basename(
            kwargs.get("filename_prefix_optional", "whisperx_output")
        )

        audio_data = self._process_audio(kwargs.get("audio"), logger)
        pbar.update(1)
        result = self._run_transcription(audio_data, args, logger)
        pbar.update(1)
        result = self._run_alignment(result, audio_data, args, logger)
        pbar.update(1)
        result = self._run_diarization(result, audio_data, args, logger)
        pbar.update(1)

        output = self._write_outputs(result, audio_basename, args, logger)
        pbar.update(1)
        return output


NODE_CLASS_MAPPINGS = {"WhisperXNode": WhisperXNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WhisperXNode": "WhisperX Transcription"}
