import gc, io, json, logging, os, sys, time
from contextlib import contextmanager
from typing import Optional
import comfy.utils, folder_paths
import numpy as np
import torch, torchaudio
from tqdm.auto import tqdm

from whispercpp.whisper_lib import WhisperCPP
from whispercpp.audio import AudioProcessor

# Model Manager — wrapped agar node tetap register meski tqdm/requests belum terinstall
WHISPERCPP_MODEL_AVAILABLE = False
try:
    from whispercpp.model import ModelManager, get_model_keys, load_custom_models
    WHISPERCPP_MODEL_AVAILABLE = True
except ImportError:
    # Fallback: node tetap jalan, tapi download model otomatis nggak akan work
    class DummyModelManager:
        def __init__(self): self._model_path = None
        def ensure_custom_config(self, *args): pass
        def get_model_path(self, key): return None
        def download_model(self, key):
            import os, sys
            logger = logging.getLogger("WhisperCPP")
            logger.error(f"Cannot download {key}: install tqdm & requests: pip install tqdm requests huggingface-hub")
            return None
    ModelManager = DummyModelManager
    GGML_FALLBACK_KEYS = ["large-v3-turbo","large-v3","medium","small","base","tiny","tiny.en","base.en","small.en","medium.en","large-v2"]
    def get_model_keys(): return GGML_FALLBACK_KEYS
    def load_custom_models(*args, **kwargs): pass

WHISPERX_AVAILABLE = False
try:
    from whisperx.alignment import align, load_align_model, DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer
    WHISPERX_AVAILABLE = True
except ImportError:
    DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF = {}, {}
    LANGUAGES, TO_LANGUAGE_CODE = {"en":"english"}, {"english":"en"}

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "whispercpp.json")

load_custom_models(CONFIG_PATH)
GGML_MODEL_KEYS = get_model_keys()

align_models_list = sorted(list(set(list(DEFAULT_ALIGN_MODELS_TORCH.values()) + list(DEFAULT_ALIGN_MODELS_HF.values()))))
diarization_models_list = ["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-2.1"]
CUSTOM_ALIGN_MODELS_MAP = {}
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        ca = cfg.get("custom_align_models", {})
        if isinstance(ca, dict):
            CUSTOM_ALIGN_MODELS_MAP = ca
            for v in ca.values():
                if v not in align_models_list: align_models_list.append(v)
            align_models_list.sort()
        dm = cfg.get("diarization_models", [])
        if dm: diarization_models_list = dm
except: pass

class ColoredLogger:
    def __init__(self, name="WhisperCPPNode"):
        self.name = name
        self.C = {"reset":"\033[0m","info":"\033[94m","success":"\033[92m","warning":"\033[93m","error":"\033[91m"}
    def _log(self, m, l): print(f"{self.C.get(l,self.C['reset'])}[{self.name}] {m}{self.C['reset']}")
    def info(self, m): self._log(m, "info")
    def success(self, m): self._log(m, "success")
    def warning(self, m): self._log(m, "warning")
    def error(self, m): self._log(m, "error")

logger = ColoredLogger()

@contextmanager
def _capture_progress(pbar):
    orig_out, orig_err = sys.stdout, sys.stderr
    class R(io.StringIO):
        def __init__(self, p, o):
            super().__init__(); self.pbar = p; self.orig = o
        def write(self, s): tqdm.write(s.rstrip(), file=self.orig)
        def flush(self): self.orig.flush()
    sys.stdout, sys.stderr = R(pbar, orig_out), R(pbar, orig_err)
    try: yield
    finally: sys.stdout, sys.stderr = orig_out, orig_err

class WhisperCPPNode:
    PBAR_FORMAT = "\033[96m{l_bar}\033[0m\033[92m{bar:15}\033[0m\033[93m{r_bar}\033[0m"
    _whisper = None
    _model_manager = None

    @classmethod
    def INPUT_TYPES(cls):
        lang_list = ["None"]
        if LANGUAGES: lang_list.extend(sorted(LANGUAGES.keys()))
        if TO_LANGUAGE_CODE: lang_list.extend(sorted(k.title() for k in TO_LANGUAGE_CODE.keys()))
        lang_list = list(dict.fromkeys(lang_list))

        required = {
            "audio": ("AUDIO",),
            "model": (GGML_MODEL_KEYS, {"default": "large-v3-turbo"}),
            "language": (lang_list, {"default": "None"}),
            "task": (["transcribe", "translate"],),
            "n_threads": ("INT", {"default": 4, "min": 1, "max": 64}),
            "device": (["auto", "cuda", "cpu", "vulkan", "metal"], {"default": "auto"}),
        }
        optional = {
            "show_advance_settings": ("BOOLEAN", {"default": False}),
            "sampling_strategy": (["greedy","beam_search"], {"default":"greedy"}),
            "best_of": ("INT", {"default":5,"min":1,"max":20}),
            "beam_size": ("INT", {"default":5,"min":1,"max":20}),
            "patience": ("FLOAT", {"default":1.0,"min":0.0,"max":10.0,"step":0.1}),
            "temperature": ("FLOAT", {"default":0.0,"min":0.0,"max":2.0,"step":0.1}),
            "temperature_inc": ("FLOAT", {"default":0.2,"min":0.0,"max":1.0,"step":0.1}),
            "max_initial_ts": ("FLOAT", {"default":1.0,"min":0.0,"max":60.0,"step":0.1}),
            "length_penalty": ("FLOAT", {"default":1.0,"min":0.0,"max":5.0,"step":0.1}),
            "n_max_text_ctx": ("INT", {"default":-1,"min":-1,"max":4096}),
            "offset_ms": ("INT", {"default":0,"min":0,"max":3600000}),
            "duration_ms": ("INT", {"default":0,"min":0,"max":3600000}),
            "no_context": ("BOOLEAN", {"default":False}),
            "single_segment": ("BOOLEAN", {"default":False}),
            "no_timestamps": ("BOOLEAN", {"default":False}),
            "max_tokens": ("INT", {"default":0,"min":0,"max":1024}),
            "max_len": ("INT", {"default":0,"min":0,"max":512}),
            "split_on_word": ("BOOLEAN", {"default":False}),
            "token_timestamps": ("BOOLEAN", {"default":False}),
            "thold_pt": ("FLOAT", {"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
            "thold_ptsum": ("FLOAT", {"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
            "suppress_blank": ("BOOLEAN", {"default":True}),
            "suppress_nst": ("BOOLEAN", {"default":True}),
            "suppress_regex": ("STRING", {"default":""}),
            "entropy_thold": ("FLOAT", {"default":2.4,"min":0.0,"max":10.0,"step":0.1}),
            "logprob_thold": ("FLOAT", {"default":-1.0,"min":-10.0,"max":0.0,"step":0.1}),
            "no_speech_thold": ("FLOAT", {"default":0.6,"min":0.0,"max":1.0,"step":0.01}),
            "initial_prompt": ("STRING", {"default":"","multiline":True}),
            "carry_initial_prompt": ("BOOLEAN", {"default":False}),
            "audio_ctx": ("INT", {"default":0,"min":0,"max":4096}),
            "debug_mode": ("BOOLEAN", {"default":False}),
            "print_special": ("BOOLEAN", {"default":False}),
            "print_progress": ("BOOLEAN", {"default":False}),
            "tdrz_enable": ("BOOLEAN", {"default":False}),
            "vad": ("BOOLEAN", {"default":False}),
            "vad_threshold": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.01}),
            "vad_min_speech_ms": ("INT", {"default":250,"min":0,"max":5000}),
            "vad_min_silence_ms": ("INT", {"default":100,"min":0,"max":5000}),
            "vad_max_speech_s": ("FLOAT", {"default":30.0,"min":1.0,"max":300.0,"step":0.5}),
            "vad_speech_pad_ms": ("INT", {"default":400,"min":0,"max":2000}),
            "filename_prefix": ("STRING", {"default":"whispercpp/output"}),
            "output_format": (["all","srt","vtt","txt","tsv","json","aud"],),
            "align_model": (["auto"]+align_models_list, {"default":"auto"}),
            "no_align": ("BOOLEAN", {"default":False}),
            "interpolate_method": (["nearest","linear","ignore"], {"default":"nearest"}),
            "return_char_alignments": ("BOOLEAN", {"default":False}),
            "diarize": ("BOOLEAN", {"default":False}),
            "diarize_model": (diarization_models_list, {"default":diarization_models_list[0]}),
            "min_speakers": ("INT", {"default":-1,"min":-1,"max":20}),
            "max_speakers": ("INT", {"default":-1,"min":-1,"max":20}),
            "hf_token": ("STRING", {"default":""}),
            "flash_attn": ("BOOLEAN", {"default":False}),
            "gpu_device": ("INT", {"default":0,"min":0,"max":8}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("text","segments_json","srt","vtt","tsv","aud","json_result")
    FUNCTION = "transcribe"
    CATEGORY = "WhisperCPP"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _get(self, kw, key, default=None):
        v = kw.get(key); return v if v is not None else default

    def _ensure_whisper(self):
        if WhisperCPPNode._whisper is None: WhisperCPPNode._whisper = WhisperCPP()
        return WhisperCPPNode._whisper

    def _ensure_mgr(self):
        if WhisperCPPNode._model_manager is None:
            WhisperCPPNode._model_manager = ModelManager()
        return WhisperCPPNode._model_manager

    def transcribe(self, **kwargs):
        logger.info("="*60); logger.info("  WhisperCPP Node starting"); logger.info("="*60)
        pbar = comfy.utils.ProgressBar(6)

        hf_cache = os.path.join(folder_paths.models_dir, "whispercpp")
        os.makedirs(hf_cache, exist_ok=True)
        out_dir = os.path.join(folder_paths.get_output_directory(), os.path.dirname(self._get(kwargs,"filename_prefix","whispercpp/output")))
        os.makedirs(out_dir, exist_ok=True)
        audio_base = os.path.basename(self._get(kwargs,"filename_prefix","whispercpp_output"))
        os.environ.update({"TORCH_HOME":hf_cache,"PYANNOTE_CACHE":hf_cache,"HF_HOME":hf_cache,"HF_HUB_CACHE":hf_cache,"HF_HUB_DISABLE_SYMLINKS_WARNING":"1"})
        hf_token = self._get(kwargs,"hf_token","") or None
        pbar.update(1)

        model_key = self._get(kwargs,"model","large-v3-turbo")
        mgr = self._ensure_mgr(); mgr.ensure_custom_config(CONFIG_PATH)
        model_path = mgr.get_model_path(model_key)
        if not model_path:
            logger.info(f"Downloading {model_key}...")
            model_path = mgr.download_model(model_key)
            if not model_path: logger.error("Model failed"); return ("","","","","","","")
        pbar.update(1)

        device = self._get(kwargs,"device","auto")
        use_gpu = device in ("auto","cuda","vulkan","metal")
        wcpp = self._ensure_whisper()
        try: wcpp.load_model(model_path, use_gpu=use_gpu, gpu_device=self._get(kwargs,"gpu_device",0), flash_attn=self._get(kwargs,"flash_attn",False))
        except RuntimeError as e: logger.error(f"Load failed: {e}"); return ("","","","","","","")
        pbar.update(1)

        audio_data = AudioProcessor.process_comfy_audio(kwargs.get("audio"))
        pbar.update(1)

        lang = self._get(kwargs,"language","None")
        if lang == "None": lang = None
        strat_str = self._get(kwargs,"sampling_strategy","greedy")
        strategy = 0 if strat_str == "greedy" else 1

        vad_params = {}
        if self._get(kwargs,"vad",False):
            vad_params = {"vad":True,"vad_threshold":self._get(kwargs,"vad_threshold",0.5),"vad_min_speech_duration_ms":self._get(kwargs,"vad_min_speech_ms",250),"vad_min_silence_duration_ms":self._get(kwargs,"vad_min_silence_ms",100),"vad_max_speech_duration_s":self._get(kwargs,"vad_max_speech_s",30.0),"vad_speech_pad_ms":self._get(kwargs,"vad_speech_pad_ms",400)}

        tp = { "strategy":strategy, "n_threads":self._get(kwargs,"n_threads",4), "language":lang, "detect_language":bool(lang is None), "task":self._get(kwargs,"task","transcribe"),
            "temperature":self._get(kwargs,"temperature",0.0), "temperature_inc":self._get(kwargs,"temperature_inc",0.2), "max_initial_ts":self._get(kwargs,"max_initial_ts",1.0), "length_penalty":self._get(kwargs,"length_penalty",1.0),
            "best_of":self._get(kwargs,"best_of",5), "beam_size":self._get(kwargs,"beam_size",5), "patience":self._get(kwargs,"patience",1.0),
            "entropy_thold":self._get(kwargs,"entropy_thold",2.4), "logprob_thold":self._get(kwargs,"logprob_thold",-1.0), "no_speech_thold":self._get(kwargs,"no_speech_thold",0.6),
            "n_max_text_ctx":self._get(kwargs,"n_max_text_ctx",-1), "offset_ms":self._get(kwargs,"offset_ms",0), "duration_ms":self._get(kwargs,"duration_ms",0),
            "no_context":self._get(kwargs,"no_context",False), "single_segment":self._get(kwargs,"single_segment",False), "no_timestamps":self._get(kwargs,"no_timestamps",False),
            "max_tokens":self._get(kwargs,"max_tokens",0), "max_len":self._get(kwargs,"max_len",0), "split_on_word":self._get(kwargs,"split_on_word",False),
            "token_timestamps":self._get(kwargs,"token_timestamps",False), "thold_pt":self._get(kwargs,"thold_pt",0.01), "thold_ptsum":self._get(kwargs,"thold_ptsum",0.01),
            "suppress_blank":self._get(kwargs,"suppress_blank",True), "suppress_nst":self._get(kwargs,"suppress_nst",True), "suppress_regex":self._get(kwargs,"suppress_regex","") or None,
            "initial_prompt":self._get(kwargs,"initial_prompt","") or None, "carry_initial_prompt":self._get(kwargs,"carry_initial_prompt",False),
            "audio_ctx":self._get(kwargs,"audio_ctx",0), "debug_mode":self._get(kwargs,"debug_mode",False),
            "print_special":self._get(kwargs,"print_special",False), "print_progress":self._get(kwargs,"print_progress",False),
            "tdrz_enable":self._get(kwargs,"tdrz_enable",False), **vad_params }

        try: result = wcpp.transcribe(audio_data, **tp)
        except Exception as e: logger.error(f"Failed: {e}"); import traceback; traceback.print_exc(); return ("","","","","","","")
        pbar.update(1)

        result = self._run_alignment(result, audio_data, kwargs, hf_cache, hf_token)
        pbar.update(1)
        result = self._run_diarization(result, audio_data, kwargs, hf_token)
        pbar.update(1)

        return self._write_outputs(result, audio_base, out_dir, kwargs)

    def _run_alignment(self, result, audio_data, kwargs, model_dir, hf_token):
        if self._get(kwargs,"no_align",False) or self._get(kwargs,"task","transcribe") != "transcribe" or not WHISPERX_AVAILABLE: return result
        logger.info("Starting alignment...")
        align_lang = result.get("language","en")
        align_val = self._get(kwargs,"align_model","auto")
        if align_val == "auto":
            if align_lang in CUSTOM_ALIGN_MODELS_MAP: align_val = CUSTOM_ALIGN_MODELS_MAP[align_lang]
            elif align_lang in DEFAULT_ALIGN_MODELS_TORCH: align_val = DEFAULT_ALIGN_MODELS_TORCH[align_lang]
            elif align_lang in DEFAULT_ALIGN_MODELS_HF: align_val = DEFAULT_ALIGN_MODELS_HF[align_lang]
            else: return result
        try:
            segs = [{"start":s["start"],"end":s["end"],"text":s["text"],"words":s.get("words",[])} for s in result["segments"]]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with tqdm(total=100,desc="Loading Align Model",bar_format=self.PBAR_FORMAT) as pb:
                with _capture_progress(pb):
                    am, ameta = load_align_model(align_lang, device, model_name=align_val, model_dir=model_dir)
            with tqdm(total=100,desc="Alignment",bar_format=self.PBAR_FORMAT) as pb:
                with _capture_progress(pb):
                    aligned = align(segs, am, ameta, audio_data, device, interpolate_method=self._get(kwargs,"interpolate_method","nearest"), return_char_alignments=self._get(kwargs,"return_char_alignments",False))
            result["segments"] = aligned; logger.success("Alignment done")
        except Exception as e: logger.warning(f"Align failed: {e}")
        return result

    def _run_diarization(self, result, audio_data, kwargs, hf_token):
        if not self._get(kwargs,"diarize",False) or not WHISPERX_AVAILABLE: return result
        logger.info("Starting diarization...")
        dm = self._get(kwargs,"diarize_model","pyannote/speaker-diarization-3.1")
        ms = self._get(kwargs,"min_speakers",-1); xs = self._get(kwargs,"max_speakers",-1)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with tqdm(total=100,desc="Loading Diarize Model",bar_format=self.PBAR_FORMAT) as pb:
                with _capture_progress(pb): dp = DiarizationPipeline(model_name=dm, use_auth_token=hf_token, device=device)
            with tqdm(total=1,desc="Diarization",bar_format=self.PBAR_FORMAT) as pb:
                dr = dp(audio_data, min_speakers=ms if ms>0 else None, max_speakers=xs if xs>0 else None)
            fr = assign_word_speakers(dr, result)
            result["segments"] = fr.get("segments", result["segments"]); logger.success("Diarization done")
        except Exception as e: logger.warning(f"Diarize failed: {e}")
        return result

    def _write_outputs(self, result, audio_base, out_dir, kwargs):
        os.makedirs(out_dir, exist_ok=True)
        full_text = " ".join([s.get("text","").strip() for s in result.get("segments",[])])
        seg_json = json.dumps(result.get("segments",[]), indent=2, ensure_ascii=False)
        fmt = self._get(kwargs,"output_format","all")
        fmts = ["srt","vtt","tsv","aud","json","txt"] if fmt=="all" else [fmt]
        out = {f:"" for f in ["srt","vtt","tsv","aud","json","txt"]}
        for f in fmts:
            path = os.path.join(out_dir, f"{audio_base}.{f}")
            if f == "json":
                jr = json.dumps(result, indent=2, ensure_ascii=False); out[f]=jr
                with open(path,"w",encoding="utf-8") as fp: fp.write(jr)
            elif f == "txt":
                out[f]=full_text; open(path,"w",encoding="utf-8").write(full_text)
            elif f == "srt":
                c = self._segs_to_srt(result.get("segments",[])); out[f]=c; open(path,"w",encoding="utf-8").write(c)
            elif f == "vtt":
                c = self._segs_to_vtt(result.get("segments",[])); out[f]=c; open(path,"w",encoding="utf-8").write(c)
            elif f == "tsv":
                c = self._segs_to_tsv(result.get("segments",[])); out[f]=c; open(path,"w",encoding="utf-8").write(c)
            elif f == "aud":
                c = self._segs_to_aud(result.get("segments",[])); out[f]=c; open(path,"w",encoding="utf-8").write(c)
        return (full_text, seg_json, out.get("srt",""), out.get("vtt",""), out.get("tsv",""), out.get("aud",""), out.get("json",""))

    @staticmethod
    def _ts(sec, fmt="srt"):
        if sec < 0: sec = 0
        h, m = int(sec//3600), int((sec%3600)//60)
        s = sec % 60
        if fmt=="vtt": return f"{h:02d}:{m:02d}:{s:06.3f}"
        ms = int((s-int(s))*1000); return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"

    def _segs_to_srt(self, segs):
        lines = []
        for i,s in enumerate(segs,1):
            lines.append(str(i)); lines.append(f"{self._ts(s['start'],'srt')} --> {self._ts(s['end'],'srt')}"); lines.append(s.get("text","").strip()); lines.append("")
        return "\n".join(lines)
    def _segs_to_vtt(self, segs):
        lines = ["WEBVTT",""]
        for s in segs:
            lines.append(f"{self._ts(s['start'],'vtt')} --> {self._ts(s['end'],'vtt')}"); lines.append(s.get("text","").strip()); lines.append("")
        return "\n".join(lines)
    def _segs_to_tsv(self, segs):
        lines = ["start\tend\ttext"]
        for s in segs: lines.append(f"{s['start']:.3f}\t{s['end']:.3f}\t{s.get('text','').strip()}")
        return "\n".join(lines)
    def _segs_to_aud(self, segs):
        return "\n".join(f"{s['start']:.3f},{s['end']:.3f},{s.get('text','').strip()}" for s in segs)

NODE_CLASS_MAPPINGS = {"WhisperCPPNode": WhisperCPPNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WhisperCPPNode": "WhisperCPP Transcription"}
