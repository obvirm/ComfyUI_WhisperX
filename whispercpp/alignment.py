"""
whispercpp/alignment.py
Standalone word-level forced alignment using torchaudio native wav2vec2.
No whisperx dependency. torchaudio sudah terinstall bareng ComfyUI.
"""

import logging
import numpy as np
import torch
import torchaudio

logger = logging.getLogger("WhisperCPP.Alignment")

WAV2VEC2_AVAILABLE = False
DEFAULT_ALIGN_MODELS = {
    "en": "WAV2VEC2_ASR_LARGE_LV60K_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "WAV2VEC2_ASR_BASE_960H",
    "es": "WAV2VEC2_ASR_BASE_960H",
    "it": "WAV2VEC2_ASR_BASE_960H",
    "pt": "WAV2VEC2_ASR_BASE_960H",
    "pl": "WAV2VEC2_ASR_BASE_960H",
    "nl": "WAV2VEC2_ASR_BASE_960H",
    "ru": "WAV2VEC2_ASR_BASE_960H",
    "ja": "WAV2VEC2_ASR_BASE_960H",
    "zh": "WAV2VEC2_ASR_BASE_960H",
}

HF_ALIGN_MODELS = {
    "default": "facebook/wav2vec2-xlsr-53-56k",
    "en": "facebook/wav2vec2-large-lv60",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese",
    "ko": "elgeish/wav2vec2-large-xlsr-53-korean",
}

ALIGN_MODELS_AVAILABLE = []
HAVE_TRANSFORMERS = False

# Cek torchaudio pipelines
try:
    if hasattr(torchaudio.pipelines, "WAV2VEC2_ASR_LARGE_LV60K_960H"):
        WAV2VEC2_AVAILABLE = True
        for name in dir(torchaudio.pipelines):
            if "WAV2VEC2" in name or "VOXPOPULI" in name:
                ALIGN_MODELS_AVAILABLE.append(name)
        ALIGN_MODELS_AVAILABLE.sort()
        logger.info(f"torchaudio wav2vec2 pipelines: {len(ALIGN_MODELS_AVAILABLE)} available")
except Exception:
    pass

# Cek transformers untuk HF models (lebih akurat, multilingual)
try:
    import transformers
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    HAVE_TRANSFORMERS = True
    for k in HF_ALIGN_MODELS.values():
        if k not in ALIGN_MODELS_AVAILABLE:
            ALIGN_MODELS_AVAILABLE.append(k)
    logger.info("transformers available for HF wav2vec2 models")
except ImportError:
    pass


class AlignModel:
    """Wrapper for wav2vec2 alignment model (torchaudio or transformers)"""
    
    def __init__(self, model, processor, labels, is_hf=False):
        self.model = model
        self.processor = processor
        self.labels = labels
        self.is_hf = is_hf
        self.blank_id = 0 if is_hf else (labels.index("<blank>") if "<blank>" in labels else 0)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self


def get_model_name(language="en", model_name=None):
    """Resolve model name for a given language"""
    if model_name and model_name != "auto":
        return model_name
    
    # Try HF multilingual model first
    if HAVE_TRANSFORMERS:
        if language in HF_ALIGN_MODELS:
            return HF_ALIGN_MODELS[language]
        return HF_ALIGN_MODELS["default"]
    
    # Fallback to torchaudio pipeline
    if language in DEFAULT_ALIGN_MODELS:
        return DEFAULT_ALIGN_MODELS[language]
    return "WAV2VEC2_ASR_LARGE_LV60K_960H"


def load_align_model(language="en", device="cuda", model_name=None):
    """Load alignment model — either torchaudio pipeline or HuggingFace"""
    resolved = get_model_name(language, model_name)
    
    # Try transformers/HF first (more accurate)
    if HAVE_TRANSFORMERS and (model_name or resolved in HF_ALIGN_MODELS.values() or not WAV2VEC2_AVAILABLE):
        try:
            logger.info(f"Loading HF wav2vec2: {resolved}")
            processor = Wav2Vec2Processor.from_pretrained(resolved)
            model = Wav2Vec2ForCTC.from_pretrained(resolved)
            model = model.to(device)
            model.eval()
            labels = list(processor.tokenizer.get_vocab().keys()) if hasattr(processor.tokenizer, 'get_vocab') else processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
            return AlignModel(model, processor, labels, is_hf=True), resolved
        except Exception as e:
            logger.warning(f"HF model {resolved} failed: {e}, fallback to torchaudio")
    
    # Fallback: torchaudio pipeline
    if not WAV2VEC2_AVAILABLE:
        raise RuntimeError("No alignment model available. Install transformers or use torchaudio >= 0.12")
    
    pipename = resolved
    if hasattr(torchaudio.pipelines, pipename):
        bundle = getattr(torchaudio.pipelines, pipename)()
    else:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H()
    
    logger.info(f"Loading torchaudio pipeline: {type(bundle).__name__}")
    model = bundle.get_model()
    model = model.to(device)
    model.eval()
    labels = bundle.get_labels()
    processor = bundle  # torchaudio bundle acts as processor
    return AlignModel(model, processor, labels, is_hf=False), resolved


def align(segments, audio, align_model, language="en", device="cuda", return_char_alignments=False):
    """
    Align whisper segments using wav2vec2 forced alignment.
    
    Args:
        segments: list of dicts with 'start','end','text'
        audio: numpy audio array (mono, 16kHz)
        align_model: AlignModel instance
        language: language code
        device: device to run on
        return_char_alignments: if True, include character-level alignment
    
    Returns:
        segments with updated word timestamps
    """
    model = align_model.model
    processor = align_model.processor
    blank_id = align_model.blank_id
    sample_rate = 16000
    
    with torch.no_grad():
        # Process full audio
        if align_model.is_hf:
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values
            inputs = inputs.to(device)
            logits = model(inputs).logits[0]
        else:
            # torchaudio pipeline
            audio_t = torch.from_numpy(audio).float().to(device)
            if audio_t.dim() == 1:
                audio_t = audio_t.unsqueeze(0)
            emissions, _ = model(audio_t)
            logits = emissions[0]
        
        emissions = torch.log_softmax(logits, dim=-1).cpu()
    
    time_per_frame = len(audio) / (emissions.shape[0] * sample_rate)
    aligned_segments = []
    
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg.get("text", "").strip()
        words = seg.get("words", [])
        
        if not text or not words:
            aligned_segments.append(seg)
            continue
        
        # Compute alignment only for word list
        # Get word-level tokens based on model type
        word_str = " ".join(w.get("word", "").strip() for w in words if w.get("word", "").strip())
        if not word_str:
            aligned_segments.append(seg)
            continue
        
        try:
            # Tokenize word string
            if align_model.is_hf:
                tokens = processor.tokenizer.tokenize(word_str.lower())
                token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
            else:
                # torchaudio: use labels to tokenize
                char_tokens = []
                for ch in word_str.lower():
                    if ch.upper() in align_model.labels:
                        char_tokens.append(ch.upper())
                    elif ch == " ":
                        char_tokens.append("|")
                token_ids = [align_model.labels.index(t) if t in align_model.labels else blank_id for t in char_tokens]
            
            if not token_ids:
                aligned_segments.append(seg)
                continue
            
            # Map to emission indices
            seg_frame_start = int(seg_start / time_per_frame)
            seg_frame_end = int(seg_end / time_per_frame)
            seg_frame_end = min(seg_frame_end, emissions.shape[0])
            
            seg_emissions = emissions[seg_frame_start:seg_frame_end]
            target = torch.tensor([token_ids], dtype=torch.int32)
            
            # Forced alignment — returns (1, T) alignment path and scores
            target = torch.tensor([token_ids], dtype=torch.int32)  # (1, L)
            alignments, scores = torchaudio.functional.forced_align(
                seg_emissions.unsqueeze(0),  # (1, T, C)
                target,                     # (1, L)
                input_lengths=torch.tensor([seg_emissions.shape[0]], dtype=torch.int32),
                target_lengths=torch.tensor([len(token_ids)], dtype=torch.int32),
                blank=blank_id,
            )
            
            # alignments: (1, T) — each position is token_id or blank
            ali = alignments[0].tolist()  # list of length T
            
            if len(ali) == 0:
                aligned_segments.append(seg)
                continue
            
            # Find token occurrences in the alignment path
            # The target token_ids list maps to the actual text
            # Group consecutive same-token regions and extract start/end frames
            word_tokens = token_ids  # one token per word is ideal, but wav2vec uses character/byte tokens
            # For simplicity: find each non-blank run and assign to the next word
            frame_times = [(seg_frame_start + i) * time_per_frame for i in range(len(ali))]
            
            # Extract non-blank regions
            in_token = False
            regions = []
            region_start = 0
            for i, tid in enumerate(ali):
                if tid != blank_id and tid < len(token_ids):
                    if not in_token:
                        region_start = i
                        in_token = True
                else:
                    if in_token:
                        regions.append((region_start, i-1, ali[region_start]))
                        in_token = False
            if in_token:
                regions.append((region_start, len(ali)-1, ali[region_start]))
            
            # Map regions to words sequentially
            word_idx = 0
            words = seg.get('words', [])
            for ri, (f_start, f_end, tid) in enumerate(regions):
                if word_idx >= len(words):
                    break
                # Map token ID back to word index (sequential assignment)
                t_start = max(seg_start, (seg_frame_start + f_start) * time_per_frame)
                t_end = min(seg_end, (seg_frame_start + f_end) * time_per_frame)
                words[word_idx]['start'] = round(t_start, 3)
                words[word_idx]['end'] = round(t_end, 3)
                word_idx += 1
            
            seg['words'] = words
            aligned_segments.append(seg)
            
        except Exception as e:
            logger.warning(f"Alignment failed for segment '{text[:30]}...': {e}")
            aligned_segments.append(seg)
    
    return aligned_segments
