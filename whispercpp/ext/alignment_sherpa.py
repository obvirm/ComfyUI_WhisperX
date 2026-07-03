"""
sherpa-onnx CTC forced alignment.
Alternative to wav2vec2 alignment — uses dolphin-base-ctc-multi-lang.
"""

import logging, os, re, numpy as np
from pathlib import Path

logger = logging.getLogger("WhisperCPP.AlignSherpa")

SHERPA_AVAILABLE = False
SHERPA_MODEL_NAME = "sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02"

# Model dirs
try:
    import folder_paths
    MODEL_DIR = Path(folder_paths.models_dir) / "alignment" / "sherpa"
except ImportError:
    MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "alignment" / "sherpa"

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SHERPA_MODEL_NAME}.tar.bz2"
)

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
    logger.info("sherpa-onnx available")
except ImportError:
    logger.warning("sherpa-onnx not installed, run: pip install sherpa-onnx")


def ensure_model():
    """Download the Dolphin CTC model if needed."""
    model_dir = MODEL_DIR / SHERPA_MODEL_NAME
    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"
    if model_file.exists() and tokens_file.exists():
        return str(model_file), str(tokens_file)

    os.makedirs(MODEL_DIR, exist_ok=True)
    import urllib.request, tarfile
    archive = MODEL_DIR / f"{SHERPA_MODEL_NAME}.tar.bz2"
    logger.info(f"Downloading {SHERPA_MODEL_NAME} (81MB)...")
    urllib.request.urlretrieve(MODEL_URL, str(archive))
    with tarfile.open(str(archive)) as tar:
        tar.extractall(MODEL_DIR)
    archive.unlink()
    logger.info("Model downloaded and extracted")
    return str(model_file), str(tokens_file)


def load_align_model(device="cpu"):
    """Load sherpa-onnx CTC recognizer for alignment."""
    if not SHERPA_AVAILABLE:
        raise ImportError("sherpa-onnx not installed")

    model_path, tokens_path = ensure_model()
    logger.info(f"Loading sherpa CTC model: {model_path}")
    provider = "cuda" if device != "cpu" else "cpu"
    recognizer = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
        model=model_path,
        tokens=tokens_path,
        provider=provider,
        debug=False,
    )
    return recognizer


def _decode_tokens(tokens, token_ids):
    """Decode BPE token IDs back to text (basic SentencePiece cleanup)."""
    # tokens_file format: each line is a token string (0-indexed)
    text_parts = []
    for tid in token_ids:
        if 0 <= tid < len(tokens):
            tok = tokens[tid]
            # SentencePiece format: space-prefixed tokens start words
            if tok.startswith("_"):
                text_parts.append(" " + tok[1:])
            elif tok.startswith("▁"):
                text_parts.append(" " + tok[1:])
            else:
                text_parts.append(tok)
    return "".join(text_parts).strip()


def _merge_tokens_to_words(tokens_list, timestamps_list):
    """
    Merge CTC subword tokens into words with start/end timestamps.
    
    Returns: list of {"word": str, "start": float, "end": float}
    """
    if not tokens_list or not timestamps_list:
        return []
    
    words = []
    current_word_parts = []
    current_start = timestamps_list[0]
    
    for tok, ts in zip(tokens_list, timestamps_list):
        # If this token starts a new word (SentencePiece convention)
        # Space/underscore prefix = word start
        is_word_start = tok.startswith("_") or tok.startswith("▁") or tok.startswith(" ")
        
        if is_word_start and current_word_parts:
            # Flush previous word
            word_text = "".join(current_word_parts).strip()
            if word_text:
                words.append({
                    "word": word_text,
                    "start": current_start,
                    "end": ts,
                })
            current_word_parts = [tok.lstrip("_▁ ")]
            current_start = ts
        else:
            current_word_parts.append(tok)
    
    # Flush last word
    if current_word_parts:
        word_text = "".join(current_word_parts).strip()
        if word_text:
            words.append({
                "word": word_text,
                "start": current_start,
                "end": timestamps_list[-1] + 0.1,  # estimate end
            })
    
    return words


def align(segments, audio_data, recognizer, language="en"):
    """
    Align whisper segments using sherpa-onnx CTC model.
    
    Args:
        segments: list of whisper segment dicts with 'text' and optional 'words'
        audio_data: numpy array (float32, mono) of full audio
        recognizer: sherpa_onnx.OfflineRecognizer (CTC)
        language: language code (not used for CTC, kept for API compat)
    
    Returns:
        Updated segments with word-level timestamps.
    """
    sr = 16000  # CTC model expects 16kHz
    
    # Resample audio if needed
    if audio_data is None or len(audio_data) == 0:
        return segments
    
    # Run CTC model on full audio
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio_data.tolist())
    recognizer.decode_stream(stream)
    result = stream.result
    
    if not result.tokens:
        logger.warning("Sherpa CTC produced no tokens, keeping original segments")
        return segments
    
    # Merge CTC subword tokens into words with timestamps
    aligned_words = _merge_tokens_to_words(result.tokens, result.timestamps)
    
    if not aligned_words:
        logger.warning("Sherpa CTC alignment empty, keeping original")
        return segments
    
    # Map aligned timestamps to whisper segments
    # Strategy: match words by position and text similarity
    logger.info(f"CTC alignment: {len(aligned_words)} words from sherpa")
    
    # Build a word index from sherpa's output
    sherpa_text = " ".join(w["word"] for w in aligned_words).lower()
    whisper_text = " ".join(s.get("text", "") for s in segments).lower()
    
    # Extract all words from segments
    all_whisper_words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            all_whisper_words.extend(seg_words)
        else:
            # Split segment text into words
            for w in seg.get("text", "").split():
                all_whisper_words.append({
                    "word": w,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "segment_index": len(all_whisper_words),
                })
    
    if not all_whisper_words:
        logger.warning("No words to align")
        return segments
    
    # Simple sequential word mapping: assign timestamps from sherpa to whisper words
    # This works well when both ASR models produce similar text
    from difflib import SequenceMatcher
    
    # Match sherpa words to whisper words using sequence matching
    sherpa_word_strs = [w["word"].lower() for w in aligned_words]
    whisper_word_strs = [w["word"].lower() for w in all_whisper_words]
    
    matcher = SequenceMatcher(None, sherpa_word_strs, whisper_word_strs)
    mapping = {}  # whisper_word_idx -> (sherpa_word_idx, confidence)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                mapping[j1 + offset] = (i1 + offset, 1.0)
        elif tag == "replace":
            # Best effort: 1-to-1 mapping within the block
            min_len = min(i2 - i1, j2 - j1)
            for offset in range(min_len):
                mapping[j1 + offset] = (i1 + offset, 0.7)
    
    # Apply timestamps to whisper words
    for w_idx, w in enumerate(all_whisper_words):
        if w_idx in mapping:
            s_idx, _ = mapping[w_idx]
            if s_idx < len(aligned_words):
                w["start"] = round(aligned_words[s_idx]["start"], 3)
                w["end"] = round(aligned_words[s_idx]["end"], 3)
    
    # Rebuild segment word lists from aligned words
    seg_word_map = {}
    for w in all_whisper_words:
        seg_idx = w.get("segment_index", 0)
        # Find which segment this word belongs to
        for si, seg in enumerate(segments):
            seg_words = seg.get("words", [])
            for sw in seg_words:
                if sw.get("word") == w.get("word") and sw.get("start", 0) == seg.get("start", 0):
                    # This is a segment word - copy timestamps from our aligned version
                    if "start" in w and "end" in w and w["start"] != seg.get("start", 0):
                        sw["start"] = w["start"]
                        sw["end"] = w["end"]
    
    # Also update segment-level start/end from first/last word
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            seg["start"] = seg_words[0].get("start", seg.get("start", 0))
            seg["end"] = seg_words[-1].get("end", seg.get("end", 0))
    
    return segments
