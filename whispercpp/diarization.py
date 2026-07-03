"""
whispercpp/diarization.py
Standalone speaker diarization using pyannote.audio directly.
No whisperx dependency.
"""

import logging
import numpy as np

logger = logging.getLogger("WhisperCPP.Diarization")

DIARIZATION_AVAILABLE = False
DIARIZATION_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-2.1",
]

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    DIARIZATION_AVAILABLE = True
    logger.info("pyannote.audio available for speaker diarization")
except ImportError:
    pass


def load_diarization_pipeline(model_name="pyannote/speaker-diarization-3.1", device="cuda", hf_token=None):
    """Load pyannote diarization pipeline"""
    if not DIARIZATION_AVAILABLE:
        raise RuntimeError("pyannote.audio not installed. Run: pip install pyannote.audio")
    
    logger.info(f"Loading diarization pipeline: {model_name}")
    pipeline = Pipeline.from_pretrained(
        model_name,
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(device)
    return pipeline


def diarize(audio, pipeline, sample_rate=16000, min_speakers=None, max_speakers=None):
    """
    Run speaker diarization on audio.
    
    Args:
        audio: numpy audio array (mono, 16kHz)
        pipeline: pyannote Pipeline
        sample_rate: sample rate of audio
        min_speakers: minimum number of speakers
        max_speakers: maximum number of speakers
    
    Returns:
        list of dicts: [{"speaker": "SPEAKER_00", "start": float, "end": float}, ...]
    """
    from pyannote.core import AudioFile
    from pyannote.audio import Audio
    
    # Convert numpy to pyannote format
    waveform = np.expand_dims(audio, axis=0)  # (1, T)
    
    # Run diarization
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    
    # Extract speaker turns
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })
    
    turns.sort(key=lambda x: x["start"])
    return turns


def assign_speakers_to_segments(segments, diarization_turns):
    """
    Assign speaker labels to whisper segments based on diarization results.
    
    Uses majority-overlap assignment: each segment gets the speaker
    that has the most overlap with it.
    """
    if not diarization_turns:
        return segments
    
    for seg in segments:
        seg_s = seg["start"]
        seg_e = seg["end"]
        
        # Find overlapping turns
        overlaps = {}
        for turn in diarization_turns:
            overlap_start = max(seg_s, turn["start"])
            overlap_end = min(seg_e, turn["end"])
            overlap_dur = max(0, overlap_end - overlap_start)
            if overlap_dur > 0:
                speaker = turn["speaker"]
                overlaps[speaker] = overlaps.get(speaker, 0) + overlap_dur
        
        if overlaps:
            best_speaker = max(overlaps, key=overlaps.get)
            seg["speaker"] = best_speaker
            
            # Assign speaker to words
            for w in seg.get("words", []):
                w_s = w.get("start", seg_s)
                w_e = w.get("end", seg_e)
                w_overlaps = {}
                for turn in diarization_turns:
                    o_start = max(w_s, turn["start"])
                    o_end = min(w_e, turn["end"])
                    o_dur = max(0, o_end - o_start)
                    if o_dur > 0:
                        w_overlaps[turn["speaker"]] = w_overlaps.get(turn["speaker"], 0) + o_dur
                if w_overlaps:
                    w["speaker"] = max(w_overlaps, key=w_overlaps.get)
    
    return segments
