#!/usr/bin/env python3
"""Generate a ~1 minute two-speaker conversation as a .wav file using Moonshine Voice TTS."""

import argparse
import struct
import wave

from moonshine_voice import TextToSpeech

FEMALE_VOICE = "kokoro_af_heart"
MALE_VOICE = "kokoro_am_michael"

CONVERSATION = [
    ("F", "Hey, did you end up trying that new coffee place on Fifth Street?"),
    ("M", "Oh yeah, I went yesterday morning. Got there right when they opened."),
    ("F", "And? Was it worth the hype?"),
    ("M", "Honestly, the cappuccino was really good. Like, surprisingly good for a place that just opened."),
    ("F", "Nice. I've been meaning to check it out but I keep getting stuck in meetings all morning."),
    ("M", "You should try going on a Saturday. It was pretty chill, not too crowded."),
    ("F", "That's a good idea. Do they have food too, or just drinks?"),
    ("M", "They had pastries and a couple of breakfast sandwiches. I tried the egg and cheese one. Pretty solid."),
    ("F", "Oh, I love a good breakfast sandwich. You're making me hungry now."),
    ("M", "Ha, sorry about that. But seriously, the vibe in there is really nice too. Lots of natural light, some plants. Very cozy."),
    ("F", "That sounds perfect for getting some work done on a weekend."),
    ("M", "Exactly what I was thinking. I might make it my regular spot."),
    ("F", "Well, save me a seat. I'll probably swing by this Saturday."),
    ("M", "Deal. I'll text you when I'm heading over."),
]

PAUSE_BETWEEN_SPEAKERS_SEC = 0.6
PAUSE_SAME_SPEAKER_SEC = 0.3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output",
        default="conversation.wav",
        help="Output .wav file path (default: conversation.wav)",
    )
    args = parser.parse_args()

    print("Initializing female voice...")
    tts_female = TextToSpeech("en-us", voice=FEMALE_VOICE)
    print("Initializing male voice...")
    tts_male = TextToSpeech("en-us", voice=MALE_VOICE)

    all_samples: list[float] = []
    sample_rate: int | None = None
    prev_speaker = None

    for speaker, text in CONVERSATION:
        tts = tts_female if speaker == "F" else tts_male
        label = "Female" if speaker == "F" else "Male"
        print(f"  [{label}] {text}")

        samples, sr = tts.synthesize(text)

        if sample_rate is None:
            sample_rate = sr
        assert sr == sample_rate, f"Sample rate mismatch: expected {sample_rate}, got {sr}"

        if all_samples:
            pause = PAUSE_BETWEEN_SPEAKERS_SEC if speaker != prev_speaker else PAUSE_SAME_SPEAKER_SEC
            all_samples.extend([0.0] * int(pause * sample_rate))

        all_samples.extend(samples)
        prev_speaker = speaker

    assert sample_rate is not None
    duration = len(all_samples) / sample_rate
    print(f"\nTotal duration: {duration:.1f}s")

    max_val = max(abs(s) for s in all_samples) or 1.0
    scale = 32767.0 / max_val

    with wave.open(args.output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for s in all_samples:
            wf.writeframes(struct.pack("<h", int(s * scale)))

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
