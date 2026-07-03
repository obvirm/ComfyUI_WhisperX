# Cpp-Annote

Port of the [pyannote](https://github.com/pyannote/pyannote-audio) diarization framework from Python to C++.

The original pyannote is the leading solution for turning raw input audio into segments of speech that are each tagged with the identity of the current speaker. It was created, and is maintained, by [Herve Bredin](https://herve.niderb.fr/) at [pyannote.ai](https://www.pyannote.ai/).

It's a great framework, with prebuilt models, but its dependencies on Python and pytorch mean it can't be used on phones, embedded devices, or WASM in the browser. This project attempts to be a faithful port of the community-1 model and code to C++, so it can be used on more platforms. I tried my best to ensure parity with the Python code, with [`scripts/eval_callhome_cpp_vs_python.py`](scripts/eval_callhome_cpp_vs_python.py) to verify the accuracy is approximately the same. There is a gap because this library uses models with quantized weights to reduce the on-disk size, so running the eval script on the first 120 seconds of each CALLHOME part two audio file shows a DER of 20.95%, instead of 19.73% for the original pyannote on the same data. If you aren't space-constrained you can use the float ONNX model files in the [artifacts](artifacts/) folder by
passing them into the CppAnnote constructor instead, which will give you 19.95% accuracy on the same evaluation.

## Build

You can use cmake to build this project, with a convenience script at [`scripts/build_cpp.sh`](scripts/build_cpp.sh). Most dependencies are bundled in the [`third_party/`](third_party/) folder, but you do need to have the headers and binary library for [ONNX Runtime](https://github.com/microsoft/onnxruntime) for your system installed at the path pointed to by the `ONNXRUNTIME_ROOT` environment variable. If you're having trouble obtaining or building ORT for your platform, you can look at [github.com/moonshine-ai/moonshine/tree/main/core/third-party/onnxruntime](https://github.com/moonshine-ai/moonshine/tree/main/core/third-party/onnxruntime) for include headers and binary libraries for many systems.

## Run

Cmake should create build/cpp-annote-cli as a binary executable. To run it, pass a .wav recording a conversation in as an argument:

```bash
build/cpp-annote-cli --wav audio/conversation.wav
```

You should see something like:

```bash
{
  "turns": [
    {"start": 0.28409375000000003, "end": 3.52409375, "speaker": 0},
    {"start": 4.9247187500000011, "end": 6.7809687500000004, "speaker": 1},
    {"start": 6.865343750000001, "end": 8.5865937500000005, "speaker": 1},
    {"start": 10.07159375, "end": 11.20221875, "speaker": 0},
    {"start": 12.585968750000001, "end": 18.120968749999999, "speaker": 1},
    {"start": 19.63971875, "end": 24.060968750000001, "speaker": 0},
    {"start": 25.478468750000001, "end": 29.494718750000001, "speaker": 1},
    {"start": 30.945968750000002, "end": 33.730343750000003, "speaker": 0},
    {"start": 35.080343750000004, "end": 41.037218750000001, "speaker": 1},
    {"start": 42.52221875, "end": 46.032218750000006, "speaker": 0},
    {"start": 47.415968750000005, "end": 53.035343750000003, "speaker": 1},
    {"start": 53.28846875, "end": 54.925343750000003, "speaker": 1},
    {"start": 56.410343750000003, "end": 59.346593750000004, "speaker": 0},
    {"start": 60.713468750000004, "end": 64.307843750000004, "speaker": 1},
    {"start": 65.792843750000003, "end": 68.813468749999998, "speaker": 0},
    {"start": 70.298468749999998, "end": 72.59346875, "speaker": 1}
  ]
}
[diarize] audio/conversation.wav  audio=73.15s  wall=17.042s  RTF=4.29x
```

This is a JSON dump of the segment and speaker identification information that the pipeline has generated for this conversation.

## Use

The CppAnnote class, defined in [src/cpp-annote.h](src/cpp-annote.h), contains the API to the library. The required models and data are compiled in to the library, so you don't need to pass anything in to the constructor. Once you have created a CppAnnote object, you can either pass in an entire conversation at once, using the `diarize()` function, or if you're operating on live audio, use the streaming methods.

Streaming mode allows you to supply audio in chunks over time, as it arrives from a microphone or other audio source. Since you might have more than one source you want to process at the same time, the API requires you to first call `create_stream()` to initialize a stream, `start_stream()` to begin processing, followed by repeated calls to `add_audio_to_stream()` to incrementally handle new audio as it comes in. The audio data should be mono, containing float values between -1.0 and 1.0, but it can be an arbitrary sample rate.

When you need to know the latest results of the segmentation and embedding, you can call `diarize_stream()`, which will return a list of turns, each of which has a start and end time, along with an estimated speaker ID. These IDs are numerical indices, starting at 0, with a new one added any time a new speaker is heard. The segmentation and speaker identification values may change over time as new audio is added, so you should not rely on segments being constant after they've been added to the list. You'll need to decide how often you want to call `diarize_stream()`, since it does involve some compute, so you'll need to figure out the right tradeoff between result frequency and overall compute load for your application.

## License

This framework is released under the MIT License (see LICENSE).

The [community-1 speaker diarization models](https://huggingface.co/pyannote/speaker-diarization-community-1) are released by pyannote.ai under the Creative Commons Attribution 4.0 License.

The code in [third_party](third_party/) is released under the licenses of the respective projects, which can be found in their respective folders:

[Eigen](https://github.com/PX4/eigen) uses the Mozilla Public License v2, with compiler flags set to exclude any of its source files that use other licenses.

[cnpy](https://github.com/rogersce/cnpy) is MIT licensed.

[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) is Apache v2 licensed.

[kissfft](https://github.com/mborgerding/kissfft) is licensed under BSD 3-clause.