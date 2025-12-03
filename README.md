# ComfyUI-WhisperX

A custom node for ComfyUI that provides advanced transcription, alignment, and diarization capabilities using the WhisperX library.

![Example 1](picture/1%20(1).png)
![Example 2](picture/1%20(2).png)

## Features

- **High-Quality Transcription:** Utilizes various sizes of the Whisper model (including `large-v3`) for accurate speech-to-text.
- **Translation:** Translate speech from any supported language into English.
- **Word-Level Timestamps:** Generates precise word-level timestamps through forced alignment.
- **Speaker Diarization:** Identifies and labels different speakers in the audio.
- **Multiple Output Formats:** Save transcriptions in popular formats like SRT, VTT, TXT, TSV, and JSON.
- **Advanced Fine-Tuning:** A comprehensive set of advanced options to control the transcription process, including temperature, VAD (Voice Activity Detection) settings, beam size, and more.
- **Custom Model Support:** Easily add and use your own custom Whisper, alignment, or diarization models via a simple JSON configuration.
- **Automatic Model Management:** Models are automatically downloaded and cached from Hugging Face.
- **User-Friendly Interface:** Advanced options are neatly tucked away in a toggleable section to keep the UI clean.

## Installation

1.  Navigate to your `ComfyUI/custom_nodes/` directory.
2.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/ComfyUI-WhisperX
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r ComfyUI-WhisperX/requirements.txt
    ```
4.  Restart ComfyUI.

## Usage

1.  Add the **WhisperX Transcription** node to your workflow.
2.  Connect an `AUDIO` output from another node (e.g., a video loader) to the `audio` input.
3.  Select the desired `model`, `language`, and `task`.
4.  To access more detailed settings, enable the `show_advance_settings` toggle. This will reveal a host of options for fine-tuning the transcription, alignment, and diarization process.

## Customization

You can extend the node with your own models by creating or editing the `whisperx.json` file in the node's directory.

### Example `whisperx.json`:

```json
{
  "whisper_models": {
    "my_custom_whisper": "your-hf-account/my-custom-faster-whisper-model"
  },
  "custom_align_models": {
    "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
  },
  "diarization_models": [
    "pyannote/speaker-diarization-3.1",
    "your-hf-account/my-custom-diarization-model"
  ]
}
```

-   **`whisper_models`**: Add custom `faster-whisper` compatible models.
-   **`custom_align_models`**: Add custom alignment models. The key is the language code.
-   **`diarization_models`**: Add custom `pyannote` diarization models to the list.

The new models will appear in the respective dropdowns in the node's properties.

## Input Node

### Required
- `audio`: The audio to be transcribed.
- `model`: The Whisper model to use.
- `language`: The language of the audio. Set to `None` for auto-detection.
- `task`: `transcribe` or `translate` (to English).
- `batch_size`: The batch size for transcription.
- `compute_type`: The compute type for the model (`float16`, `float32`, `int8`).
- `device`: The device to run the model on (`cuda` or `cpu`).

### Optional (Advanced)
A large number of advanced options are available when `show_advance_settings` is enabled. These allow for detailed control over VAD, alignment, diarization, and the core Whisper transcription parameters. Hover over each option in ComfyUI for a detailed tooltip.

## Output Node

- `text`: The full transcribed text as a single string.
- `segments_json`: A JSON string of the transcription segments.
- `srt`, `vtt`, `tsv`, `aud`: The transcription in the respective file formats.
- `json_result`: The complete result object from WhisperX as a JSON string, including word-level details if alignment is enabled.

