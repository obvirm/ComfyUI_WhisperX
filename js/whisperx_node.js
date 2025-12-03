
import { app } from "/scripts/app.js";

const OPTIONAL_WIDGETS_NAMES = [
    "align_model_optional",
    "diarize_optional",
    "diarize_model_optional",
    "min_speakers_optional",
    "max_speakers_optional",
    "speaker_embeddings_optional",
    "hf_token_optional",
    "filename_prefix_optional",
    "output_format_optional",
    "vad_method_optional",
    "vad_onset_optional",
    "vad_offset_optional",
    "chunk_size_optional",
    "temperature_optional",
    "temperature_increment_on_fallback_optional",
    "initial_prompt_optional",
    "suppress_numerals_optional",
    "suppress_tokens_optional",
    "condition_on_previous_text_optional",
    "beam_size_optional",
    "best_of_optional",
    "patience_optional",
    "length_penalty_optional",
    "logprob_threshold_optional",
    "no_speech_threshold_optional",
    "compression_ratio_threshold_optional",
    "no_align_optional",
    "return_char_alignments_optional",
    "interpolate_method_optional",
    "max_line_width_optional",
    "max_line_count_optional",
    "highlight_words_optional",
    "threads_optional",
    "hotwords_optional",
    "allow_tf32_optional",
    "propagate_log_optional",
];

app.registerExtension({
    name: "WhisperX.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperXNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const toggleWidget = this.widgets.find(w => w.name === "show_advance_settings");

                const optionalWidgets = [];
                for (const name of OPTIONAL_WIDGETS_NAMES) {
                    const widget = this.widgets.find(w => w.name === name);
                    if (widget) {
                        optionalWidgets.push(widget);
                    }
                }

                const updateVisibility = (show) => {
                    // Remove optional widgets if they are present
                    if (!show) {
                        this.widgets = this.widgets.filter(w => !optionalWidgets.includes(w));
                    } else {
                        // Add widgets back if they are not present
                        const widgetsToAdd = optionalWidgets.filter(w => !this.widgets.includes(w));
                        if (widgetsToAdd.length > 0) {
                           const toggleIndex = this.widgets.indexOf(toggleWidget);
                           // We need to add them back in their original order, but for now, just add them.
                           // A better implementation would store original indices.
                           this.widgets.splice(toggleIndex + 1, 0, ...widgetsToAdd);
                        }
                    }

                    // Forcefully compute and set the new size
                    const newSize = this.computeSize();
                    this.size = newSize;

                    this.setDirtyCanvas(true, true);
                };

                toggleWidget.callback = (value) => {
                    updateVisibility(value);
                };

                // Initial update
                setTimeout(() => updateVisibility(toggleWidget.value), 10);
            };
        }
    },
});
