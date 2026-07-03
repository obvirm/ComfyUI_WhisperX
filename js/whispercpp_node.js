import { app } from "/scripts/app.js";

const OPTIONAL_WIDGETS_NAMES = [
    "sampling_strategy","best_of","beam_size","patience",
    "temperature","temperature_inc","max_initial_ts","length_penalty",
    "n_max_text_ctx","offset_ms","duration_ms",
    "no_context","single_segment","no_timestamps","max_tokens","max_len","split_on_word",
    "token_timestamps","thold_pt","thold_ptsum",
    "suppress_blank","suppress_nst","suppress_regex",
    "entropy_thold","logprob_thold","no_speech_thold",
    "initial_prompt","carry_initial_prompt",
    "audio_ctx","debug_mode","print_special","print_progress",
    "tdrz_enable","grammar_penalty",
    "vad","vad_threshold","vad_min_speech_ms","vad_min_silence_ms","vad_max_speech_s","vad_speech_pad_ms",
    "filename_prefix","output_format",
    "flash_attn","gpu_device",
    "dtw_token_timestamps","dtw_aheads_preset","dtw_n_top",
    "no_align","align_model","return_char_alignments",
    "diarize","diarize_model","min_speakers","max_speakers","hf_token",
];

app.registerExtension({
    name: "WhisperCPP.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperCPPNode") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onCreated?.apply(this, arguments);
                const toggle = this.widgets.find(w => w.name === "show_advance_settings");
                const optWidgets = OPTIONAL_WIDGETS_NAMES.map(n => this.widgets.find(w => w.name === n)).filter(Boolean);
                const update = (show) => {
                    if (!show) { this.widgets = this.widgets.filter(w => !optWidgets.includes(w)); }
                    else {
                        const toAdd = optWidgets.filter(w => !this.widgets.includes(w));
                        if (toAdd.length) this.widgets.splice(this.widgets.indexOf(toggle)+1, 0, ...toAdd);
                    }
                    this.size = this.computeSize(); this.setDirtyCanvas(true, true);
                };
                toggle.callback = (v) => update(v);
                setTimeout(() => update(toggle.value), 10);
            };
        }
    },
});
