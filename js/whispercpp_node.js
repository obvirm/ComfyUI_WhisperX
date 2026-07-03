import { app } from "/scripts/app.js";

const CPP_WIDGETS = [
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
    "vad","vad_model_path","vad_threshold","vad_min_speech_ms",
    "vad_min_silence_ms","vad_max_speech_s","vad_speech_pad_ms",
    "flash_attn","gpu_device",
    "dtw_token_timestamps","dtw_aheads_preset","dtw_n_top",
];

const EXT_WIDGETS = [
    "no_align","align_model","return_char_alignments",
    "diarize","diarize_model","min_speakers","max_speakers","hf_token",
];

function toggleWidgets(node, toggleWidget, widgetNames, show) {
    const widgets = widgetNames.map(n => node.widgets.find(w => w.name === n)).filter(Boolean);
    if (!show) {
        node.widgets = node.widgets.filter(w => !widgets.includes(w));
    } else {
        const toAdd = widgets.filter(w => !node.widgets.includes(w));
        if (toAdd.length) {
            const idx = node.widgets.indexOf(toggleWidget);
            node.widgets.splice(idx + 1, 0, ...toAdd);
        }
    }
    node.size = node.computeSize();
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "WhisperCPP.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperCPPNode") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onCreated?.apply(this, arguments);
                const cppToggle = this.widgets.find(w => w.name === "show_advance_cpp");
                const extToggle = this.widgets.find(w => w.name === "show_advance_ext");
                if (cppToggle) {
                    cppToggle.callback = (v) => toggleWidgets(this, cppToggle, CPP_WIDGETS, v);
                    setTimeout(() => toggleWidgets(this, cppToggle, CPP_WIDGETS, cppToggle.value), 10);
                }
                if (extToggle) {
                    extToggle.callback = (v) => toggleWidgets(this, extToggle, EXT_WIDGETS, v);
                    setTimeout(() => toggleWidgets(this, extToggle, EXT_WIDGETS, extToggle.value), 10);
                }
            };
        }
    },
});
