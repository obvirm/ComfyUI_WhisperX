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

app.registerExtension({
    name: "WhisperCPP.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperCPPNode") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onCreated?.apply(this, arguments);

                const cppToggle = this.widgets.find(w => w.name === "show_advance_cpp");
                const extToggle = this.widgets.find(w => w.name === "show_advance_ext");

                // Grab widget REFERENCES once, keep via closure
                const cppRefs = CPP_WIDGETS.map(n => this.widgets.find(w => w.name === n)).filter(Boolean);
                const extRefs = EXT_WIDGETS.map(n => this.widgets.find(w => w.name === n)).filter(Boolean);

                const setup = (toggle, refs) => {
                    if (!toggle) return;
                    const update = (show) => {
                        if (!show) {
                            this.widgets = this.widgets.filter(w => !refs.includes(w));
                        } else {
                            const toAdd = refs.filter(w => !this.widgets.includes(w));
                            if (toAdd.length) {
                                const idx = this.widgets.indexOf(toggle);
                                this.widgets.splice(idx + 1, 0, ...toAdd);
                            }
                        }
                        this.size = this.computeSize();
                        this.setDirtyCanvas(true, true);
                    };
                    toggle.callback = (v) => update(v);
                    setTimeout(() => update(toggle.value), 10);
                };

                setup(cppToggle, cppRefs);
                setup(extToggle, extRefs);
            };
        }
    },
});
