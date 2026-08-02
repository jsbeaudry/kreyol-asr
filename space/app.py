"""Haitian Creole streaming ASR — single model, fast path.

A 0.6B cache-aware streaming FastConformer RNN-T fine-tuned on 45.9 h of Haitian
Creole, prompted with `ht-HT` (slot 105).

Runs on ZeroGPU: the GPU exists only for the duration of a @spaces.GPU call, so
the model is restored to CPU at startup and moved onto CUDA per request. Doing
the download and restore at import time keeps the first request from paying for
2.5 GB of I/O.
"""

import json
import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import soxr
import spaces
import torch

SAMPLE_RATE = 16000
REPO = "jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht"
FILENAME = "nemotron-3.5-asr-streaming-0.6b-ht.nemo"
LANG = "ht-HT"

# Right context -> latency. Left context is always 56: it matches the encoder's
# sliding_window of 57, and anything else degrades the streaming behaviour the
# checkpoint was built around. Lower right context = lower latency, higher WER
# (measured: 17.3% at 80 ms, 14.7% at 1120 ms).
LATENCIES = {"80 ms": 0, "320 ms (default)": 3, "560 ms": 6, "1120 ms": 13}


def _restore():
    """Download + restore at import, so requests don't pay for it."""
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    path = hf_hub_download(REPO, FILENAME, token=os.environ.get("HF_TOKEN"))
    model = ASRModel.restore_from(path, map_location="cpu")
    model.eval()
    model.freeze()
    return model


MODEL = _restore()


def _to_wav16k(audio) -> str:
    """Gradio hands back (sample_rate, np.ndarray); NeMo wants a 16 kHz mono file."""
    sr, data = audio
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.abs(arr).max() or 0.0)
    if peak > 1.0:  # uploaded files arrive int16-scaled
        arr = arr / 32768.0
    if sr != SAMPLE_RATE:
        arr = soxr.resample(arr, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, np.clip(arr, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    return tmp.name


def _manifest(wav: str) -> str:
    """One-line manifest carrying `lang` — the only way to select the prompt slot.

    `transcribe(..., target_lang=...)` looks like it should work and does set
    `default_lang` on the dataloader config, but the Lhotse dataset ignores that
    key. It reads `cut.supervisions[0].language`, which comes from the manifest's
    `lang` field. Bare file paths make NeMo write a manifest without one, and
    decoding dies with "Unknown prompt key: 'None'".
    """
    info = sf.info(wav)
    mf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    mf.write(json.dumps({"audio_filepath": wav, "duration": info.duration,
                         "text": "", "lang": LANG, "target_lang": LANG}) + "\n")
    mf.close()
    return mf.name


@spaces.GPU(duration=90)
def run(audio, latency_label):
    if audio is None:
        return "Record or upload a clip first.", ""
    right = LATENCIES[latency_label]
    wav = _to_wav16k(audio)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL.encoder.set_default_att_context_size([56, right])
    MODEL.to(device)
    try:
        with torch.inference_mode():
            out = MODEL.transcribe(_manifest(wav), batch_size=1, verbose=False)
    except Exception as e:  # noqa: BLE001 - surfaced in the UI
        return f"Transcription failed: {type(e).__name__}: {e}", ""
    finally:
        MODEL.to("cpu")  # ZeroGPU reclaims the device once the call returns

    text = (getattr(out[0], "text", None) or str(out[0])) if out else "(no output)"
    return text, f"`att_context_size [56, {right}]` · {latency_label} · `{device}`"


with gr.Blocks(title="Haitian Creole streaming ASR") as demo:
    gr.Markdown(
        "# 🎙️ Haitian Creole streaming ASR\n"
        "[`jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht`]"
        "(https://huggingface.co/jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht) — "
        "a 0.6B streaming FastConformer RNN-T fine-tuned on 45.9 h of Kreyòl.\n\n"
        "**15.7% WER** at 320 ms on a 385-clip speaker-disjoint test set "
        "(the base model, which has no Creole slot, scores 98.2%)."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                            label="Creole audio")
        latency = gr.Dropdown(list(LATENCIES), value="320 ms (default)",
                              label="Streaming latency",
                              info="Less lookahead is faster but less accurate: "
                                   "17.3% WER at 80 ms vs 14.7% at 1120 ms.")
    go = gr.Button("Transcribe", variant="primary")
    out_text = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
    note_out = gr.Markdown()

    go.click(run, [audio_in, latency], [out_text, note_out])

    gr.Markdown(
        "---\n**Haitian Creole only.** Fine-tuning on Creole-only data moved the "
        "shared encoder and decoder, so this model does not retain the base model's "
        "other languages — prompting it with `fr-FR` or `en-US` returns Creole. Use "
        "[`nvidia/nemotron-3.5-asr-streaming-0.6b`]"
        "(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for anything else."
    )

if __name__ == "__main__":
    demo.launch()
