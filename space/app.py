"""Haitian Creole streaming ASR — thin client over a RunPod Serverless endpoint.

The model does not run here. This Space records audio, converts it to 16 kHz mono
WAV, and calls a RunPod worker that holds the fine-tuned checkpoint. That keeps the
build to a few light wheels instead of a 15-minute NeMo-from-source install, and
means the Space needs no GPU at all.

Requires two Space secrets/variables:
    RUNPOD_API_KEY   (secret)   your RunPod key
    RUNPOD_ENDPOINT  (variable) endpoint id, e.g. 9fds364d4gicy0
"""

import base64
import io
import os
import time

import gradio as gr
import numpy as np
import requests
import soundfile as sf
import soxr

SAMPLE_RATE = 16000
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT = os.environ.get("RUNPOD_ENDPOINT", "9fds364d4gicy0")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"

# Right context -> latency, with the WER measured on a 385-clip speaker-disjoint
# test set. Left context is fixed at 56 by the worker and is not selectable.
LATENCIES = {
    "80 ms — 17.3% WER": 80,
    "320 ms — 15.7% WER": 320,
    "560 ms — 14.9% WER": 560,
    "1120 ms — 14.7% WER": 1120,
}

# A cold worker pulls ~13 GB and imports NeMo before it can decode.
POLL_SECONDS = 2
MAX_WAIT_SECONDS = 240


def _wav16k_base64(audio) -> tuple[str, float]:
    """Gradio gives (sample_rate, np.ndarray). The worker decodes with libsndfile,
    so hand it plain 16 kHz mono PCM16 rather than whatever the browser recorded."""
    sr, data = audio
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.abs(arr).max() or 0.0)
    if peak > 1.0:  # uploaded files arrive int16-scaled
        arr = arr / 32768.0
    if sr != SAMPLE_RATE:
        arr = soxr.resample(arr, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.clip(arr, -1.0, 1.0), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode(), len(arr) / SAMPLE_RATE


def transcribe(audio, latency_label, progress=gr.Progress()):
    if audio is None:
        return "Record or upload a clip first.", ""
    if not API_KEY:
        return ("RUNPOD_API_KEY is not set. Add it under Settings → Variables and "
                "secrets, then restart the Space."), ""

    started = time.time()
    try:
        progress(0, desc="Converting to 16 kHz WAV")
        b64, seconds = _wav16k_base64(audio)
    except Exception as e:  # noqa: BLE001
        return f"Could not read that audio: {type(e).__name__}: {e}", ""

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"input": {"audio_base64": b64, "latency_ms": LATENCIES[latency_label]}}

    try:
        # /run + poll rather than /runsync: a cold start can take ~2 minutes,
        # which a synchronous call will not sit through.
        r = requests.post(f"{BASE}/run", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        job = r.json().get("id")
        if not job:
            return f"No job id in response: {r.text[:200]}", ""

        deadline = time.time() + MAX_WAIT_SECONDS
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            s = requests.get(f"{BASE}/status/{job}", headers=headers, timeout=30).json()
            state = s.get("status", "?")
            if state == "COMPLETED":
                out = s.get("output") or {}
                if out.get("error"):
                    return f"Worker error: {out['error']}", ""
                text = out.get("text") or ""
                if not text.strip():
                    note = out.get("note", "no speech decoded")
                    return (f"(nothing transcribed) — {note}",
                            _metrics(s, out, seconds, started))
                return text, _metrics(s, out, seconds, started)
            if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
                return f"{state}: {str(s.get('error') or s.get('output'))[:300]}", ""
            waited = int(time.time() - started)
            progress(min(waited / MAX_WAIT_SECONDS, 0.95),
                     desc=f"{state.lower().replace('_', ' ')} — {waited}s"
                          + (" (cold start takes ~2 min)" if waited > 15 else ""))
        return f"Timed out after {MAX_WAIT_SECONDS}s.", ""
    except requests.RequestException as e:
        return f"Request failed: {type(e).__name__}: {e}", ""


def _metrics(status_json, out, seconds, started) -> str:
    cold = (status_json.get("delayTime") or 0) / 1000
    exec_s = (status_json.get("executionTime") or 0) / 1000
    wall = time.time() - started
    note = " (worker was cold)" if cold > 20 else ""
    level = ""
    if out.get("audio_rms") is not None:
        level = f" · rms {out['audio_rms']:.4f} / peak {out.get('audio_peak', 0):.3f}"
    return (f"**{out.get('latency_ms')} ms** · audio {out.get('duration_s', seconds):.2f}s · "
            f"inference **{exec_s:.2f}s** · queue/cold {cold:.1f}s{note} · "
            f"total {wall:.1f}s · `{out.get('device', '?')}`{level}")


with gr.Blocks(title="Haitian Creole streaming ASR") as demo:
    gr.Markdown(
        "# 🎙️ Haitian Creole streaming ASR\n"
        "[`jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht`]"
        "(https://huggingface.co/jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht) — a 0.6B "
        "streaming FastConformer RNN-T fine-tuned on 45.9 h of Kreyòl.\n\n"
        "**15.7% WER** at 320 ms on a 385-clip speaker-disjoint test set; the base model, "
        "which has no Creole slot, scores 98.2%.\n\n"
        "_Inference runs on a RunPod Serverless worker. The first request after an idle "
        "period takes ~2 minutes while a worker starts; subsequent ones are ~2 s._"
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                            label="Creole audio")
        latency = gr.Dropdown(list(LATENCIES), value="320 ms — 15.7% WER",
                              label="Streaming latency",
                              info="Less lookahead is faster but less accurate.")
    go = gr.Button("Transcribe", variant="primary")
    out_text = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
    metrics = gr.Markdown()

    go.click(transcribe, [audio_in, latency], [out_text, metrics])

    gr.Markdown(
        "---\n**Haitian Creole only.** Fine-tuning on Creole-only data moved the shared "
        "encoder and decoder, so this model does not retain the base model's other "
        "languages — prompting it with `fr-FR` or `en-US` returns Creole. Use "
        "[`nvidia/nemotron-3.5-asr-streaming-0.6b`]"
        "(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for anything else."
    )

if __name__ == "__main__":
    demo.launch()
