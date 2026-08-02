"""Haitian Creole speech — thin client over two RunPod Serverless endpoints.

Neither model runs here. This Space converts audio, calls a worker, and renders
the result. That keeps the build to a few light wheels instead of a 15-minute
NeMo-from-source install, and means the Space needs no GPU at all.

ASR and TTS are separate endpoints because they cannot share a Python
environment: kani-tts pins nemo-toolkit[tts]==2.4.0 while the ASR model needs
nemo_toolkit[asr] from git main.

Requires, under Settings -> Variables and secrets:
    RUNPOD_API_KEY       (secret)   your RunPod key — must be scoped to BOTH
                                    endpoints, or the other tab returns 403
    RUNPOD_ENDPOINT      (variable) ASR endpoint id, e.g. 9fds364d4gicy0
    RUNPOD_TTS_ENDPOINT  (variable) TTS endpoint id, e.g. 90fnsmvwgqfl6y
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
ASR_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT", "9fds364d4gicy0")
TTS_ENDPOINT = os.environ.get("RUNPOD_TTS_ENDPOINT", "90fnsmvwgqfl6y")
ASR_BASE = f"https://api.runpod.ai/v2/{ASR_ENDPOINT}"
TTS_BASE = f"https://api.runpod.ai/v2/{TTS_ENDPOINT}"

# Right context -> latency, with the WER measured on a 385-clip speaker-disjoint
# test set. Left context is fixed at 56 by the worker and is not selectable.
LATENCIES = {
    "80 ms — 17.3% WER": 80,
    "320 ms — 15.7% WER": 320,
    "560 ms — 14.9% WER": 560,
    "1120 ms — 14.7% WER": 1120,
}

VOICES = ["nana", "deniz", "mako", "mariz", "klodin", "jan", "job", "leo"]
MAX_CHARS = 600  # mirrors the worker's own limit

# A cold worker pulls >10 GB and imports NeMo before it can serve anything.
POLL_SECONDS = 2
MAX_WAIT_SECONDS = 240


class WorkerError(RuntimeError):
    """A message already fit to show the user."""


def _run_and_poll(base, payload, started, progress):
    """Submit to /run and poll /status. Both workers can cold-start for minutes,
    which a synchronous /runsync call will not sit through."""
    if not API_KEY:
        raise WorkerError("RUNPOD_API_KEY is not set. Add it under Settings → "
                          "Variables and secrets, then restart the Space.")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(f"{base}/run", headers=headers, json=payload, timeout=60)
        if r.status_code == 403:
            # A restricted RunPod key is scoped to specific endpoints; one that
            # works for the other tab can still be forbidden here.
            raise WorkerError(
                f"403 from RunPod for endpoint `{base.rsplit('/', 1)[-1]}`. The API "
                "key is likely a restricted key scoped to other endpoints — add "
                "this one to its scope in the RunPod console, or use an "
                "unrestricted key.")
        r.raise_for_status()
        job = r.json().get("id")
        if not job:
            raise WorkerError(f"No job id in response: {r.text[:200]}")

        deadline = time.time() + MAX_WAIT_SECONDS
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            s = requests.get(f"{base}/status/{job}", headers=headers, timeout=30).json()
            state = s.get("status", "?")
            if state == "COMPLETED":
                out = s.get("output") or {}
                if out.get("error"):
                    raise WorkerError(f"Worker error: {out['error']}")
                return s, out
            if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise WorkerError(f"{state}: {str(s.get('error') or s.get('output'))[:300]}")
            waited = int(time.time() - started)
            progress(min(waited / MAX_WAIT_SECONDS, 0.95),
                     desc=f"{state.lower().replace('_', ' ')} — {waited}s"
                          + (" (cold start takes ~2 min)" if waited > 15 else ""))
        raise WorkerError(f"Timed out after {MAX_WAIT_SECONDS}s.")
    except requests.RequestException as e:
        raise WorkerError(f"Request failed: {type(e).__name__}: {e}") from e


def _timing(status_json, started, extra="") -> str:
    cold = (status_json.get("delayTime") or 0) / 1000
    exec_s = (status_json.get("executionTime") or 0) / 1000
    wall = time.time() - started
    note = " (worker was cold)" if cold > 20 else ""
    return (f"inference **{exec_s:.2f}s** · queue/cold {cold:.1f}s{note} · "
            f"total {wall:.1f}s{extra}")


# --- ASR ---------------------------------------------------------------------

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
    started = time.time()
    try:
        progress(0, desc="Converting to 16 kHz WAV")
        b64, seconds = _wav16k_base64(audio)
    except Exception as e:  # noqa: BLE001
        return f"Could not read that audio: {type(e).__name__}: {e}", ""

    payload = {"input": {"audio_base64": b64, "latency_ms": LATENCIES[latency_label]}}
    try:
        status_json, out = _run_and_poll(ASR_BASE, payload, started, progress)
    except WorkerError as e:
        return str(e), ""

    level = ""
    if out.get("audio_rms") is not None:
        level = f" · rms {out['audio_rms']:.4f} / peak {out.get('audio_peak', 0):.3f}"
    if out.get("gain_applied"):
        level += f" · gain ×{out['gain_applied']:.1f}"
    metrics = (f"**{out.get('latency_ms')} ms** · audio "
               f"{out.get('duration_s', seconds):.2f}s · "
               + _timing(status_json, started,
                         f" · `{out.get('device', '?')}`{level}"))

    text = out.get("text") or ""
    if not text.strip():
        return f"(nothing transcribed) — {out.get('note', 'no speech decoded')}", metrics
    return text, metrics


# --- TTS ---------------------------------------------------------------------

def synthesize(text, voice, progress=gr.Progress()):
    text = (text or "").strip()
    if not text:
        return None, "Type some Creole text first."
    if len(text) > MAX_CHARS:
        return None, f"That is {len(text)} characters; the worker's limit is {MAX_CHARS}."

    started = time.time()
    payload = {"input": {"text": text, "voice": voice}}
    try:
        status_json, out = _run_and_poll(TTS_BASE, payload, started, progress)
    except WorkerError as e:
        return None, str(e)

    b64 = out.get("audio_base64")
    if not b64:
        return None, "The worker returned no audio."
    try:
        data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    except Exception as e:  # noqa: BLE001
        return None, f"Could not decode the returned audio: {type(e).__name__}: {e}"

    # Voice levels vary a lot between speakers, so show the peak rather than
    # leaving a quiet result looking like a failure.
    peak = float(np.abs(data).max() or 0.0)
    metrics = (f"**{voice}** · {out.get('duration_s', len(data) / sr):.2f}s @ "
               f"{out.get('sample_rate', sr)} Hz · "
               + _timing(status_json, started,
                         f" · `{out.get('device', '?')}` · peak {peak:.3f}"))
    return (sr, data), metrics


# --- UI ----------------------------------------------------------------------

with gr.Blocks(title="Haitian Creole speech") as demo:
    gr.Markdown(
        "# 🇭🇹 Haitian Creole speech\n"
        "Speech-to-text and text-to-speech for Kreyòl. Both models run on RunPod "
        "Serverless workers — the first request after an idle period takes ~1–2 "
        "minutes while a worker starts; subsequent ones are seconds."
    )

    with gr.Tabs():
        with gr.Tab("🎙️ Speech to text"):
            gr.Markdown(
                "[`jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht`]"
                "(https://huggingface.co/jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht)"
                " — a 0.6B streaming FastConformer RNN-T fine-tuned on 45.9 h of "
                "Kreyòl. **15.7% WER** at 320 ms on a 385-clip speaker-disjoint "
                "test set; the base model, which has no Creole slot, scores 98.2%."
            )
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                    label="Creole audio")
                latency = gr.Dropdown(list(LATENCIES), value="320 ms — 15.7% WER",
                                      label="Streaming latency",
                                      info="Less lookahead is faster but less accurate.")
            asr_go = gr.Button("Transcribe", variant="primary")
            out_text = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
            asr_metrics = gr.Markdown()
            asr_go.click(transcribe, [audio_in, latency], [out_text, asr_metrics])

            gr.Markdown(
                "**Haitian Creole only.** Fine-tuning on Creole-only data moved the "
                "shared encoder and decoder, so this model does not retain the base "
                "model's other languages — prompting it with `fr-FR` or `en-US` "
                "returns Creole. Use [`nvidia/nemotron-3.5-asr-streaming-0.6b`]"
                "(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) "
                "for anything else."
            )

        with gr.Tab("🔊 Text to speech"):
            gr.Markdown(
                "[`jsbeaudry/haitian-kani-ht-v3`]"
                "(https://huggingface.co/jsbeaudry/haitian-kani-ht-v3) — a KaniTTS "
                "voice model for Kreyòl, with eight speakers."
            )
            with gr.Row():
                tts_text = gr.Textbox(
                    label="Creole text", lines=4, max_lines=8,
                    placeholder="Bonjou, koman ou ye?",
                    info=f"Up to {MAX_CHARS} characters.")
                voice = gr.Dropdown(VOICES, value="nana", label="Voice")
            tts_go = gr.Button("Generate speech", variant="primary")
            out_audio = gr.Audio(label="Generated speech", type="numpy",
                                 autoplay=False, show_download_button=True)
            tts_metrics = gr.Markdown()
            tts_go.click(synthesize, [tts_text, voice], [out_audio, tts_metrics])

            gr.Examples(
                [["Bonjou, koman ou ye?", "nana"],
                 ["Mwen kontan tande ou jodi a.", "jan"],
                 ["Ayiti se yon peyi ki gen anpil istwa.", "mariz"]],
                inputs=[tts_text, voice],
            )
            gr.Markdown(
                "_Level varies between speakers — some voices are noticeably "
                "quieter than others. The peak is reported above so a quiet "
                "result is not mistaken for a failed one._"
            )

if __name__ == "__main__":
    demo.launch()
