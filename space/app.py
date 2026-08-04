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
import re
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

# Right context -> latency, with the WER measured on a 332-clip speaker-disjoint
# test set. Left context is fixed at 56 by the worker and is not selectable.
LATENCIES = {
    "80 ms — 11.9% WER": 80,
    "320 ms — 10.1% WER": 320,
    "560 ms — 10.3% WER": 560,
    "1120 ms — 10.1% WER": 1120,
}

VOICES = ["nana", "deniz", "mako", "mariz", "klodin", "jan", "job", "leo"]
# 200, not higher: the model silently truncates past ~250 characters and fails
# outright at 600, so a larger segment would quietly drop the end of the text.
MAX_CHARS = 200        # the worker's per-request limit, mirrored here
MAX_TOTAL_CHARS = 2000  # guard on one submission: every segment is a billed call
SEGMENT_GAP_S = 0.15    # silence joined between segments, so sentences breathe

# Sentence enders, keeping the punctuation with the sentence it closes.
_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")

# A cold worker pulls >10 GB and imports NeMo before it can serve anything.
POLL_SECONDS = 2
MAX_WAIT_SECONDS = 240


class WorkerError(RuntimeError):
    """A message already fit to show the user."""


def _run_and_poll(base, payload, started, progress, label=""):
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
                     desc=(f"{label}{state.lower().replace('_', ' ')} — {waited}s"
                           + (" (cold start takes ~2 min)" if waited > 15 else "")))
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

def _hard_wrap(unit: str, limit: int) -> list[str]:
    """Last resort for a clause with no punctuation left to break on."""
    out, line = [], ""
    for word in unit.split(" "):
        if len(word) > limit:              # one unsplittable token
            if line:
                out.append(line)
                line = ""
            out.extend(word[i:i + limit] for i in range(0, len(word), limit))
        elif not line:
            line = word
        elif len(line) + 1 + len(word) <= limit:
            line = f"{line} {word}"
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def _break_down(unit: str, limit: int) -> list[str]:
    """A single sentence longer than the worker allows, split at clause
    punctuation first so the cut lands somewhere a speaker would pause."""
    if len(unit) <= limit:
        return [unit]
    out, buf = [], ""
    for part in _CLAUSE_END.split(unit):
        if len(part) > limit:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_hard_wrap(part, limit))
        elif not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= limit:
            buf = f"{buf} {part}"
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def _split_for_tts(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Cut text into pieces that each fit one TTS request.

    Sentence boundaries first, since that is where a natural pause already is.
    Consecutive sentences are then packed back together up to the limit, because
    every piece is a separate billed round trip.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_break_down(sentence, limit))
    packed = []
    for piece in pieces:
        if packed and len(packed[-1]) + 1 + len(piece) <= limit:
            packed[-1] = f"{packed[-1]} {piece}"
        else:
            packed.append(piece)
    return packed


def synthesize(text, voice, temperature, top_p, repetition_penalty,
               progress=gr.Progress()):
    text = (text or "").strip()
    if not text:
        return None, "Type some Creole text first."
    if len(text) > MAX_TOTAL_CHARS:
        return None, (f"That is {len(text)} characters; this Space sends at most "
                      f"{MAX_TOTAL_CHARS} in one go.")

    segments = _split_for_tts(text)
    if not segments:
        return None, "Nothing to say."

    started = time.time()
    chunks, sample_rate, exec_total, cold_total = [], None, 0.0, 0.0
    used, truncated_segments = {}, []
    for i, segment in enumerate(segments, 1):
        payload = {"input": {"text": segment, "voice": voice,
                             "temperature": temperature, "top_p": top_p,
                             "repetition_penalty": repetition_penalty}}
        label = f"segment {i}/{len(segments)} · " if len(segments) > 1 else ""
        try:
            status_json, out = _run_and_poll(TTS_BASE, payload, started, progress,
                                             label=label)
        except WorkerError as e:
            done = f" {i - 1} of {len(segments)} segments had already succeeded." \
                   if i > 1 else ""
            return None, f"Segment {i} of {len(segments)} failed — {e}{done}"

        b64 = out.get("audio_base64")
        if not b64:
            return None, f"Segment {i} of {len(segments)} came back with no audio."
        try:
            data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
        except Exception as e:  # noqa: BLE001
            return None, (f"Could not decode segment {i} of {len(segments)}: "
                          f"{type(e).__name__}: {e}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Never seen from this model, but concatenating mismatched rates
            # would silently change the pitch of part of the clip.
            return None, (f"Segment {i} came back at {sr} Hz but segment 1 was "
                          f"{sample_rate} Hz; refusing to join them.")
        chunks.append(data)
        if out.get("truncated"):
            # The worker still returns audio, but the end of the text was not
            # spoken. Saying so beats handing back a clip that stops mid-thought.
            truncated_segments.append(i)
        exec_total += (status_json.get("executionTime") or 0) / 1000
        cold_total += (status_json.get("delayTime") or 0) / 1000
        used = {k: out[k] for k in ("temperature", "top_p", "repetition_penalty")
                if out.get(k) is not None} or used

    if len(chunks) > 1:
        gap = np.zeros(int(SEGMENT_GAP_S * sample_rate), dtype=np.float32)
        joined = [chunks[0]]
        for c in chunks[1:]:
            joined.extend((gap, c))
        audio = np.concatenate(joined)
    else:
        audio = chunks[0]

    # Voice levels vary a lot between speakers, so show the peak rather than
    # leaving a quiet result looking like a failure.
    peak = float(np.abs(audio).max() or 0.0)
    knobs = " · ".join(f"{k} {v:g}" for k, v in used.items())
    wall = time.time() - started
    note = " (worker was cold)" if cold_total > 20 else ""
    seg_note = (f"**{len(segments)} segments** · " if len(segments) > 1 else "")
    warning = ""
    if truncated_segments:
        which = ", ".join(str(i) for i in truncated_segments)
        warning = (f"\n\n⚠️ **Segment {which} looks truncated** — the model stopped "
                   f"before the end of that text. Try shorter sentences.")
    metrics = (f"**{voice}** · {seg_note}{len(audio) / sample_rate:.2f}s @ "
               f"{sample_rate} Hz · inference **{exec_total:.2f}s** · "
               f"queue/cold {cold_total:.1f}s{note} · total {wall:.1f}s · "
               f"peak {peak:.3f}"
               + (f"\n\n<sub>{knobs}</sub>" if knobs else "") + warning)
    return (sample_rate, audio), metrics


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
                " — a 0.6B streaming FastConformer RNN-T fine-tuned on 58.8 h of "
                "Kreyòl. **10.1% WER** at 320 ms on a 332-clip speaker-disjoint "
                "test set; the base model, which has no Creole slot, scores 98.6%."
            )
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                    label="Creole audio")
                latency = gr.Dropdown(list(LATENCIES), value="320 ms — 10.1% WER",
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
                    label="Creole text", lines=6, max_lines=14,
                    placeholder="Bonjou, koman ou ye? Mwen kontan tande ou jodi a.",
                    info=f"Up to {MAX_TOTAL_CHARS} characters. Longer text is "
                         f"split at sentence boundaries into {MAX_CHARS}-character "
                         f"segments, synthesised one at a time and joined back "
                         f"into a single clip.")
                voice = gr.Dropdown(VOICES, value="nana", label="Voice")
            with gr.Accordion("Generation settings", open=False):
                gr.Markdown(
                    "_KaniTTS exposes only these three per call — `max_new_tokens` "
                    "is fixed when the model loads. Values outside a slider's range "
                    "are clamped by the worker, which reports back what it used._")
                with gr.Row():
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05,
                                            label="Temperature",
                                            info="Higher is more varied, less stable.")
                    top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.01,
                                      label="Top-p",
                                      info="Nucleus sampling cutoff.")
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05,
                                                   label="Repetition penalty",
                                                   info="Higher discourages loops "
                                                        "and stuck syllables.")
                reset = gr.Button("Reset to defaults", size="sm")
                reset.click(lambda: (1.0, 0.95, 1.1), None,
                            [temperature, top_p, repetition_penalty])
            tts_go = gr.Button("Generate speech", variant="primary")
            out_audio = gr.Audio(label="Generated speech", type="numpy",
                                 autoplay=False, show_download_button=True)
            tts_metrics = gr.Markdown()
            tts_go.click(synthesize,
                         [tts_text, voice, temperature, top_p, repetition_penalty],
                         [out_audio, tts_metrics])

            gr.Examples(
                [["Bonjou, koman ou ye?", "nana"],
                 ["Mwen kontan tande ou jodi a.", "jan"],
                 ["Ayiti se yon peyi ki gen anpil istwa.", "mariz"]],
                inputs=[tts_text, voice],
            )
            gr.Markdown(
                "_Level varies between speakers — some voices are noticeably "
                "quieter than others. The peak is reported above so a quiet "
                "result is not mistaken for a failed one._\n\n"
                "_Each segment is a separate call to the worker, so a long "
                "passage takes roughly (segments × ~2 s) once a worker is warm._"
            )

if __name__ == "__main__":
    # ssr_mode=False: with Gradio 5.49's SSR renderer the Accordion, its Sliders
    # and the Examples dataset were served in /config but never mounted in the
    # DOM, and the node server logged 405s alongside "Too little data for
    # declared Content-Length". Client-side rendering puts them back.
    demo.launch(ssr_mode=False)
