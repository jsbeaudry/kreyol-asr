"""RunPod Serverless handler — Haitian Creole TTS (KaniTTS).

Request:
    {"input": {"text": "Bonjou, koman ou ye?", "voice": "nana"}}
    {"input": {"text": "...", "voice_id": "2ed43a0fa899"}}   # raw id

Response:
    {"audio_base64": "<wav>", "sample_rate": 22050, "duration_s": 1.8,
     "voice": "nana", "text": "Bonjou, koman ou ye?"}

Separate from the ASR worker on purpose: kani-tts pins nemo-toolkit[tts]==2.4.0
while the ASR model needs nemo_toolkit[asr] from git main. They cannot share an
environment.
"""

import base64
import os
import tempfile

import runpod
import soundfile as sf
import torch

MODEL_REPO = os.environ.get("MODEL_REPO", "jsbeaudry/haitian-kani-ht-v3")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))

# KaniTTS selects a speaker by prefixing the prompt with an opaque id. Exposing
# names keeps that detail out of every caller.
VOICES = {
    "nana":   "2ed43a0fa899",
    "deniz":  "0047599005d8",
    "mako":   "3939afe3ea20",
    "mariz":  "121adceef217",
    "klodin": "25d65d04313e",
    "jan":    "49db5343dd8a",
    "job":    "job",
    "leo":    "leo",
}
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "nana")

from kani_tts import KaniTTS  # noqa: E402 - after env setup

MODEL = KaniTTS(MODEL_REPO)
print(f"[init] {MODEL_REPO} loaded | cuda={torch.cuda.is_available()} "
      f"| voices={sorted(VOICES)}", flush=True)


def _resolve_voice(inp: dict) -> tuple[str, str]:
    """Return (display_name, speaker_id)."""
    if inp.get("voice_id"):
        vid = str(inp["voice_id"])
        name = next((n for n, i in VOICES.items() if i == vid), vid)
        return name, vid
    name = str(inp.get("voice", DEFAULT_VOICE)).lower()
    if name not in VOICES:
        raise ValueError(f"unknown voice {name!r}; choose from {sorted(VOICES)} "
                         f"or pass voice_id")
    return name, VOICES[name]


def handler(job):
    inp = (job or {}).get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "provide `text`"}
    if len(text) > MAX_CHARS:
        return {"error": f"text is {len(text)} chars; limit is {MAX_CHARS}"}

    try:
        voice, speaker_id = _resolve_voice(inp)
    except ValueError as e:
        return {"error": str(e)}

    try:
        audio, spoken = MODEL(f"{speaker_id}:{text}")
    except Exception as e:  # noqa: BLE001 - returned to the caller
        return {"error": f"{type(e).__name__}: {e}"}

    # save_audio owns the sample rate and array layout, so let it write the file
    # rather than guessing either here.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        MODEL.save_audio(audio, tmp.name)
        with open(tmp.name, "rb") as fh:
            blob = fh.read()
        info = sf.info(tmp.name)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not encode audio: {type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # The model echoes back the whole prompt, speaker id and all. Returning that
    # verbatim leaks the id the `voice` names exist to hide.
    spoken = spoken if isinstance(spoken, str) else text
    prefix = f"{speaker_id}:"
    if spoken.startswith(prefix):
        spoken = spoken[len(prefix):]

    return {
        "audio_base64": base64.b64encode(blob).decode(),
        "sample_rate": info.samplerate,
        "duration_s": round(info.duration, 3),
        "voice": voice,
        "voice_id": speaker_id,
        "text": spoken,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


runpod.serverless.start({"handler": handler})
