"""RunPod Serverless handler — Haitian Creole TTS (KaniTTS).

Request:
    {"input": {"text": "Bonjou, koman ou ye?", "voice": "nana"}}
    {"input": {"text": "...", "voice_id": "2ed43a0fa899"}}   # raw id
    {"input": {"text": "...", "temperature": 0.8, "top_p": 0.9,
               "repetition_penalty": 1.2}}                    # sampling knobs

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
# 200, not higher: the model silently truncates past ~250 characters (audio
# length plateaus near 16 s and the tail of the text is simply never spoken)
# and fails outright at 600 with "Special speech tokens not exist!".
MAX_CHARS = int(os.environ.get("MAX_CHARS", "200"))

# Truncation guard.
#
# The failure above is silent: generation stops, the request still succeeds, and
# short audio comes back as if it were complete. It shows up in the ratio of
# input characters to audio seconds, because the text keeps growing while the
# audio does not. Measured by round-tripping generated speech back through ASR:
#
#   clean      120-240 chars   11.6 - 14.9 chars/s   (transcript matched input)
#   truncated  260-450 chars   16.7 - 28.6 chars/s   (transcript lost the tail)
#
# 16.0 sits in the gap. This only annotates the response — the audio is still
# returned, because a caller may prefer partial speech to nothing, and normal
# speech rate varies with content.
CHARS_PER_SEC_LIMIT = float(os.environ.get("TRUNCATION_CHARS_PER_SEC", "16.0"))

# The only per-call generation knobs KaniTTS.__call__ takes; max_new_tokens is
# fixed at model construction, so it cannot be set per request.
# name -> (default, low, high)
GEN_PARAMS = {
    "temperature": (1.0, 0.1, 2.0),
    "top_p": (0.95, 0.05, 1.0),
    "repetition_penalty": (1.1, 1.0, 2.0),
}

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


def _resolve_gen(inp: dict) -> dict:
    """Clamp rather than reject: a slider a shade out of range should still make
    audio, and the effective value is reported back so it is not silently lost."""
    gen = {}
    for name, (default, low, high) in GEN_PARAMS.items():
        raw = inp.get(name, default)
        if raw is None:
            raw = default
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {raw!r}") from None
        if val != val:  # NaN survives float() and poisons generation
            raise ValueError(f"{name} must be a number, got {raw!r}")
        gen[name] = min(max(val, low), high)
    return gen


def handler(job):
    inp = (job or {}).get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "provide `text`"}
    if len(text) > MAX_CHARS:
        return {"error": f"text is {len(text)} chars; limit is {MAX_CHARS}"}

    try:
        voice, speaker_id = _resolve_voice(inp)
        gen = _resolve_gen(inp)
    except ValueError as e:
        return {"error": str(e)}

    try:
        audio, spoken = MODEL(f"{speaker_id}:{text}", **gen)
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

    chars_per_sec = (len(text) / info.duration) if info.duration > 0 else float("inf")
    truncated = chars_per_sec > CHARS_PER_SEC_LIMIT

    result = {
        "audio_base64": base64.b64encode(blob).decode(),
        "sample_rate": info.samplerate,
        "duration_s": round(info.duration, 3),
        "voice": voice,
        "voice_id": speaker_id,
        "text": spoken,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "chars_per_second": round(chars_per_sec, 2),
        "truncated": truncated,
        # Echo the values actually used, post-clamp.
        **gen,
    }
    if truncated:
        # Say what was measured, not just that something is wrong.
        result["note"] = (
            f"Audio looks truncated: {len(text)} characters in "
            f"{info.duration:.2f}s is {chars_per_sec:.1f} chars/s, above the "
            f"{CHARS_PER_SEC_LIMIT:.1f} threshold. The end of the text was "
            f"probably not spoken — send it in shorter pieces."
        )
        print(f"[truncation] {result['note']}", flush=True)
    return result


runpod.serverless.start({"handler": handler})
