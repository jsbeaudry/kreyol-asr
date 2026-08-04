"""RunPod Serverless handler — Haitian Creole streaming ASR.

Request:
    {"input": {"audio_base64": "<base64 wav/flac/mp3>", "latency_ms": 320}}
    {"input": {"audio_url": "https://.../clip.wav"}}

Response:
    {"text": "...", "latency_ms": 320, "att_context_size": [56, 3],
     "duration_s": 6.1, "lang": "ht-HT"}

The model is restored to CUDA at import, not per request: a serverless worker
owns its GPU for its whole lifetime, so the only thing a warm invocation pays
for is decode.
"""

import base64
import binascii
import io
import json
import os
import tempfile
import urllib.request

import numpy as np
import runpod
import soundfile as sf
import soxr
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

SAMPLE_RATE = 16000
LANG = os.environ.get("TARGET_LANG", "ht-HT")
# v2 is PRIVATE, unlike v1 — HF_TOKEN must be present at image build (the Dockerfile
# pre-downloads the checkpoint) and is safest to keep at runtime too, since
# hf_hub_download still revalidates the revision against the Hub.
MODEL_REPO = os.environ.get("MODEL_REPO", "jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht-v2")
MODEL_FILE = os.environ.get("MODEL_FILE", "nemotron-3.5-asr-streaming-0.6b-ht-v2.nemo")

# Right context -> latency. Left context is ALWAYS 56: it matches the encoder's
# sliding_window of 57. Any other value degrades the streaming behaviour the
# checkpoint was trained for.
LEFT_CONTEXT = 56
LATENCY_TO_RIGHT = {80: 0, 320: 3, 560: 6, 1120: 13}
DEFAULT_LATENCY_MS = 320
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS", "120"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_path = hf_hub_download(MODEL_REPO, MODEL_FILE, token=os.environ.get("HF_TOKEN") or None)
MODEL = ASRModel.restore_from(_path, map_location=DEVICE)
MODEL.eval()
MODEL.freeze()
print(f"[init] {MODEL_REPO} restored on {DEVICE}, lang={LANG}", flush=True)


def _fetch(inp: dict) -> bytes:
    if inp.get("audio_base64"):
        raw = inp["audio_base64"]
        if "," in raw[:64] and raw[:5] == "data:":  # data: URI
            raw = raw.split(",", 1)[1]
        try:
            return base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"audio_base64 is not valid base64: {e}") from e
    url = inp.get("audio_url")
    if not url:
        raise ValueError("provide either `audio_base64` or `audio_url`")
    if not url.startswith(("http://", "https://")):
        raise ValueError("audio_url must be http(s)")
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - scheme checked
        return r.read()


def _to_wav16k(blob: bytes) -> tuple[str, float, float, float, float, float]:
    """Decode whatever was sent to 16 kHz mono PCM16 on disk."""
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not decode audio: {e}") from e
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.size == 0:
        raise ValueError("audio is empty")
    if sr != SAMPLE_RATE:
        # Band-limited resampling; naive interpolation aliases into the mel bins.
        arr = soxr.resample(arr, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    # Peak-normalise quiet input. Training audio was clean, well-levelled read
    # speech peaking near full scale; a laptop mic often lands 20-30 dB below
    # that, and an RNN-T decoder facing an out-of-distribution level can emit
    # nothing at all rather than a bad guess. Applied only when there is real
    # signal, so silence stays silence.
    gain = 1.0
    raw_peak = float(np.abs(arr).max())
    if 1e-4 < raw_peak < 0.5:
        gain = 0.95 / raw_peak
        arr = arr * gain

    seconds = len(arr) / SAMPLE_RATE
    if seconds < 0.3:
        raise ValueError(f"audio is only {seconds:.2f}s; too short to transcribe")
    if seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"audio is {seconds:.1f}s; limit is {MAX_AUDIO_SECONDS:.0f}s")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, np.clip(arr, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    return (tmp.name, seconds, float(np.sqrt((arr ** 2).mean())),
            float(np.abs(arr).max()), round(gain, 2), round(raw_peak, 5))


def _manifest(wav: str, seconds: float) -> str:
    """One-line manifest carrying `lang` — the only way to select the prompt slot.

    `transcribe(..., target_lang=...)` appears to work and does set `default_lang`
    on the dataloader config, but the Lhotse dataset ignores that key. It reads
    `cut.supervisions[0].language`, which comes from the manifest's `lang` field.
    Passing bare file paths makes NeMo write a manifest without one, and decoding
    fails with "Unknown prompt key: 'None'".
    """
    mf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    mf.write(json.dumps({"audio_filepath": wav, "duration": seconds, "text": "",
                         "lang": LANG, "target_lang": LANG}) + "\n")
    mf.close()
    return mf.name


def _right_context(inp: dict) -> tuple[int, int]:
    if "right_context" in inp:
        right = int(inp["right_context"])
        if right not in LATENCY_TO_RIGHT.values():
            raise ValueError(f"right_context must be one of {sorted(LATENCY_TO_RIGHT.values())}")
        ms = next(k for k, v in LATENCY_TO_RIGHT.items() if v == right)
        return right, ms
    ms = int(inp.get("latency_ms", DEFAULT_LATENCY_MS))
    if ms not in LATENCY_TO_RIGHT:
        raise ValueError(f"latency_ms must be one of {sorted(LATENCY_TO_RIGHT)}")
    return LATENCY_TO_RIGHT[ms], ms


def handler(job):
    inp = (job or {}).get("input") or {}
    try:
        right, latency_ms = _right_context(inp)
        wav, seconds, rms, peak, gain, raw_peak = _to_wav16k(_fetch(inp))
    except ValueError as e:
        return {"error": str(e)}

    try:
        MODEL.encoder.set_default_att_context_size([LEFT_CONTEXT, right])
        with torch.inference_mode():
            out = MODEL.transcribe(_manifest(wav, seconds), batch_size=1, verbose=False)
    except Exception as e:  # noqa: BLE001 - returned to the caller
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass

    # `Hypothesis.text` is "" when nothing was decoded, and "" is falsy — an
    # `or str(hyp)` fallback here dumps the whole Hypothesis repr into the
    # response and hides the real problem, which is that no tokens came out.
    text = ""
    if out:
        first = out[0]
        text = first if isinstance(first, str) else getattr(first, "text", "")
        if text is None:
            text = ""

    result = {"text": text, "lang": LANG, "latency_ms": latency_ms,
              "att_context_size": [LEFT_CONTEXT, right],
              "duration_s": round(seconds, 3), "device": DEVICE,
              "audio_rms": round(rms, 5), "audio_peak": round(peak, 5),
              "input_peak": raw_peak, "gain_applied": gain}
    if not text.strip():
        result["note"] = (
            "no speech decoded — audio is silent or near-silent"
            if rms < 0.005 else
            f"no speech decoded despite signal (input peak {raw_peak:.3f}, "
            f"gain x{gain:.1f} applied). Likely too quiet, too noisy, or not "
            f"Creole speech — try speaking closer to the mic."
        )
    return result


runpod.serverless.start({"handler": handler})
