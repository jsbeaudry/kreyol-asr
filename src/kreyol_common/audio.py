"""Decoding and resampling, shared by the ASR (16 kHz) and TTS (24 kHz) pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def to_mono(array: np.ndarray, sr: int, target_sr: int, quality: str = "HQ") -> np.ndarray:
    """Mono-mix and resample to `target_sr`.

    Was `_to_mono_16k`, with 16 kHz baked in — an ASR choice, not a property of the
    data. Kokoro synthesizes at 24 kHz, so the rate is now the caller's business.

    `quality` is soxr's: HQ is ample for an ASR feature extractor, VHQ is worth the
    cost when the output is a vocoder training target.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:  # (samples, channels) or (channels, samples)
        arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[1] else 0)
    if sr != target_sr:
        import soxr  # band-limited resampling; np.interp would alias into the mel bins

        arr = soxr.resample(arr, sr, target_sr, quality=quality).astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def _decode(entry: Any) -> tuple[np.ndarray, int] | None:
    """Decode one audio cell to (samples, sample_rate). Never resamples.

    We decode with soundfile rather than letting `datasets` do it: datasets>=4
    routes decoding through torchcodec (which needs ffmpeg), while datasets<4
    returns a plain array dict. Handling the raw bytes ourselves works on both
    and keeps resampling under our control.
    """
    import io

    if entry is None:
        return None
    # datasets<4 with decode=True
    if isinstance(entry, dict) and entry.get("array") is not None:
        return np.asarray(entry["array"]), int(entry.get("sampling_rate") or 0) or 16000
    # decode=False -> {"path": ..., "bytes": ...}
    if isinstance(entry, dict):
        if entry.get("bytes"):
            data, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32", always_2d=False)
            return data, sr
        if entry.get("path"):
            data, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
            return data, sr
        return None
    if isinstance(entry, (str, Path)):
        data, sr = sf.read(str(entry), dtype="float32", always_2d=False)
        return data, sr
    # datasets>=4 torchcodec AudioDecoder, if it is installed after all
    if hasattr(entry, "get_all_samples"):
        s = entry.get_all_samples()
        return np.asarray(s.data).squeeze(), int(s.sample_rate)
    return None
