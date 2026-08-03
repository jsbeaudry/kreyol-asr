"""Per-clip conditioning for a vocoder training target.

None of this has an ASR analogue, and two steps matter more than they look:

* **Consistent silence padding.** StyleTTS 2's duration predictor learns whatever
  silence distribution it is fed. Inconsistent leading silence means inconsistent
  onset delay at inference — and latency is the entire point of this project.
* **Bandwidth measurement before resampling.** A 24 kHz header says nothing about
  content. Measuring after a resample would destroy the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BANDWIDTH_SPLIT_HZ = 8000.0   # Nyquist of 16 kHz audio
_NFFT = 2048


@dataclass
class ClipMeasurement:
    native_sr: int
    e8k: float
    f999: float
    peak: float
    clip_ratio: float
    dc_offset: float


def measure(x: np.ndarray, sr: int) -> ClipMeasurement:
    """Characterise a clip at its native rate, before any resampling."""
    absx = np.abs(x)
    e8k, f999 = spectral(x, sr)
    return ClipMeasurement(
        native_sr=int(sr), e8k=e8k, f999=f999,
        peak=float(absx.max()) if len(absx) else 0.0,
        clip_ratio=float((absx >= 0.999).mean()) if len(absx) else 0.0,
        dc_offset=float(abs(x.mean())) if len(x) else 0.0,
    )


def spectral(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(fraction of energy above 8 kHz, 99.9% rolloff Hz).

    Averaged periodogram, not one transform over the whole clip: a single FFT is
    dominated by whichever moment was loudest, and we are characterising the
    recording chain, not one syllable.
    """
    n = len(x)
    if n < 256:
        return 0.0, 0.0
    nfft = min(_NFFT, 1 << int(np.floor(np.log2(n))))
    win = np.hanning(nfft)
    hop = nfft // 2
    frames = [x[i:i + nfft] * win for i in range(0, n - nfft + 1, hop)]
    if not frames:
        frames = [np.pad(x, (0, nfft - n))[:nfft] * win]
    power = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    total = float(power.sum())
    if total <= 0:
        return 0.0, 0.0
    e8k = float(power[freqs > BANDWIDTH_SPLIT_HZ].sum() / total)
    cum = np.cumsum(power) / total
    f999 = float(freqs[min(int(np.searchsorted(cum, 0.999)), len(freqs) - 1)])
    return e8k, f999


def highpass(x: np.ndarray, sr: int, hz: float = 60.0) -> np.ndarray:
    """Butterworth order-4 high-pass. Removes DC offset and rumble the vocoder
    would otherwise learn to reproduce. `cmu` fails a DC-offset gate without it."""
    from scipy.signal import butter, filtfilt

    if hz <= 0:
        return x
    b, a = butter(4, hz / (sr / 2.0), btype="highpass")
    return filtfilt(b, a, x).astype(np.float32)


def _frame_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    idx = range(0, len(x) - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2)) for i in idx])
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def trim_silence(x: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    """Drop leading/trailing frames more than `top_db` below the loudest frame.

    Hand-rolled rather than librosa.effects.trim so the TTS extra stays installable
    on macOS arm64 without dragging in numba.
    """
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    db = _frame_db(x, frame, hop)
    if not len(db):
        return x
    keep = np.flatnonzero(db > db.max() - top_db)
    if not len(keep):
        return x
    return x[max(0, keep[0] * hop): min(len(x), keep[-1] * hop + frame)]


def pad_silence(x: np.ndarray, sr: int, lead_ms: float, tail_ms: float) -> np.ndarray:
    """Re-pad to a fixed window so every clip has the same onset delay."""
    return np.concatenate([
        np.zeros(int(sr * lead_ms / 1000.0), dtype=np.float32),
        x.astype(np.float32),
        np.zeros(int(sr * tail_ms / 1000.0), dtype=np.float32),
    ])


def snr_db(x: np.ndarray, sr: int) -> float:
    """Reference-free SNR estimate: loud-frame energy over quiet-frame energy.

    A percentile ratio rather than WADA — transparent, dependency-free, and adequate
    for a reject/keep gate. Reported as an estimate, never as a calibrated figure.
    """
    db = _frame_db(x, int(0.025 * sr), int(0.010 * sr))
    if len(db) < 10:
        return 0.0
    return float(np.percentile(db, 95) - np.percentile(db, 5))


def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float,
                       peak_ceiling_db: float) -> np.ndarray:
    """EBU R128 loudness normalisation, then a true-peak ceiling.

    Order matters: normalising after limiting would undo the ceiling.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(x)
    except Exception:  # noqa: BLE001 - clip too short to measure; leave it alone
        loudness = None
    if loudness is not None and np.isfinite(loudness):
        x = x * (10.0 ** ((target_lufs - loudness) / 20.0))
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > ceiling:
        x = x * (ceiling / peak)
    return np.clip(x, -1.0, 1.0).astype(np.float32)
