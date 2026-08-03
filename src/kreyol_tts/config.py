"""Typed view over configs/tts.ht.yaml.

Separate from `kreyol_asr.config` because the filters barely overlap: the ASR pipeline
cares about transcript charset and RNN-T memory, the TTS pipeline about bandwidth,
loudness, clipping and silence. `Source` is shared, since a dataset repo is a dataset
repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kreyol_asr.config import Source, load_yaml

DEFAULT_QUALITY = {
    "tier_a_min_e8k": 5.0e-3,
    "tier_b_min_e8k": 5.0e-4,
    "min_native_sr": 22050,
    "stage1_tiers": ["A", "B", "C"],
    "stage2_tiers": ["A", "B"],
}

DEFAULT_FILTERS = {
    "min_duration": 1.0,       # StyleTTS2 alignment is unreliable below ~1s
    "max_duration": 12.0,      # mel frames = duration * 80
    "min_phonemes": 5,
    "max_phonemes": 400,       # PLBERT max_position_embeddings is 512; leave headroom
    "max_phonemes_per_second": 22,
    "max_clip_ratio": 0.005,   # fraction of samples at |x| >= 0.999
    "max_dc_offset": 5.0e-3,
    "min_snr_db": 15.0,
}

DEFAULT_AUDIO = {
    "highpass_hz": 60,
    "target_lufs": -23.0,      # EBU R128
    "peak_ceiling_db": -1.0,
    "resample_quality": "VHQ",  # HQ suffices for ASR features; a vocoder target wants VHQ
    "trim_top_db": 35.0,
    "lead_silence_ms": 50,
    "tail_silence_ms": 150,
}

DEFAULT_TEXT = {
    "lowercase": True,         # opposite of the ASR config — case is not phonetic
    "strip_bracketed": True,
    "normalize_apostrophes": True,
    "expand_numbers": True,
}

DEFAULT_STAGE2 = {"min_clips": 300, "max_clips": 900}


@dataclass
class TTSDataConfig:
    language: str = "ht"
    sample_rate: int = 24000
    sources: list[Source] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    stage2: dict[str, Any] = field(default_factory=dict)
    val_utterances: int = 400
    seed: int = 1804
    output_dir: Path = Path("data/ht_tts")

    @classmethod
    def load(cls, path: str | Path) -> "TTSDataConfig":
        raw = load_yaml(path)
        sources = [Source(**s) for s in raw.get("sources", [])]
        if not sources:
            raise ValueError(f"{path}: no `sources` listed — nothing to prepare.")
        sr = int(raw.get("sample_rate", 24000))
        if sr != 24000:
            raise ValueError(
                f"{path}: sample_rate is {sr}, but Kokoro's iSTFTNet stack is "
                f"upsample_rates [10,6] x gen_istft_hop_size 5 = 300x, which equals "
                f"StyleTTS 2's hop_length only at 24000. Changing it invalidates every "
                f"pretrained decoder filter — i.e. the entire point of warm-starting — "
                f"and the stock `kokoro` package hardcodes 24 kHz output regardless."
            )
        return cls(
            language=raw.get("language", "ht"),
            sample_rate=sr,
            sources=sources,
            quality=DEFAULT_QUALITY | raw.get("quality", {}),
            filters=DEFAULT_FILTERS | raw.get("filters", {}),
            audio=DEFAULT_AUDIO | raw.get("audio", {}),
            text=DEFAULT_TEXT | raw.get("text", {}),
            stage2=DEFAULT_STAGE2 | raw.get("stage2", {}),
            val_utterances=int(raw.get("val_utterances", 400)),
            seed=int(raw.get("seed", 1804)),
            output_dir=Path(raw.get("output_dir", "data/ht_tts")),
        )
