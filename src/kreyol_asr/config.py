"""Typed views over the two YAML config files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class Source:
    repo_id: str
    split: str = "train"
    config: str | None = None
    audio_column: str | None = None
    text_column: str | None = None
    speaker_column: str | None = None
    weight: float = 1.0
    # TTS-generated speech. Useful as training augmentation, but it has narrow
    # speaker/channel diversity, so it is kept out of val and test — scoring a
    # model on its own synthesis flatters the WER and hides real-world failure.
    synthetic: bool = False

    @property
    def slug(self) -> str:
        base = self.repo_id.replace("/", "__")
        return f"{base}__{self.config}" if self.config else base


@dataclass
class DataConfig:
    language: str = "ht-HT"
    sources: list[Source] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = Path("data/ht")

    @classmethod
    def load(cls, path: str | Path) -> DataConfig:
        raw = load_yaml(path)
        sources = [Source(**s) for s in raw.get("sources", [])]
        if not sources:
            raise ValueError(f"{path}: no `sources` listed — nothing to prepare.")
        return cls(
            language=raw.get("language", "ht-HT"),
            sources=sources,
            split={"train": 0.95, "val": 0.025, "test": 0.025,
                   "speaker_disjoint": True, "seed": 1804} | raw.get("split", {}),
            filters={"min_duration": 0.5, "max_duration": 30.0,
                     "drop_empty_text": True, "drop_out_of_charset": False,
                     "max_chars_per_second": 25.0}
                    | raw.get("filters", {}),
            text={"lowercase": False, "strip_bracketed": True,
                  "normalize_apostrophes": True} | raw.get("text", {}),
            output_dir=Path(raw.get("output_dir", "data/ht")),
        )


@dataclass
class FinetuneConfig:
    base_model: str
    language: dict[str, Any]
    train: dict[str, Any]
    eval: dict[str, Any]
    publish: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> FinetuneConfig:
        raw = load_yaml(path)
        from . import BASE_MODEL

        return cls(
            base_model=raw.get("base_model", BASE_MODEL),
            language={"tag": "ht-HT", "slot": 105, "warm_start_from": "fr-FR"}
                     | raw.get("language", {}),
            train=raw.get("train", {}),
            eval=raw.get("eval", {}),
            publish=raw.get("publish", {}),
        )


def hf_token() -> str | None:
    """Token for private datasets / pushing. Also honours the standard HF vars."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    return None
