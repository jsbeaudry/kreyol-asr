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
    # Exactly one of `repo_id` (Hugging Face) or `local_dir` (an audiofolder on
    # disk). Local sources exist because Radio Haiti-Inter is 6 GB of Zenodo zips
    # that get segmented on the training pod — round-tripping the slices through
    # a private Hub repo would cost an upload per gate iteration and buy nothing.
    repo_id: str | None = None
    local_dir: str | None = None
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
    # Real recordings, machine-generated transcripts. Kept out of val/test for a
    # different reason than `synthetic`: the audio is fine, the *labels* are
    # another model's output, so scoring against them measures agreement with
    # that model rather than accuracy. Counted separately in the data report —
    # lumping the two under one number hides which risk you are carrying.
    pseudo_labeled: bool = False

    def __post_init__(self) -> None:
        if bool(self.repo_id) == bool(self.local_dir):
            raise ValueError(
                "Source needs exactly one of `repo_id` or `local_dir` "
                f"(got repo_id={self.repo_id!r}, local_dir={self.local_dir!r})"
            )

    @property
    def train_only(self) -> bool:
        """Trains, never evaluates — for either reason."""
        return self.synthetic or self.pseudo_labeled

    @property
    def kind(self) -> str:
        if self.pseudo_labeled:
            return "pseudo-labeled"
        return "synthetic" if self.synthetic else "real"

    @property
    def slug(self) -> str:
        base = (self.repo_id or Path(self.local_dir).name).replace("/", "__")
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
