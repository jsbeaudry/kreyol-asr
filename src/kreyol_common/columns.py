"""Column auto-detection, so a new dataset repo needs no config in the common case."""

from __future__ import annotations

AUDIO_CANDIDATES = ["audio", "wav", "speech", "audio_filepath", "file", "path", "sound"]
TEXT_CANDIDATES = ["text", "transcription", "transcript", "sentence", "normalized_text",
                   "target", "label", "caption", "content"]
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "client_id", "spk_id", "spk", "session_id"]


def _pick(explicit: str | None, columns: list[str], candidates: list[str], kind: str,
          required: bool = True) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{kind} column {explicit!r} not in dataset columns {columns}")
        return explicit
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    if required:
        raise ValueError(
            f"Could not auto-detect the {kind} column among {columns}. "
            f"Set `{kind}_column:` explicitly in the dataset config."
        )
    return None
