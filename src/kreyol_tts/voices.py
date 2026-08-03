"""Named voices, and which audio each is allowed to be built from.

Speaker ids are opaque strings chosen by the TTS vendor; the names are ours. The
source of truth for the mapping is `serverless-tts/handler.py:VOICES` — keep in sync.

Bandwidth is a per-(source, speaker) property, not a per-source one, which is the
single most important thing the Phase 0 audit found. `m1-collect`'s source-level
median E>8k is 8.0e-07, which reads as uniformly hopeless — but that is `nana`'s 80
upsampled clips dragging it down. Split by speaker, the same repo is bimodal:

    klodin  1.74e-03  Tier B      nana   4.17e-08  Tier C  (398,031x below Kokoro)
    deniz   1.22e-03  Tier B      mariz  1.83e-07  Tier C
                                  mako   4.43e-06  Tier C

So Stage 2 eligibility is decided per voice against its *best* source, never against
the repo it happens to live in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Voice:
    name: str
    speaker_id: str
    gender: str
    stage2: bool                      # eligible for a voicepack in v1
    note: str = ""
    # When set, Stage 2 draws only from these source slugs. Everything else the
    # speaker appears in still feeds Stage 1.
    stage2_sources: list[str] = field(default_factory=list)


# Measured 2026-08-02 (benchmarks/tts/audit.md), n=27-185 clips per speaker.
VOICES: dict[str, Voice] = {
    "deniz": Voice("deniz", "0047599005d8", "f", True,
                   "E>8k 1.71e-03 in learn-the-numbers (10x below Kokoro), Tier B",
                   ["jsbeaudry__learn-the-numbers", "jsbeaudry__m1-collect",
                    "jsbeaudry__justice-in-creole"]),
    "klodin": Voice("klodin", "25d65d04313e", "f", True,
                    "E>8k 1.74e-03 in m1-collect (10x below Kokoro), Tier B",
                    ["jsbeaudry__m1-collect"]),
    "jan": Voice("jan", "49db5343dd8a", "m", True,
                 "E>8k 6.96e-04 in learn-the-numbers (24x below Kokoro), Tier B",
                 ["jsbeaudry__learn-the-numbers"]),
    "mariz": Voice("mariz", "121adceef217", "f", True,
                   "E>8k 6.76e-04 in learn-the-numbers (25x below). Her m1-collect "
                   "and finance-in-creole audio is Tier C and must not be used.",
                   ["jsbeaudry__learn-the-numbers"]),

    # --- Stage 1 only ------------------------------------------------------
    "nana": Voice("nana", "2ed43a0fa899", "f", False,
                  "NO wideband audio anywhere: m1-collect 4.17e-08, media-social-2 "
                  "4.06e-08, media-social 8.64e-08 — all ~400,000x below Kokoro, all "
                  "upsampled upstream. She is the current production DEFAULT_VOICE, so "
                  "shipping without her is a product decision, not a training one."),
    "mako": Voice("mako", "3939afe3ea20", "m", False,
                  "m1-collect only, E>8k 4.43e-06 (3,751x below Kokoro), Tier C"),
    "job": Voice("job", "Job", "m", False,
                 "bible-chapters-data only, E>8k 1.75e-04 (95x below), Tier C"),
    "leo": Voice("leo", "Leo", "m", False,
                 "history-data only, natively 16 kHz with E>8k exactly 0.00 — a 24 kHz "
                 "model cannot be built from it and no resampler adds the band back."),
}

VOICE_BY_SPEAKER_ID = {v.speaker_id: name for name, v in VOICES.items()}
# The ids are case-inconsistent in the corpus ("Job"/"job", "Leo"/"leo").
VOICE_BY_SPEAKER_ID.update({v.speaker_id.lower(): name for name, v in VOICES.items()})

# Kokoro's convention is <lang><gender>_<name>. `h` is already Hindi, so Kreyòl takes
# `k`. Purely a naming choice — confirm before publishing, nothing depends on it.
LANG_PREFIX = "k"


def pack_name(voice: str) -> str:
    v = VOICES[voice]
    return f"{LANG_PREFIX}{v.gender}_{v.name}"


def shipping_voices() -> list[Voice]:
    return [v for v in VOICES.values() if v.stage2]
