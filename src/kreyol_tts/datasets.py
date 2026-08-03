"""Hugging Face dataset repos in, a 24 kHz corpus and StyleTTS 2 manifests out.

Output layout (under `output_dir`):
    wav/<source-slug>/<idx>.wav          24 kHz mono PCM16
    manifests/stage1_{train,val}.txt     path|phonemes|speaker_index
    manifests/stage2_<voice>_{train,val}.txt
    manifests/ood_texts.txt              phoneme-only lines StyleTTS 2 samples from
    speakers.json                        voice -> index, clips, hours, tier
    data_report.md / prepare_stats.json

Differs from the ASR prepare pass in three ways that matter:

* 24 kHz, not 16 — and bandwidth is measured *before* resampling, because a 24 kHz
  header says nothing about content.
* Tiering is per (source, speaker). `m1-collect`'s source median is Tier C, but
  `klodin` and `deniz` inside it are Tier B. Aggregating by source loses two voices.
* **Val is held out by utterance, not by speaker** — the exact opposite of
  `kreyol_asr.datasets._split_real`. Stage 1 val must contain the same speakers as
  train or the mel loss is meaningless.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from kreyol_common import console
from kreyol_common.audio import _decode, to_mono

from . import dsp
from .config import TTSDataConfig
from .g2p import PhonemeCoverage, g2p
from .voices import VOICE_BY_SPEAKER_ID, VOICES


def tier_of(native_sr: int, e8k: float, q: dict[str, Any]) -> str:
    if native_sr < q["min_native_sr"]:
        return "D"
    if e8k >= q["tier_a_min_e8k"]:
        return "A"
    if e8k >= q["tier_b_min_e8k"]:
        return "B"
    return "C"


def _process(raw_audio, cfg: TTSDataConfig) -> tuple[np.ndarray, dsp.ClipMeasurement] | None:
    """Decode and condition one clip. Returns None if it cannot be decoded."""
    decoded = _decode(raw_audio)
    if decoded is None:
        return None
    arr, sr = decoded
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1 if x.shape[0] > x.shape[1] else 0)
    if len(x) == 0 or sr <= 0:
        return None

    a = cfg.audio
    x = dsp.highpass(x, sr, a["highpass_hz"])
    m = dsp.measure(x, sr)                      # BEFORE resampling — see module docstring
    x = to_mono(x, sr, cfg.sample_rate, quality=a["resample_quality"])
    x = dsp.trim_silence(x, cfg.sample_rate, a["trim_top_db"])
    x = dsp.pad_silence(x, cfg.sample_rate, a["lead_silence_ms"], a["tail_silence_ms"])
    x = dsp.loudness_normalize(x, cfg.sample_rate, a["target_lufs"], a["peak_ceiling_db"])
    return x, m


def prepare(cfg: TTSDataConfig, token: str | None = None,
            limit: int | None = None) -> dict[str, Any]:
    from kreyol_common.sources import _iter_source

    out = cfg.output_dir
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    f, q = cfg.filters, cfg.quality
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    drops: Counter[str] = Counter()
    coverage = PhonemeCoverage()

    for src in cfg.sources:
        wav_dir = out / "wav" / src.slug
        wav_dir.mkdir(parents=True, exist_ok=True)
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TextColumn("{task.completed}"), TimeElapsedColumn(),
                      console=console) as bar:
            task = bar.add_task(f"  {src.repo_id}", total=None)
            for item in _iter_source(src, token, limit):
                bar.advance(task)
                try:
                    processed = _process(item["audio"], cfg)
                except Exception:  # noqa: BLE001 - one bad clip must not kill the run
                    processed = None
                if processed is None:
                    drops["undecodable_audio"] += 1
                    continue
                x, m = processed

                if m.clip_ratio > f["max_clip_ratio"]:
                    drops["clipped"] += 1
                    continue
                if m.dc_offset > f["max_dc_offset"]:
                    drops["dc_offset"] += 1
                    continue

                duration = len(x) / cfg.sample_rate
                if duration < f["min_duration"]:
                    drops["too_short"] += 1
                    continue
                if duration > f["max_duration"]:
                    drops["too_long"] += 1
                    continue
                if dsp.snr_db(x, cfg.sample_rate) < f["min_snr_db"]:
                    drops["low_snr"] += 1
                    continue

                phonemes = g2p(item["raw_text"] or "", keep_punct=True)
                bare = phonemes.replace(" ", "")
                if len(bare) < f["min_phonemes"]:
                    drops["too_few_phonemes"] += 1
                    continue
                if len(bare) > f["max_phonemes"]:
                    drops["too_many_phonemes"] += 1
                    continue
                # A transcript nobody could have spoken in the time available. The ASR
                # pipeline learned this the hard way from machine-labelled repetition
                # loops ("pou pou pou ..."); the same sources feed this one.
                if duration and len(bare) / duration > f["max_phonemes_per_second"]:
                    drops["implausible_phoneme_rate"] += 1
                    continue

                key = hashlib.md5(x.tobytes()).hexdigest() + "|" + bare
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)

                speaker = item["speaker"] or "<none>"
                voice = VOICE_BY_SPEAKER_ID.get(speaker) or \
                    VOICE_BY_SPEAKER_ID.get(speaker.lower())
                path = wav_dir / f"{item['index']:07d}.wav"
                sf.write(path, x, cfg.sample_rate, subtype="PCM_16")
                coverage.counts.update(c for c in phonemes if not c.isspace())
                coverage.texts += 1

                records.append({
                    "path": str(path.resolve()), "phonemes": phonemes,
                    "duration": round(duration, 3), "source": src.slug,
                    "speaker": speaker, "voice": voice, "synthetic": src.synthetic,
                    "native_sr": m.native_sr, "e8k": m.e8k, "f999": m.f999,
                    "tier": tier_of(m.native_sr, m.e8k, q),
                })

    if not records:
        raise RuntimeError("No usable clips survived filtering — check the drop counts.")

    stats = _emit(cfg, records, drops, coverage)
    console.print(f"\n[green]Wrote[/green] {out/'manifests'} and {out/'data_report.md'}")
    return stats


def _speaker_tier(records: list[dict], speaker: str, q: dict) -> str:
    """Best tier available to a speaker, computed per (source, speaker).

    Aggregating by source would mark `klodin` and `deniz` Tier C because `nana`'s
    upsampled clips dominate `m1-collect`'s median. Two shippable voices lost.
    """
    best = "D"
    by_src: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["speaker"] == speaker:
            by_src[r["source"]].append(r["e8k"])
    for src, es in by_src.items():
        srs = [r["native_sr"] for r in records
               if r["speaker"] == speaker and r["source"] == src]
        t = tier_of(int(np.median(srs)), float(np.median(es)), q)
        best = min(best, t)          # "A" < "B" < "C" < "D"
    return best


def _emit(cfg: TTSDataConfig, records: list[dict], drops: Counter,
          coverage: PhonemeCoverage) -> dict[str, Any]:
    out, q = cfg.output_dir, cfg.quality
    rng = random.Random(cfg.seed)

    speakers = sorted({r["speaker"] for r in records})
    index_of = {s: i for i, s in enumerate(speakers)}
    tiers = {s: _speaker_tier(records, s, q) for s in speakers}

    stage1 = [r for r in records if tiers[r["speaker"]] in q["stage1_tiers"]
              or tiers[r["speaker"]] == "D"]
    # Val holds out utterances, NOT speakers — the opposite of the ASR splitter. It is
    # also drawn from real audio only, so mel loss is not flattered by the easy
    # synthetic distribution.
    real = [r for r in stage1 if not r["synthetic"]]
    rng.shuffle(real)
    # Cap val at a tenth of the real audio as well as at the absolute target. Without
    # the fraction, a --limit run hands every real clip to val and trains on nothing
    # but the synthetic remainder — which still "succeeds" and writes a plausible
    # report. At full scale the absolute target binds and this is a no-op.
    n_val = min(cfg.val_utterances, max(1, len(real) // 10))
    val = real[:n_val]
    val_paths = {r["path"] for r in val}
    train = [r for r in stage1 if r["path"] not in val_paths]

    def write(path: Path, rows: list[dict], speaker_index) -> None:
        with open(path, "w") as fh:
            for r in rows:
                fh.write(f"{r['path']}|{r['phonemes']}|{speaker_index(r)}\n")

    m = out / "manifests"
    write(m / "stage1_train.txt", train, lambda r: index_of[r["speaker"]])
    write(m / "stage1_val.txt", val, lambda r: index_of[r["speaker"]])
    # StyleTTS 2 samples OOD phoneme lines for the SLM adversarial objective.
    with open(m / "ood_texts.txt", "w") as fh:
        for r in rng.sample(train, min(len(train), 2000)):
            fh.write(f"{r['path']}|{r['phonemes']}|0\n")

    per_voice: dict[str, Any] = {}
    for name, v in VOICES.items():
        mine = [r for r in records if r["voice"] == name]
        eligible = [r for r in mine
                    if r["tier"] in q["stage2_tiers"]
                    and (not v.stage2_sources or r["source"] in v.stage2_sources)]
        rng.shuffle(eligible)
        capped = eligible[:cfg.stage2["max_clips"]]
        ready = v.stage2 and len(capped) >= cfg.stage2["min_clips"]
        if capped:
            n_val = max(10, len(capped) // 20)
            write(m / f"stage2_{name}_val.txt", capped[:n_val], lambda r: 0)
            write(m / f"stage2_{name}_train.txt", capped[n_val:], lambda r: 0)
        per_voice[name] = {
            "speaker_id": v.speaker_id, "gender": v.gender,
            "index": index_of.get(v.speaker_id, index_of.get(v.speaker_id.lower())),
            "clips_total": len(mine), "clips_stage2": len(capped),
            "hours_total": round(sum(r["duration"] for r in mine) / 3600, 3),
            "tier": tiers.get(v.speaker_id) or tiers.get(v.speaker_id.lower()) or "?",
            "stage2_ready": bool(ready),
            "status": "PASS" if ready else ("THIN" if v.stage2 else "BLOCKED"),
            "note": v.note,
        }

    (out / "speakers.json").write_text(json.dumps(per_voice, indent=2, ensure_ascii=False))

    per_source = {}
    for slug in sorted({r["source"] for r in records}):
        rs = [r for r in records if r["source"] == slug]
        per_source[slug] = {
            "clips": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 3),
            "median_e8k": float(np.median([r["e8k"] for r in rs])),
            "native_sr": Counter(r["native_sr"] for r in rs).most_common(1)[0][0],
            "tier": tier_of(int(np.median([r["native_sr"] for r in rs])),
                            float(np.median([r["e8k"] for r in rs])), q),
        }

    stats = {
        "language": cfg.language, "sample_rate": cfg.sample_rate,
        "total_clips": len(records),
        "total_hours": round(sum(r["duration"] for r in records) / 3600, 3),
        "stage1": {"train": len(train), "val": len(val)},
        "speakers": len(speakers), "per_source": per_source, "voices": per_voice,
        "dropped": dict(drops), "phoneme_coverage": coverage.report(),
    }
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "data_report.md").write_text(_render(stats))
    return stats


def _render(s: dict[str, Any]) -> str:
    cov = s["phoneme_coverage"]
    L = [
        "# TTS data report", "",
        f"- {s['total_hours']} h across {s['total_clips']} clips at {s['sample_rate']} Hz",
        f"- Stage 1: {s['stage1']['train']} train / {s['stage1']['val']} val "
        f"({s['speakers']} speakers)", "",
        "> Val holds out **utterances, not speakers** — Stage 1 val must contain the same",
        "> speakers as train or the mel loss is meaningless. This is deliberately the",
        "> opposite of the ASR splitter's `speaker_disjoint: true`; do not \"fix\" it.", "",
        "## Voices", "",
        "| voice | tier | clips | Stage 2 clips | hours | status |",
        "|---|:--:|---:|---:|---:|:--:|",
    ]
    for name, v in s["voices"].items():
        L.append(f"| {name} | **{v['tier']}** | {v['clips_total']} | {v['clips_stage2']} | "
                 f"{v['hours_total']} | **{v['status']}** |")
    L += ["", "Notes:", ""] + [f"- `{n}`: {v['note']}" for n, v in s["voices"].items() if v["note"]]

    L += ["", "## Sources", "", "| source | clips | hours | native SR | median E>8k | tier |",
          "|---|---:|---:|---:|---:|:--:|"]
    for slug, v in s["per_source"].items():
        L.append(f"| `{slug}` | {v['clips']} | {v['hours']} | {v['native_sr']} | "
                 f"{v['median_e8k']:.1e} | **{v['tier']}** |")

    L += ["", "## Dropped", ""] + ([f"- {k}: {v}" for k, v in s["dropped"].items()] or ["- none"])
    L += ["", "## Phoneme coverage", "",
          f"- {cov['distinct_phonemes']} distinct phonemes over {cov['texts']} texts",
          f"- Verdict: **{cov['verdict']}**"]
    if cov["out_of_vocab"]:
        L.append(f"- OUT OF VOCAB (hard stop): `{list(cov['out_of_vocab'])}`")
    if cov["rare_phonemes"]:
        L.append(f"- Rare (<50, present but undertrained): `{cov['rare_phonemes']}`")
    return "\n".join(L) + "\n"
