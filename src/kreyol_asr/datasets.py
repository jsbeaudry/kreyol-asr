"""Hugging Face dataset URLs in, NeMo manifests out.

Output layout (under `output_dir`):
    wav/<source-slug>/<idx>.wav      16 kHz mono PCM16
    manifests/{train,val,test}.json  NeMo JSON-lines
    data_report.md                   hours, drops, tokenizer coverage, style check
    prepare_stats.json               same numbers, machine-readable
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

from . import SAMPLE_RATE
from .config import DataConfig
from .text import TokenizerCoverage, normalize, out_of_charset, style_stats

# Moved to `kreyol_common` when the TTS pipeline needed the same Hub access, and
# re-exported here so every existing import and test keeps working unchanged.
# The Hub helpers in particular encode three specific, expensively-learned failure
# modes; two copies would mean the next one gets fixed in only one of them.
from kreyol_common import console  # noqa: F401
from kreyol_common.audio import AUDIO_EXT, _decode, to_mono  # noqa: F401
from kreyol_common.columns import (AUDIO_CANDIDATES, SPEAKER_CANDIDATES,  # noqa: F401
                                   TEXT_CANDIDATES, _pick)
from kreyol_common.hub import (_explain_hub_error, _hf_home_problem,  # noqa: F401
                               _probe_repo_access)
from kreyol_common.sources import (_is_audiofolder, _iter_audiofolder,  # noqa: F401
                                   _iter_source)


def _to_mono_16k(array: np.ndarray, sr: int) -> np.ndarray:
    """ASR resamples to 16 kHz; TTS calls `to_mono` with 24000 instead."""
    return to_mono(array, sr, SAMPLE_RATE)


def prepare(cfg: DataConfig, base_model: str, token: str | None = None,
            limit: int | None = None) -> dict[str, Any]:
    out = cfg.output_dir
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    coverage = TokenizerCoverage(base_model, token=token)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    drops: Counter[str] = Counter()
    per_source_sec: defaultdict[str, float] = defaultdict(float)
    weights: dict[str, float] = {}

    min_d = float(cfg.filters["min_duration"])
    max_d = float(cfg.filters["max_duration"])
    max_cps = float(cfg.filters.get("max_chars_per_second", 25.0))

    for src in cfg.sources:
        weights[src.slug] = src.weight
        wav_dir = out / "wav" / src.slug
        wav_dir.mkdir(parents=True, exist_ok=True)

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TextColumn("{task.completed}"), TimeElapsedColumn(),
                      console=console) as bar:
            task = bar.add_task(f"  {src.slug}", total=None)
            for item in _iter_source(src, token, limit):
                bar.advance(task)
                try:
                    decoded = _decode(item["audio"])
                except Exception:
                    decoded = None
                if decoded is None or len(decoded[0]) == 0:
                    drops["undecodable_audio"] += 1
                    continue

                text = normalize(
                    item["raw_text"] or "",
                    lowercase=cfg.text["lowercase"],
                    strip_bracketed=cfg.text["strip_bracketed"],
                    normalize_apostrophes=cfg.text["normalize_apostrophes"],
                )
                if cfg.text.get("sanitize_to_vocab", True):
                    # Never let a character the BPE can't encode reach the manifest:
                    # it would become an <unk> training target.
                    text = coverage.sanitize(text)
                if cfg.filters["drop_empty_text"] and not text.strip():
                    drops["empty_text"] += 1
                    continue

                oov = out_of_charset(text)
                if oov and cfg.filters["drop_out_of_charset"]:
                    drops["out_of_charset"] += 1
                    continue

                wav = _to_mono_16k(*decoded)
                duration = len(wav) / SAMPLE_RATE  # measured, never trusted from metadata
                if duration < min_d:
                    drops["too_short"] += 1
                    continue
                if duration > max_d:
                    drops["too_long"] += 1
                    continue

                # Reject transcripts nobody could have spoken in the time available.
                # Machine-labelled sources degenerate into repetition loops
                # ("pou pou pou pou ..."), which hit 600+ chars/sec against a normal
                # 12-16. They are poison twice over: they teach the model to emit
                # loops, and a long target against short audio makes the RNNT lattice
                # explode — a single clip cost 7 GiB and OOM-killed a training run.
                cps = len(text) / duration if duration else float("inf")
                if cps > max_cps:
                    drops["implausible_chars_per_second"] += 1
                    continue

                key = hashlib.md5(wav.tobytes()).hexdigest() + "|" + text.lower()
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)

                path = wav_dir / f"{item['index']:07d}.wav"
                sf.write(path, wav, SAMPLE_RATE, subtype="PCM_16")

                coverage.add(text)
                per_source_sec[src.slug] += duration
                records.append({
                    "audio_filepath": str(path.resolve()),
                    "duration": round(duration, 3),
                    "text": text,
                    "lang": cfg.language,
                    "target_lang": cfg.language,
                    "_source": src.slug,
                    "_speaker": item["speaker"],
                    "_synthetic": src.synthetic,
                    "_pseudo": src.pseudo_labeled,
                    "_train_only": src.train_only,
                })

    if not records:
        raise RuntimeError("No usable clips survived filtering — check the drop counts above.")

    splits = _split(records, cfg.split)
    counts = {}
    for name, rows in splits.items():
        # Duplicate weighted sources in train only; val/test stay unweighted so
        # metrics reflect the real distribution.
        emit = _apply_weights(rows, weights) if name == "train" else rows
        path = out / "manifests" / f"{name}.json"
        with open(path, "w") as f:
            for r in emit:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")}, ensure_ascii=False) + "\n")
        counts[name] = {"clips": len(emit),
                        "hours": round(sum(r["duration"] for r in emit) / 3600, 3)}

    cov = coverage.report()
    stats = {
        "language": cfg.language,
        "total_clips": len(records),
        "total_hours": round(sum(r["duration"] for r in records) / 3600, 3),
        "per_source_hours": {k: round(v / 3600, 3) for k, v in per_source_sec.items()},
        "splits": counts,
        "dropped": dict(drops),
        "split_mode": _split_real.last_mode,
        "synthetic_hours": round(_split.synthetic_hours, 3),
        "pseudo_labeled_hours": round(_split.pseudo_hours, 3),
        "real_hours": round(_split.real_hours, 3),
        "tokenizer_coverage": cov,
        "tokenizer_verdict": TokenizerCoverage.verdict(cov),
        "style": style_stats([r["text"] for r in records]),
    }
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "data_report.md").write_text(_render_report(stats))
    console.print(f"\n[green]Wrote[/green] {out/'manifests'} and {out/'data_report.md'}")
    return stats


def _apply_weights(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    emitted: list[dict] = []
    for r in rows:
        w = weights.get(r["_source"], 1.0)
        emitted.extend([r] * max(1, int(round(w))))
    return emitted


def _split(records: list[dict], spec: dict[str, Any]) -> dict[str, list[dict]]:
    """Split, keeping train-only clips out of val and test.

    Two kinds of clip are held back, for two different reasons:

    - `synthetic`: TTS audio. A model scored on synthesized speech is scored on
      the narrow acoustic distribution it was trained on, reporting a WER it will
      not reproduce on real recordings.
    - `pseudo_labeled`: real audio, machine-generated transcripts. Scoring against
      another model's output measures agreement with that model, not accuracy —
      and here the labelling model is plausibly the weaker of the two.

    Both train; only human-labelled real recordings evaluate.
    """
    # `_train_only` is derived, so fall back to the flags it derives from — a
    # record built before this key existed must still be held back, not leak
    # into test because a bookkeeping field was missing.
    def train_only(r: dict) -> bool:
        return bool(r.get("_train_only", r.get("_synthetic") or r.get("_pseudo")))

    held = [r for r in records if train_only(r)]
    real = [r for r in records if not train_only(r)]

    if held and not real:
        raise RuntimeError(
            "Every source is marked `synthetic: true` or `pseudo_labeled: true`, so "
            "there is no real audio to evaluate on — val and test need human-labelled "
            "recordings. Add at least one such source, or drop the flag if these "
            "really are verified recordings."
        )

    sh = sum(r["duration"] for r in records if r.get("_synthetic")) / 3600
    ph = sum(r["duration"] for r in records if r.get("_pseudo")) / 3600
    rh = sum(r["duration"] for r in real) / 3600
    _split.synthetic_hours = sh
    _split.pseudo_hours = ph
    _split.real_hours = rh

    if not held:
        return _split_real(real, spec)

    out = _split_real(real, spec)
    out["train"].extend(held)
    parts = []
    if sh:
        parts.append(f"{sh:.2f} h synthetic")
    if ph:
        parts.append(f"{ph:.2f} h pseudo-labeled")
    console.print(f"Held {' + '.join(parts)} to train only; "
                  f"val/test drawn from {rh:.2f} h of human-labelled real audio")
    return out


def _split_real(records: list[dict], spec: dict[str, Any]) -> dict[str, list[dict]]:
    """Split by speaker group when possible, else by a deterministic per-clip hash."""
    rng = random.Random(spec.get("seed", 1804))
    ratios = {"train": spec["train"], "val": spec["val"], "test": spec["test"]}
    have_speakers = all(r["_speaker"] for r in records)

    if spec.get("speaker_disjoint") and have_speakers:
        _split_real.last_mode = "speaker-disjoint"
        groups: defaultdict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[r["_speaker"]].append(r)
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(r["duration"] for r in records)
        out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        # Fill val and test to their duration targets first, everything else trains.
        targets = {"val": ratios["val"] * total, "test": ratios["test"] * total}
        acc = {"val": 0.0, "test": 0.0}
        for k in keys:
            dur = sum(r["duration"] for r in groups[k])
            for name in ("test", "val"):
                if acc[name] + dur <= targets[name] * 1.15 and acc[name] < targets[name]:
                    out[name].extend(groups[k])
                    acc[name] += dur
                    break
            else:
                out["train"].extend(groups[k])
        if not out["val"] or not out["test"]:
            console.print("[yellow]Speaker groups too coarse for a clean split; "
                          "falling back to hash split.[/yellow]")
        else:
            return out

    _split_real.last_mode = "hash (no usable speaker column)" if not have_speakers else "hash"
    out = {"train": [], "val": [], "test": []}
    for r in records:
        h = int(hashlib.md5(r["audio_filepath"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < ratios["test"]:
            out["test"].append(r)
        elif h < ratios["test"] + ratios["val"]:
            out["val"].append(r)
        else:
            out["train"].append(r)
    return out


_split_real.last_mode = "unset"
_split.synthetic_hours = 0.0
_split.pseudo_hours = 0.0
_split.real_hours = 0.0


def _bucket_lines(s: dict[str, Any]) -> list[str]:
    """Report the three data buckets separately.

    Synthetic and pseudo-labeled are both train-only, but they carry different
    risks — narrow acoustics versus noisy labels — and one combined number would
    hide which one you are carrying.
    """
    synth, pseudo = s.get("synthetic_hours") or 0, s.get("pseudo_labeled_hours") or 0
    if not synth and not pseudo:
        return ["- All sources are real recordings with human transcripts"]
    lines = [f"- Real, human-labelled: **{s['real_hours']} h** (trains and evaluates)"]
    if synth:
        lines.append(f"- Synthetic (TTS): **{synth} h** — trains only. Scoring on "
                     f"synthesized speech reports a WER real recordings will not reproduce.")
    if pseudo:
        lines.append(f"- Pseudo-labeled: **{pseudo} h** — trains only. Real audio, "
                     f"machine-generated transcripts; scoring against them would measure "
                     f"agreement with the labelling model, not accuracy.")
    return lines


def _render_report(s: dict[str, Any]) -> str:
    cov = s["tokenizer_coverage"]
    style = s["style"]
    lines = [
        "# Data report",
        "",
        f"- Language tag: `{s['language']}`",
        f"- Total: **{s['total_hours']} h** across **{s['total_clips']}** clips",
        f"- Split mode: {s['split_mode']}",
        *_bucket_lines(s),
        "",
        "## Splits",
        "",
        "| split | clips | hours |",
        "|---|---:|---:|",
    ]
    for name, v in s["splits"].items():
        lines.append(f"| {name} | {v['clips']} | {v['hours']} |")
    lines += ["", "## Hours per source", "", "| source | hours |", "|---|---:|"]
    for k, v in s["per_source_hours"].items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Dropped clips", ""]
    lines += [f"- {k}: {v}" for k, v in s["dropped"].items()] or ["- none"]
    lines += [
        "",
        "## Tokenizer coverage (pretrained BPE, vocab 13088)",
        "",
        f"- tokens/word: **{cov['tokens_per_word']}**",
        f"- `<unk>` rate: **{cov['unk_rate']}** ({cov['unk_tokens']} tokens)",
        f"- Verdict: **{s['tokenizer_verdict']}**",
        "",
        f"- Substituted (vocab can't encode it, mapped to nearest): "
        f"`{cov.get('substituted_chars', {})}`",
        f"- Dropped (vocab can't encode it, no fallback): `{cov.get('dropped_chars', {})}`",
        f"- Out-of-charset characters: `{cov['out_of_charset_chars']}`",
        f"- `<unk>` pieces: `{cov['unk_pieces']}`",
        "",
        "## Transcript style vs base model",
        "",
        f"- Clips containing uppercase: **{style['cased_ratio']:.1%}**",
        f"- Clips containing punctuation: **{style['punctuated_ratio']:.1%}**",
        "",
    ]
    if style["cased_ratio"] < 0.2 or style["punctuated_ratio"] < 0.2:
        lines.append(
            "> **Style mismatch.** The base model was trained on punctuated, properly-cased "
            "text. This corpus is mostly lowercase/unpunctuated, so the fine-tuned model "
            "will emit that style too and the baseline WER comparison will be unfairly "
            "harsh on the base model. Either restyle the transcripts, or score both models "
            "with the same lowercase+no-punct normalization (`bench --normalize-scoring`)."
        )
    else:
        lines.append("> Style matches the base model's punctuated, cased output. Good.")
    return "\n".join(lines) + "\n"
