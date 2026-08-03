"""Measure what the corpus actually contains, before spending anything on GPU.

The ASR pipeline resamples to 16 kHz, so it never had to care what the sources
natively were. Kokoro does: it synthesizes at 24 kHz, and a clip whose real content
stops at 8 kHz teaches the vocoder a spectral cliff no resampler can undo. A 24 kHz
WAV header is not evidence of 24 kHz content — this module measures the difference.

Per (source, speaker) it reports native sample rate, codec, the fraction of spectral
energy above 8 kHz, the 99.9% rolloff frequency, clipping, DC offset, duration, and
the resulting bandwidth tier. It also answers three questions the dataset config
raises but does not settle: exact row counts, which repos are duplicates of each
other, and which `speaker_id` values map onto the named TTS voices.

Reads parquet footers and a bounded stream of rows — never a full repo download.

    kreyol-tts audit --config configs/datasets.ht.yaml
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from rich.console import Console

# Phase 1 moves these into `kreyol_common`; importing them here keeps the Hub
# diagnostics and audiofolder handling in one place until then.
from kreyol_asr.datasets import _decode, _iter_source
from kreyol_tts import MIN_NATIVE_SR, TIER_A_MIN_E8K, TIER_B_MIN_E8K

console = Console()

# Speaker ids are opaque strings chosen by the TTS vendor; the names are ours.
# Source of truth is serverless-tts/handler.py:VOICES — keep them in sync.
VOICE_BY_SPEAKER_ID = {
    "2ed43a0fa899": "nana",
    "0047599005d8": "deniz",
    "3939afe3ea20": "mako",
    "121adceef217": "mariz",
    "25d65d04313e": "klodin",
    "49db5343dd8a": "jan",
    "job": "job",
    "leo": "leo",
}

# 8 kHz is the Nyquist of 16 kHz audio, so E>8k is the single number that separates
# "really wideband" from "narrowband in a wideband container".
BANDWIDTH_SPLIT_HZ = 8000.0
_NFFT = 2048

# Stock Kokoro's own 24 kHz output, measured with `_spectral` on af_heart
# (2026-08-02): median E>8k = 1.66e-2, f99.9 ~ 11.4 kHz. This is the distribution a
# fine-tune is being asked to reproduce, so it is the only honest yardstick — an
# absolute E>8k means little, but "40,000x below what the decoder currently emits"
# is a statement about how the result will actually sound.
#
# The measurement itself is validated against synthetic controls: a 1 kHz tone gives
# 8.9e-17, a 10 kHz tone 1.0, white noise 0.33 (theory: 4/12), and a speech-tilted
# signal round-tripped through 16 kHz collapses from 3.5e-2 to 1.1e-13 — which is
# the upsampling signature this module exists to detect.
KOKORO_REF_E8K = 1.66e-2


@dataclass
class ClipStats:
    native_sr: int
    fmt: str
    subtype: str
    duration: float
    e8k: float          # fraction of spectral energy above 8 kHz
    f999: float         # frequency below which 99.9% of energy lies
    peak: float
    clip_ratio: float   # fraction of samples at |x| >= 0.999
    dc_offset: float
    speaker: str | None
    audio_md5: str


@dataclass
class SourceReport:
    repo_id: str
    slug: str
    synthetic_flag: bool
    rows_total: int | None = None
    clips: list[ClipStats] = field(default_factory=list)
    error: str | None = None


def _spectral(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (fraction of energy above 8 kHz, 99.9% rolloff frequency).

    Averaged periodogram rather than a single FFT: one transform over a whole clip
    is dominated by whichever moment happened to be loudest, and we are trying to
    characterise the recording chain, not one syllable.
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
    # Above Nyquist/2 of a 16 kHz recording there is nothing to find; report 0
    # rather than a meaningless ratio over an empty band.
    e8k = float(power[freqs > BANDWIDTH_SPLIT_HZ].sum() / total)
    cumulative = np.cumsum(power) / total
    f999 = float(freqs[min(int(np.searchsorted(cumulative, 0.999)), len(freqs) - 1)])
    return e8k, f999


def _measure(raw: dict[str, Any], speaker: str | None) -> ClipStats | None:
    """Decode one clip and characterise it, without resampling anything."""
    blob = raw.get("bytes") if isinstance(raw, dict) else None
    md5 = hashlib.md5(blob).hexdigest() if blob else ""

    fmt = subtype = "?"
    if blob:
        try:
            import io
            info = sf.info(io.BytesIO(blob))
            fmt, subtype = info.format, info.subtype
        except Exception:  # noqa: BLE001 - format is nice to have, not required
            pass

    decoded = _decode(raw)
    if decoded is None:
        return None
    arr, sr = decoded
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1 if x.shape[0] > x.shape[1] else 0)
    if len(x) == 0 or sr <= 0:
        return None

    e8k, f999 = _spectral(x, sr)
    absx = np.abs(x)
    return ClipStats(
        native_sr=int(sr),
        fmt=fmt,
        subtype=subtype,
        duration=len(x) / sr,
        e8k=e8k,
        f999=f999,
        peak=float(absx.max()),
        clip_ratio=float((absx >= 0.999).mean()),
        dc_offset=float(abs(x.mean())),
        speaker=speaker,
        audio_md5=md5,
    )


def tier_of(native_sr: int, e8k: float) -> str:
    """Bandwidth tier. Measured content wins over the container's header."""
    if native_sr < MIN_NATIVE_SR:
        return "D"
    if e8k >= TIER_A_MIN_E8K:
        return "A"
    if e8k >= TIER_B_MIN_E8K:
        return "B"
    return "C"


def exact_row_count(repo_id: str, token: str | None) -> int | None:
    """Sum num_rows across parquet footers — footer bytes only, no row groups."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem

        files = [f for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
                 if f.endswith(".parquet")]
        if not files:
            return None
        fs = HfFileSystem(token=token)
        total = 0
        for f in files:
            with fs.open(f"datasets/{repo_id}/{f}") as fh:
                total += pq.ParquetFile(fh).metadata.num_rows
        return total
    except Exception:  # noqa: BLE001 - an unavailable count must not abort the audit
        return None


_MAX_ROW_GROUPS = 5  # bounds the download; each read costs that group's bytes


def _iter_stratified(src, token: str | None, n: int):
    """Sample `n` rows spread across the repo, reading only a few parquet row groups.

    Two rejected alternatives, for the record:

    * `_iter_source` reads in repo order. Fine for the ASR prepare pass, wrong here —
      these repos are grouped by speaker, so the first N rows are one speaker in one
      session, which is precisely what a per-speaker bandwidth audit must not assume.
    * `IterableDataset.shuffle` fixes the bias but fills a reservoir of `buffer_size`
      real rows first. At ~0.5 MB per clip that is hundreds of MB per repo, for a
      measurement that only needs a few dozen clips.

    Reading a handful of row groups spread evenly over (shard, group) pairs gives the
    speaker diversity without the download. Row-group *selection* costs footer bytes
    only; each group we actually read costs its own bytes, hence the cap.

    Audiofolder repos have no row groups, so they fall back to `_iter_source`.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    from kreyol_asr.datasets import (AUDIO_CANDIDATES, SPEAKER_CANDIDATES,
                                     _is_audiofolder, _pick)

    if _is_audiofolder(src.repo_id, token):
        yield from _iter_source(src, token, n)
        return

    files = sorted(f for f in HfApi(token=token).list_repo_files(src.repo_id, repo_type="dataset")
                   if f.endswith(".parquet"))
    if not files:
        yield from _iter_source(src, token, n)
        return

    fs = HfFileSystem(token=token)
    # (shard, row_group) index built from footers alone.
    pairs: list[tuple[str, int]] = []
    for f in files:
        with fs.open(f"datasets/{src.repo_id}/{f}") as fh:
            pairs += [(f, i) for i in range(pq.ParquetFile(fh).metadata.num_row_groups)]
    if not pairs:
        yield from _iter_source(src, token, n)
        return

    k = min(_MAX_ROW_GROUPS, len(pairs))
    picks = [pairs[round(i * (len(pairs) - 1) / max(k - 1, 1))] for i in range(k)]
    per_group = max(1, n // k)
    console.print(f"  {len(files)} shard(s), {len(pairs)} row group(s) -> "
                  f"reading {k} spread evenly, {per_group} clips each")

    emitted = 0
    for path, rg in picks:
        if emitted >= n:
            break
        with fs.open(f"datasets/{src.repo_id}/{path}") as fh:
            pf = pq.ParquetFile(fh)
            cols = pf.schema_arrow.names
            audio_col = _pick(src.audio_column, cols, AUDIO_CANDIDATES, "audio")
            spk_col = _pick(src.speaker_column, cols, SPEAKER_CANDIDATES, "speaker",
                            required=False)
            want = [audio_col] + ([spk_col] if spk_col else [])
            tbl = pf.read_row_group(rg, columns=want)

        rows = tbl.to_pylist()
        # Spread within the group too: adjacent rows are usually the same utterance
        # session, so a contiguous head would re-introduce the bias we just removed.
        step = max(1, len(rows) // per_group)
        for row in rows[::step][:per_group]:
            if emitted >= n:
                break
            spk = row.get(spk_col) if spk_col else None
            yield {"index": emitted, "audio": row[audio_col],
                   "speaker": str(spk) if spk is not None else None}
            emitted += 1


def probe(cfg, token: str | None, n_samples: int = 30) -> list[SourceReport]:
    reports: list[SourceReport] = []
    for src in cfg.sources:
        rep = SourceReport(repo_id=src.repo_id, slug=src.slug, synthetic_flag=src.synthetic)
        console.print(f"[bold]probing[/bold] {src.repo_id} (n={n_samples})")
        rep.rows_total = exact_row_count(src.repo_id, token)
        try:
            for item in _iter_stratified(src, token, n_samples):
                stats = _measure(item["audio"], item["speaker"])
                if stats:
                    rep.clips.append(stats)
        except Exception as e:  # noqa: BLE001 - one dead repo must not kill the run
            rep.error = f"{type(e).__name__}: {e}"
            console.print(f"  [red]failed[/red] {rep.error[:160]}")
        if rep.clips:
            srs = Counter(c.native_sr for c in rep.clips)
            med_e8k = statistics.median(c.e8k for c in rep.clips)
            console.print(f"  {len(rep.clips)} clips | sr={dict(srs)} | "
                          f"median E>8k={med_e8k:.2e} | rows={rep.rows_total}")
        reports.append(rep)
    return reports


def _agg(clips: list[ClipStats]) -> dict[str, Any]:
    med = statistics.median
    srs = Counter(c.native_sr for c in clips)
    dominant_sr = srs.most_common(1)[0][0]
    e8k = med(c.e8k for c in clips)
    return {
        "clips": len(clips),
        "sr_hist": dict(sorted(srs.items())),
        "dominant_sr": dominant_sr,
        "fmt": Counter(c.fmt for c in clips).most_common(1)[0][0],
        "median_e8k": e8k,
        "median_f999": med(c.f999 for c in clips),
        "median_duration": med(c.duration for c in clips),
        "total_duration": sum(c.duration for c in clips),
        "clipped_frac": sum(1 for c in clips if c.clip_ratio > 0.005) / len(clips),
        "max_dc": max(c.dc_offset for c in clips),
        "tier": tier_of(dominant_sr, e8k),
    }


def analyse(reports: list[SourceReport]) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    per_speaker: dict[str, Any] = defaultdict(dict)
    md5_by_repo: dict[str, set[str]] = {}

    for rep in reports:
        if not rep.clips:
            per_source[rep.slug] = {"error": rep.error, "rows": rep.rows_total}
            continue
        per_source[rep.slug] = {
            "repo_id": rep.repo_id, "rows": rep.rows_total,
            "synthetic_flag": rep.synthetic_flag, **_agg(rep.clips),
        }
        md5_by_repo[rep.slug] = {c.audio_md5 for c in rep.clips if c.audio_md5}

        by_spk: dict[str, list[ClipStats]] = defaultdict(list)
        for c in rep.clips:
            by_spk[c.speaker or "<none>"].append(c)
        for spk, cs in by_spk.items():
            per_speaker[spk][rep.slug] = _agg(cs)

    # Repos that are the same corpus re-exported. Compared on audio md5 of the
    # sampled rows, so a hit is strong evidence and a miss is only suggestive.
    duplicates = []
    slugs = sorted(md5_by_repo)
    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            sa, sb = md5_by_repo[a], md5_by_repo[b]
            if not sa or not sb:
                continue
            overlap = len(sa & sb)
            if overlap:
                duplicates.append({
                    "a": a, "b": b, "shared": overlap,
                    "frac_a": overlap / len(sa), "frac_b": overlap / len(sb),
                })

    voices: dict[str, Any] = {}
    for spk, sources in per_speaker.items():
        name = VOICE_BY_SPEAKER_ID.get(spk) or VOICE_BY_SPEAKER_ID.get(spk.lower())
        best = min((s["tier"] for s in sources.values()), default="D")
        voices[spk] = {
            "voice": name,
            "sources": {k: {"clips": v["clips"], "tier": v["tier"],
                            "median_e8k": v["median_e8k"]} for k, v in sources.items()},
            "best_tier": best,
            "stage2_eligible": best in ("A", "B"),
        }

    return {"per_source": per_source, "per_speaker": dict(per_speaker),
            "duplicates": duplicates, "voices": voices}


def render(result: dict[str, Any], n_samples: int) -> str:
    L = [
        "# TTS data audit",
        "",
        f"Sampled up to **{n_samples} clips per source**, drawn from row groups spread evenly across",
        "all parquet shards (and spread again within each group), so the sample crosses speaker and",
        "session boundaries instead of reading the head of shard 0. Row counts are exact, read from",
        "parquet footers; every other number is a sample statistic.",
        "",
        "`E>8k` is the fraction of spectral energy above 8 kHz — the Nyquist of 16 kHz audio.",
        "A 24 kHz file with `E>8k` near zero was upsampled from something narrower, and no",
        "amount of correct resampling puts the missing band back.",
        "",
        f"Tiers: **A** `E>8k >= {TIER_A_MIN_E8K:.0e}` (Stage 1 + 2) · "
        f"**B** `>= {TIER_B_MIN_E8K:.0e}` (Stage 1; Stage 2 only if no Tier A) · "
        f"**C** below that (Stage 1 only) · **D** native SR < {MIN_NATIVE_SR} Hz.",
        "",
        f"Reference: **stock Kokoro's own output measures E>8k = {KOKORO_REF_E8K:.2e}** "
        "(f99.9 ~ 11.4 kHz).",
        "That is the distribution the fine-tune must learn to emit, so the `vs Kokoro` column —",
        "how many times below it a source sits — predicts how much duller the result will sound",
        "far better than the raw number does.",
        "",
        "## Sources",
        "",
        "| source | rows | sampled | fmt | native SR | median E>8k | vs Kokoro | f99.9 | clipped | tier | flagged synthetic |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|:--:|:--:|",
    ]
    for slug, s in sorted(result["per_source"].items()):
        if "error" in s:
            L.append(f"| `{slug}` | {s.get('rows') or '?'} | — | — | — | — | — | — | — | ERR | — |")
            continue
        srs = "/".join(str(k) for k in s["sr_hist"])
        ratio = KOKORO_REF_E8K / max(s["median_e8k"], 1e-12)
        L.append(
            f"| `{slug}` | {s['rows'] or '?'} | {s['clips']} | {s['fmt']} | {srs} | "
            f"{s['median_e8k']:.1e} | {ratio:,.0f}x low | {s['median_f999']/1000:.1f}k | "
            f"{s['clipped_frac']:.0%} | **{s['tier']}** | {'yes' if s['synthetic_flag'] else 'no'} |"
        )

    errs = {k: v["error"] for k, v in result["per_source"].items() if "error" in v}
    if errs:
        L += ["", "### Sources that could not be read", ""]
        L += [f"- `{k}`: {v}" for k, v in errs.items()]

    L += ["", "## Voices", "",
          "`speaker_id` values matched against `serverless-tts/handler.py:VOICES`.", "",
          "| speaker_id | voice | sampled clips | sources (tier) | best tier | Stage 2 |",
          "|---|---|---:|---|:--:|:--:|"]
    for spk, v in sorted(result["voices"].items(),
                         key=lambda kv: (kv[1]["voice"] is None, kv[0])):
        srcs = ", ".join(f"{k}({d['tier']})" for k, d in sorted(v["sources"].items()))
        total = sum(d["clips"] for d in v["sources"].values())
        name = v["voice"] or "—"
        L.append(f"| `{spk}` | {name} | {total} | {srcs} | **{v['best_tier']}** | "
                 f"{'yes' if v['stage2_eligible'] else 'NO'} |")

    L += ["", "## Duplicate corpora", ""]
    if result["duplicates"]:
        L.append("| a | b | shared md5s | % of a | % of b |")
        L.append("|---|---|---:|---:|---:|")
        for d in result["duplicates"]:
            L.append(f"| `{d['a']}` | `{d['b']}` | {d['shared']} | "
                     f"{d['frac_a']:.0%} | {d['frac_b']:.0%} |")
        L.append("")
        L.append("Audio md5s match, so these are the same recordings re-exported. Prep dedupes on "
                 "`md5(pcm) + phonemes`, so the collapse is automatic — but the *effective* corpus "
                 "size is smaller than the row counts suggest.")
    else:
        L.append("None detected among the sampled rows.")

    L += ["", "## Row counts vs the dataset config", "",
          "The config's inline counts are comments, not assertions. Exact footer counts:", "",
          "| source | exact rows |", "|---|---:|"]
    for slug, s in sorted(result["per_source"].items()):
        L.append(f"| `{slug}` | {s.get('rows') if s.get('rows') is not None else 'unknown'} |")

    return "\n".join(L) + "\n"


def run(cfg, token: str | None, out_dir: Path, n_samples: int = 30) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = probe(cfg, token, n_samples)
    result = analyse(reports)
    result["n_samples"] = n_samples
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "audit.md").write_text(render(result, n_samples))
    console.print(f"\n[green]Wrote[/green] {out_dir/'audit.md'}")
    return result
