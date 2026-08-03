"""Radio Haiti-Inter (Zenodo 17818122) -> an audiofolder `prepare` already understands.

The corpus is ~60 h of archival Haitian radio (1957-2003) with **machine-generated**
transcripts: Havard et al.'s Kreyòl wav2vec2/data2vec ASR, which scores ~21% CER on its
own test set. The EAF headers say so outright (`AUTHOR="ASR transcription - model-unknown"`).
That model is plausibly *weaker* than the fine-tune in this repo, so nothing here is
trusted by default — `radio_gate` decides what survives, and this module only produces
candidates.

Layout inside the three zips (measured, not assumed):

    recordings/{train,val,test}/<uuid>.wav       16 kHz mono PCM16, 219 files
    eaf/{train,val,test}/<uuid>.eaf              ELAN, word- and character-aligned
    transcriptions/{train,val,test}/<uuid>.csv   hypothesis,ctc_score,confidence,
                                                 recording_id,segment_start,segment_end,speaker
    transcriptions/{train,val,test}/<uuid>.json  "<start>-><end>" -> {timesteps,
                                                 final_non_blank, unit_confidence, word_confidence}

The CSV is the primary source: it alone carries text, times, confidence, speaker and
recording. The EAF is opened only for `words@<speaker>` boundaries, and only for the
recordings that actually hold an over-long segment — otherwise this would parse 1.5 GB
of XML to serve the ~96% of segments that are already short enough.
"""

from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import soundfile as sf

from . import SAMPLE_RATE
from .text import TokenizerCoverage, normalize, style_stats

from kreyol_common import console
from kreyol_common.audio import to_mono

ZENODO_RECORD = "17818122"
ZENODO_DOI = "10.5281/zenodo.17818122"
# Verified against the Zenodo API. A truncated 6 GB download that silently unzips
# partially is the failure mode worth pinning these for.
ZENODO_FILES = {
    "recordings.zip": ("0d5e520e06574651d75261210f24cc22", 6_022_232_541),
    "eaf.zip": ("38a5de184ab358dff2140ddbb5ef71ea", 108_821_504),
    "transcriptions.zip": ("d8083623b47286486dca204a4266f89e", 34_298_191),
}
CORPUS_SPLITS = ("train", "val", "test")

# Defaults, each tied to something measured rather than picked:
#   MAX_S  15, not 18 — `prepare` hard-drops above max_duration: 18.0 measured from
#          decoded frames, so 3 s of margin means padding and rounding never silently
#          discard work already done.
#   MIN_S  1.0, not 0.5 — 3.9% of source segments fall under 0.5 s and `prepare` would
#          drop them anyway; merging beats losing them, and sub-second RNN-T targets
#          are noisy.
#   PAD_MS diarization boundaries clip onsets; clamped to neighbours so padding never
#          eats into the next speaker.
MAX_S = 15.0
MIN_S = 1.0
TARGET_S = 12.0
MAX_GAP_MS = 400
PAD_MS = 100

# The corpus writes Creole clitics space-separated ("n ap", "k ap"); this repo's other
# 18 sources write them with an apostrophe, and text.py documents why that matters —
# a mismatched apostrophe fragments the BPE on the highest-frequency tokens in the
# language. Deliberately narrow: only the `ap` progressive, which is the case text.py
# calls out. Everything else ("m te", "pou l tande") is space-separated in IPN too, so
# rewriting it would invent an orthography rather than match one.
_CLITIC_AP = re.compile(r"\b([mnlkty]) ap\b")
# The inverse, for comparing our model's output against the corpus's spelling.
_CLITIC_APOSTROPHE = re.compile(r"\b([mnlkty])'ap\b")


def restyle_clitics(text: str) -> str:
    """`n ap` -> `n'ap`. Rule-based over a closed set; no hallucination surface."""
    return _CLITIC_AP.sub(r"\1'ap", text)


def unstyle_clitics(text: str) -> str:
    """`n'ap` -> `n ap`. Used to compare across the two spellings, never to write data."""
    return _CLITIC_APOSTROPHE.sub(r"\1 ap", text)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Segment:
    recording_id: str
    corpus_split: str
    speaker: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    ctc_score: float = 0.0
    words: tuple[Word, ...] = ()
    origin: str = "csv"  # csv | split | merged | split+merged

    @property
    def duration(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def speaker_key(self) -> str:
        """Namespaced, because `SPEAKER_00` appears in all 219 recordings.

        These are pyannote diarization labels scoped per recording, not identities.
        Passed to `_split_real` unnamespaced they would make 219 different people
        look like one speaker and put that "speaker" in both train and test —
        silently destroying the speaker-disjoint guarantee.
        """
        return f"{self.recording_id}:{self.speaker}"

    @property
    def chars_per_second(self) -> float:
        return len(self.text) / self.duration if self.duration else float("inf")


# --------------------------------------------------------------------------- parsing


def read_segments_csv(path: Path, corpus_split: str = "") -> list[Segment]:
    """One `<uuid>.csv` -> segments. Times are milliseconds in the source."""
    out: list[Segment] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                start, end = int(row["segment_start"]), int(row["segment_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            out.append(Segment(
                recording_id=row.get("recording_id") or path.stem,
                corpus_split=corpus_split or path.parent.name,
                speaker=row.get("speaker") or "SPEAKER_00",
                start_ms=start,
                end_ms=end,
                # The hypothesis field carries a leading space in every row observed.
                text=(row.get("hypothesis") or "").strip(),
                confidence=float(row.get("confidence") or 0.0),
                ctc_score=float(row.get("ctc_score") or 0.0),
            ))
    out.sort(key=lambda s: (s.speaker, s.start_ms))
    return out


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_eaf_words(path: Path) -> dict[str, list[Word]]:
    """`{speaker: [Word, ...]}` from the `words@<speaker>` tiers.

    Three traps this handles, all present in the real files:
      - `words@SPEAKER_00` and `characters@SPEAKER_00` are child tiers of the
        transcription tier. Reading them as transcriptions would duplicate every
        utterance at word and character granularity.
      - TIME_SLOTs are referenced by id and are not declared in time order.
      - Annotations may be REF_ANNOTATION (no time slots of their own) rather than
        ALIGNABLE_ANNOTATION; those carry no timing and are skipped.
    """
    slots: dict[str, int] = {}
    words: dict[str, list[Word]] = defaultdict(list)

    for _, elem in ET.iterparse(str(path), events=("end",)):
        tag = _strip_ns(elem.tag)
        if tag == "TIME_SLOT":
            sid, val = elem.get("TIME_SLOT_ID"), elem.get("TIME_VALUE")
            if sid is not None and val is not None:
                slots[sid] = int(val)
        elif tag == "TIER":
            tier_id = elem.get("TIER_ID") or ""
            if not tier_id.startswith("words@"):
                elem.clear()
                continue
            speaker = tier_id.split("@", 1)[1]
            for ann in elem.iter():
                if _strip_ns(ann.tag) != "ALIGNABLE_ANNOTATION":
                    continue
                r1, r2 = ann.get("TIME_SLOT_REF1"), ann.get("TIME_SLOT_REF2")
                if r1 not in slots or r2 not in slots:
                    continue
                value = ""
                for child in ann:
                    if _strip_ns(child.tag) == "ANNOTATION_VALUE":
                        value = (child.text or "").strip()
                if not value:
                    continue
                words[speaker].append(Word(value, slots[r1], slots[r2]))
            elem.clear()

    for spk in words:
        words[spk].sort(key=lambda w: w.start_ms)
    return dict(words)


def attach_words(segs: Iterable[Segment], words: dict[str, list[Word]]) -> list[Segment]:
    """Hang each segment's word alignment off it, for segments that may need splitting."""
    out = []
    for seg in segs:
        pool = words.get(seg.speaker) or []
        inside = tuple(w for w in pool
                       if w.start_ms >= seg.start_ms - 1 and w.end_ms <= seg.end_ms + 1)
        out.append(replace(seg, words=inside) if inside else seg)
    return out


def words_match_text(seg: Segment) -> bool:
    """Do the EAF words reconstruct the CSV hypothesis?

    If they do not, the alignment belongs to different text and splitting on it would
    pair audio with a transcript that does not describe it. Compared loosely — the CSV
    text has already been normalized by the time this runs.
    """
    if not seg.words:
        return False
    joined = " ".join(w.text for w in seg.words)
    squash = lambda t: re.sub(r"[^a-zàèò]", "", t.lower())  # noqa: E731
    return squash(joined) == squash(seg.text)


# ------------------------------------------------------------------- re-segmentation


def split_long(seg: Segment, *, max_s: float = MAX_S, min_s: float = MIN_S) -> list[Segment]:
    """Cut an over-long segment at word boundaries. `[]` when it cannot be cut safely.

    Returning `[]` rather than guessing is the point: a mid-word cut pairs half a word
    of audio with a whole word of text on both sides of the seam, which is worse than
    losing the segment. The caller counts the drop.
    """
    if seg.duration <= max_s:
        return [seg]
    if not seg.words or not words_match_text(seg):
        return []

    cut_i = _best_cut(seg, min_s)
    if cut_i is None:
        return []

    left_words, right_words = seg.words[:cut_i], seg.words[cut_i:]
    boundary = (left_words[-1].end_ms + right_words[0].start_ms) // 2
    left = replace(seg, end_ms=boundary, words=left_words,
                   text=" ".join(w.text for w in left_words), origin="split")
    right = replace(seg, start_ms=boundary, words=right_words,
                    text=" ".join(w.text for w in right_words), origin="split")
    return split_long(left, max_s=max_s, min_s=min_s) + \
        split_long(right, max_s=max_s, min_s=min_s)


def _best_cut(seg: Segment, min_s: float) -> int | None:
    """Index into `seg.words` to cut before: widest inter-word gap, then most central.

    Gap dominates so real pauses win. Where words are contiguous (every gap 0 ms, which
    happens across a whole segment) the centrality tie-break takes over and produces a
    balanced split instead of shaving off one word at a time.
    """
    mid = (seg.start_ms + seg.end_ms) // 2
    best, best_key = None, None
    for i in range(1, len(seg.words)):
        prev, nxt = seg.words[i - 1], seg.words[i]
        cut = (prev.end_ms + nxt.start_ms) // 2
        if (cut - seg.start_ms) / 1000 < min_s or (seg.end_ms - cut) / 1000 < min_s:
            continue
        key = (nxt.start_ms - prev.end_ms, -abs(cut - mid))
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def merge_short(segs: list[Segment], *, target_s: float = TARGET_S,
                max_gap_ms: int = MAX_GAP_MS) -> list[Segment]:
    """Join consecutive same-speaker segments separated by a short gap.

    The median source segment is a couple of seconds long, which is inefficient for
    RNN-T and gives the model very little context. Merging is bounded by `max_gap_ms`
    so only true continuations join — a long pause is a topic boundary, and gluing
    across it would fabricate an utterance that was never spoken as one.

    Confidence of the merged span is duration-weighted, so a long confident segment is
    not dragged down by a short doubtful one it absorbed.
    """
    if not segs:
        return []
    out: list[Segment] = []
    run: list[Segment] = [segs[0]]

    def flush() -> None:
        if len(run) == 1:
            out.append(run[0])
            return
        total = sum(s.duration for s in run) or 1.0
        origin = "split+merged" if any("split" in s.origin for s in run) else "merged"
        out.append(replace(
            run[0],
            end_ms=run[-1].end_ms,
            text=" ".join(s.text for s in run if s.text).strip(),
            confidence=sum(s.confidence * s.duration for s in run) / total,
            ctc_score=sum(s.ctc_score for s in run),
            words=tuple(w for s in run for w in s.words),
            origin=origin,
        ))

    for seg in segs[1:]:
        prev = run[-1]
        joined_s = (seg.end_ms - run[0].start_ms) / 1000.0
        if (seg.speaker == prev.speaker
                and seg.recording_id == prev.recording_id
                and 0 <= seg.start_ms - prev.end_ms <= max_gap_ms
                and joined_s <= target_s):
            run.append(seg)
            continue
        flush()
        run = [seg]
    flush()
    return out


def resegment(segs: list[Segment], *, max_s: float = MAX_S, min_s: float = MIN_S,
              target_s: float = TARGET_S, max_gap_ms: int = MAX_GAP_MS,
              merge: bool = True) -> tuple[list[Segment], Counter]:
    """Split first, then merge — so split fragments can also absorb their neighbours."""
    drops: Counter = Counter()
    split: list[Segment] = []
    for seg in segs:
        parts = split_long(seg, max_s=max_s, min_s=min_s)
        if not parts:
            drops["too_long_no_word_alignment"] += 1
            drops["_hours_lost"] += seg.duration / 3600
            continue
        split.extend(parts)

    merged = merge_short(split, target_s=target_s, max_gap_ms=max_gap_ms) if merge else split

    out = []
    for seg in merged:
        if seg.duration < min_s:
            drops["too_short_unmergeable"] += 1
            continue
        if not seg.text:
            drops["empty_text"] += 1
            continue
        out.append(seg)
    return out, drops


# ----------------------------------------------------------------------- audio


def slice_recording(wav_path: Path, segs: list[Segment], out_dir: Path, *,
                    pad_ms: int = PAD_MS) -> Iterator[tuple[Path, Segment, float]]:
    """Cut a recording into per-segment 16 kHz mono wavs.

    Padding is clamped to the recording and to the neighbouring segment, so extra
    context never eats into the next speaker's audio.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE or audio.ndim > 1:
        audio = to_mono(audio, sr, SAMPLE_RATE)
    total_ms = int(len(audio) / SAMPLE_RATE * 1000)

    ordered = sorted(segs, key=lambda s: s.start_ms)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(ordered):
        prev_end = ordered[i - 1].end_ms if i else 0
        next_start = ordered[i + 1].start_ms if i + 1 < len(ordered) else total_ms
        lo = max(0, prev_end, seg.start_ms - pad_ms)
        hi = min(total_ms, next_start, seg.end_ms + pad_ms)
        a, b = int(lo * SAMPLE_RATE / 1000), int(hi * SAMPLE_RATE / 1000)
        clip = audio[a:b]
        if len(clip) == 0:
            continue
        path = out_dir / f"{seg.recording_id}__{i:04d}.wav"
        sf.write(path, clip, SAMPLE_RATE, subtype="PCM_16")
        yield path, seg, len(clip) / SAMPLE_RATE


# ----------------------------------------------------------------------- ingest


def _recordings(root: Path, splits: Iterable[str]) -> list[tuple[str, Path, Path, Path]]:
    """(split, wav, csv, eaf) per recording, skipping incomplete triples."""
    out = []
    for split in splits:
        tdir = root / "transcriptions" / split
        if not tdir.is_dir():
            continue
        for csv_path in sorted(tdir.glob("*.csv")):
            uid = csv_path.stem
            wav = root / "recordings" / split / f"{uid}.wav"
            eaf = root / "eaf" / split / f"{uid}.eaf"
            if wav.exists():
                out.append((split, wav, csv_path, eaf))
    return out


def ingest(root: Path, out: Path, *, base_model: str, splits: Iterable[str] = ("train",),
           limit: int | None = None, restyle: str = "clitics", merge: bool = True,
           max_s: float = MAX_S, min_s: float = MIN_S, target_s: float = TARGET_S,
           pad_ms: int = PAD_MS, token: str | None = None) -> dict[str, Any]:
    """Zenodo tree -> `metadata.all.jsonl` + sliced wavs.

    Writes `metadata.all.jsonl`, never `metadata.jsonl`: `_iter_localdir` only accepts
    the latter, so an ungated ingest cannot reach training by accident.
    """
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "wav"
    coverage = TokenizerCoverage(base_model, token=token)

    recordings = _recordings(root, splits)
    if not recordings:
        raise RuntimeError(
            f"No recordings under {root} for splits={list(splits)}. Expected "
            f"{root}/transcriptions/<split>/*.csv alongside {root}/recordings/<split>/*.wav — "
            f"run scripts/fetch_radio_haiti.sh first."
        )
    if limit:
        recordings = recordings[:limit]

    rows: list[dict[str, Any]] = []
    drops: Counter = Counter()
    restyled = 0
    hours_in = 0.0

    for split, wav, csv_path, eaf in recordings:
        segs = read_segments_csv(csv_path, split)
        hours_in += sum(s.duration for s in segs) / 3600

        # Only pay for the XML when this recording actually holds an over-long segment.
        if eaf.exists() and any(s.duration > max_s for s in segs):
            try:
                segs = attach_words(segs, read_eaf_words(eaf))
            except ET.ParseError:
                drops["unparsable_eaf"] += 1

        kept, d = resegment(segs, max_s=max_s, min_s=min_s, target_s=target_s, merge=merge)
        drops.update(d)

        cleaned: list[Segment] = []
        for seg in kept:
            text = normalize(seg.text, lowercase=False, strip_bracketed=True,
                             normalize_apostrophes=True)
            if restyle.startswith("clitics"):
                styled = restyle_clitics(text)
                restyled += styled != text
                text = styled
            if "period" in restyle and text and text[-1] not in ".!?":
                text += "."
            text = coverage.sanitize(text)
            if not text:
                drops["empty_after_normalize"] += 1
                continue
            cleaned.append(replace(seg, text=text))

        if not cleaned:
            continue  # nothing survived — do not read a 30-minute wav to slice zero clips
        for path, seg, duration in slice_recording(wav, cleaned, wav_dir, pad_ms=pad_ms):
            if duration > 18.0:  # `prepare` would drop it; do not ship audio that dies later
                drops["over_prepare_max_duration"] += 1
                path.unlink(missing_ok=True)
                continue
            coverage.add(seg.text)
            rows.append({
                "file_name": str(path.relative_to(out)),
                "text": seg.text,
                "speaker": seg.speaker_key,
                "recording_id": seg.recording_id,
                "corpus_split": seg.corpus_split,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "duration": round(duration, 3),
                "confidence": round(seg.confidence, 6),
                "ctc_score": round(seg.ctc_score, 3),
                "n_words": len(seg.text.split()),
                "origin": seg.origin,
            })

    if not rows:
        raise RuntimeError("No usable segments survived ingest — check the drop counts above.")

    meta = out / "metadata.all.jsonl"
    with open(meta, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = summarize(rows, drops, hours_in=hours_in, restyled=restyled,
                      recordings=len(recordings), coverage=coverage.report())
    (out / "ingest_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "ingest_report.md").write_text(render_ingest_report(stats))
    console.print(f"[green]Wrote[/green] {meta} ({stats['hours']} h over {len(rows)} clips)")
    console.print("[yellow]Not trainable yet[/yellow] — run `kreyol-asr radio agree` "
                  "then `kreyol-asr radio gate` to produce metadata.jsonl.")
    return stats


# ---------------------------------------------------------------------- reporting


def percentiles(values: list[float], ps: Iterable[int] = (5, 10, 25, 50, 75, 90, 95)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{p}": round(ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))], 6)
            for p in ps}


def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[tuple[str, int]]:
    if not values:
        return []
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        i = min(bins - 1, max(0, int((v - lo) / width)))
        counts[i] += 1
    return [(f"{lo + i * width:.2f}-{lo + (i + 1) * width:.2f}", c) for i, c in enumerate(counts)]


def _sum_by(rows: list[dict], key: str) -> dict[str, float]:
    acc: defaultdict[str, float] = defaultdict(float)
    for r in rows:
        acc[r[key]] += r["duration"]
    return dict(acc)


def summarize(rows: list[dict], drops: Counter, **extra: Any) -> dict[str, Any]:
    durs = [r["duration"] for r in rows]
    texts = [r["text"] for r in rows]
    words = sum(r["n_words"] for r in rows)
    return {
        "clips": len(rows),
        "hours": round(sum(durs) / 3600, 3),
        "recordings_seen": extra.get("recordings"),
        "hours_before_resegment": round(extra.get("hours_in", 0.0), 3),
        "duration": percentiles(durs) | {"mean": round(sum(durs) / len(durs), 3)},
        "duration_histogram": histogram(durs, 16, 0, 16),
        "confidence": percentiles([r["confidence"] for r in rows]),
        "chars_per_second": percentiles([len(r["text"]) / r["duration"] for r in rows if r["duration"]]),
        "origin": dict(Counter(r["origin"] for r in rows)),
        # A single dominant programme is a diversity risk, not a win — surface the
        # top recordings so it is visible rather than buried in an hours total.
        "top_recording_hours": dict(Counter(
            {k: round(v / 3600, 3) for k, v in
             _sum_by(rows, "recording_id").items()}).most_common(10)),
        "speakers": len({r["speaker"] for r in rows}),
        "words": words,
        "style": style_stats(texts),
        "apostrophes_per_1k_words": round(sum(t.count("'") for t in texts) / max(words, 1) * 1000, 2),
        "digits_per_1k_words": round(
            sum(c.isdigit() for t in texts for c in t) / max(words, 1) * 1000, 2),
        "charset": "".join(sorted({c for t in texts for c in t})),
        "restyled_clips": extra.get("restyled"),
        "dropped": {k: (round(v, 3) if k.startswith("_") else v) for k, v in drops.items()},
        "tokenizer_coverage": extra.get("coverage"),
        "tokenizer_verdict": TokenizerCoverage.verdict(extra["coverage"]) if extra.get("coverage") else None,
    }


def band_energy(wav_path: Path, *, frame_ms: int = 25, speech_percentile: float = 60.0,
                max_seconds: float = 120.0) -> dict[str, float]:
    """Where the energy actually sits, measured on speech frames only.

    A whole-file band ratio is uninterpretable on this material: 85-99% of the energy
    is below 1 kHz (rumble and mains hum on 40-year-old tape), which swamps whatever is
    happening at 4-8 kHz. Framing and keeping only the loudest frames restricts the
    measurement to speech, which is the thing we actually want to know the bandwidth of.

    The point is to find out whether these recordings are genuinely 8 kHz material
    upsampled to 16 kHz — which is fine, even useful, as long as we know it rather than
    discover it after training.
    """
    import numpy as np

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False,
                        frames=int(max_seconds * 48000))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = int(sr * frame_ms / 1000)
    if len(audio) < n * 4:
        return {}
    frames = audio[:len(audio) // n * n].reshape(-1, n)
    energy = (frames ** 2).sum(axis=1)
    speech = frames[energy >= np.percentile(energy, speech_percentile)]
    if not len(speech):
        return {}

    spec = np.abs(np.fft.rfft(speech * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    power = spec.mean(axis=0)
    total = power.sum() or 1.0

    edges = [(0, 1000), (1000, 3400), (3400, 4000), (4000, 6000), (6000, 8000)]
    out = {f"e_{lo}_{hi}": round(float(power[(freqs >= lo) & (freqs < hi)].sum() / total), 6)
           for lo, hi in edges if hi <= sr / 2}
    cumulative = np.cumsum(power) / total
    out["rolloff95_hz"] = round(float(freqs[int(np.searchsorted(cumulative, 0.95))]), 1)
    out["sample_rate"] = sr
    return out


def inspect_corpus(root: Path, *, base_model: str, audio_sample: int = 40,
                   skip_audio: bool = False, compare_manifest: Path | None = None,
                   token: str | None = None, seed: int = 1804) -> dict[str, Any]:
    """Measure the archive before converting any of it. Read-only."""
    recordings = _recordings(root, CORPUS_SPLITS)
    if not recordings:
        raise RuntimeError(f"No recordings under {root} — run scripts/fetch_radio_haiti.sh.")

    per_split: Counter = Counter()
    segs_all: list[Segment] = []
    speaker_counts: Counter = Counter()
    for split, _wav, csv_path, _eaf in recordings:
        per_split[split] += 1
        segs = read_segments_csv(csv_path, split)
        segs_all.extend(segs)
        speaker_counts[len({s.speaker for s in segs})] += 1

    # Orphans in both directions — a wav with no transcript is invisible to
    # `_recordings`, so count it explicitly rather than letting it disappear.
    have = {(s, p.stem) for s, _w, p, _e in recordings}
    wavs = {(sp, p.stem) for sp in CORPUS_SPLITS
            for p in (root / "recordings" / sp).glob("*.wav")} if (root / "recordings").is_dir() else set()
    eafs = {(sp, p.stem) for sp in CORPUS_SPLITS
            for p in (root / "eaf" / sp).glob("*.eaf")} if (root / "eaf").is_dir() else set()

    durs = [s.duration for s in segs_all]
    texts = [s.text for s in segs_all]
    coverage = TokenizerCoverage(base_model, token=token)
    for t in texts:
        if t:
            coverage.add(normalize(t))

    long_segs = [s for s in segs_all if s.duration > MAX_S]
    stats: dict[str, Any] = {
        "root": str(root),
        "recordings": len(recordings),
        "recordings_per_split": dict(per_split),
        "wav_without_transcript": sorted(f"{s}/{u}" for s, u in wavs - have)[:20],
        "transcript_without_eaf": sorted(f"{s}/{u}" for s, u in have - eafs)[:20],
        "segments": len(segs_all),
        "segment_hours": round(sum(durs) / 3600, 3),
        "duration": percentiles(durs) | {"mean": round(sum(durs) / max(len(durs), 1), 3)},
        "duration_histogram": histogram(durs, 20, 0, 40),
        "over_max_s": {
            "segments": len(long_segs),
            "share_of_segments": round(len(long_segs) / max(len(segs_all), 1), 4),
            "share_of_hours": round(sum(s.duration for s in long_segs) / max(sum(durs), 1e-9), 4),
        },
        "confidence": percentiles([s.confidence for s in segs_all]),
        # ctc_score grows with length, so it is not a quality signal on its own —
        # reported per second so that is visible rather than assumed either way.
        "ctc_score_per_second": percentiles(
            [s.ctc_score / s.duration for s in segs_all if s.duration]),
        "chars_per_second": percentiles([s.chars_per_second for s in segs_all if s.duration]),
        "empty_hypothesis_rate": round(
            sum(1 for s in segs_all if not s.text) / max(len(segs_all), 1), 5),
        "speakers_per_recording": dict(sorted(speaker_counts.items())),
        "distinct_speaker_labels": len({s.speaker for s in segs_all}),
        "distinct_speaker_keys": len({s.speaker_key for s in segs_all}),
        "style": style_stats(texts),
        "charset": "".join(sorted({c for t in texts for c in t})),
        "apostrophes": sum(t.count("'") for t in texts),
        "digits": sum(c.isdigit() for t in texts for c in t),
        "tokenizer_coverage": coverage.report(),
    }
    stats["tokenizer_verdict"] = TokenizerCoverage.verdict(stats["tokenizer_coverage"])

    # Does word alignment exist everywhere it is needed? This decides whether the 25%
    # of hours living in over-long segments is recoverable or lost.
    rng = random.Random(seed)
    with_long = [r for r in recordings if r[3].exists()]
    rng.shuffle(with_long)
    checked, aligned = 0, 0
    for _split, _wav, csv_path, eaf in with_long[:min(20, len(with_long))]:
        segs = read_segments_csv(csv_path)
        longs = [s for s in segs if s.duration > MAX_S]
        if not longs:
            continue
        try:
            attached = attach_words(longs, read_eaf_words(eaf))
        except ET.ParseError:
            continue
        checked += len(attached)
        aligned += sum(1 for s in attached if words_match_text(s))
    stats["word_alignment"] = {
        "long_segments_checked": checked,
        "with_usable_alignment": aligned,
        "share": round(aligned / checked, 3) if checked else None,
    }

    if not skip_audio:
        rng2 = random.Random(seed)
        sample = list(recordings)
        rng2.shuffle(sample)
        bands = [band_energy(w) for _s, w, _c, _e in sample[:audio_sample]]
        bands = [b for b in bands if b]
        if bands:
            keys = [k for k in bands[0] if k.startswith("e_")]
            stats["audio"] = {
                "measured": len(bands),
                "sample_rates": dict(Counter(b["sample_rate"] for b in bands)),
                "band_energy_mean": {k: round(sum(b[k] for b in bands) / len(bands), 5)
                                     for k in keys},
                "rolloff95_hz": percentiles([b["rolloff95_hz"] for b in bands]),
                "rolloff95_histogram": histogram(
                    [b["rolloff95_hz"] for b in bands], 16, 0, 8000),
            }

    if compare_manifest and Path(compare_manifest).exists():
        stats["vs_existing_corpus"] = _compare_vocab(texts, Path(compare_manifest))
    return stats


def _compare_vocab(texts: list[str], manifest: Path) -> dict[str, Any]:
    """How much of Radio Haiti's vocabulary the existing corpus has never seen.

    Doubles as the code-switching proxy. A grapheme-based French detector is useless
    here: the charset has no `q`, `x`, `é` or `ç`, so the teacher physically cannot
    spell French verbatim — French words come out respelled phonetically, which only a
    vocabulary comparison catches.

    Also reports how the existing corpus spells the clitics, which is the evidence for
    or against `--restyle clitics`.
    """
    ours: Counter = Counter()
    for line in manifest.read_text().splitlines():
        if line.strip():
            ours.update(json.loads(line).get("text", "").lower().split())
    theirs: Counter = Counter()
    for t in texts:
        theirs.update(t.lower().split())

    unseen = [w for w in theirs if w not in ours]
    joined = sum(v for k, v in ours.items() if _CLITIC_APOSTROPHE.fullmatch(k))
    spaced = sum(1 for k in ours if k in set("mnlkty"))
    return {
        "existing_vocab": len(ours),
        "radio_vocab": len(theirs),
        "types_unseen_in_existing": len(unseen),
        "type_oov_rate": round(len(unseen) / max(len(theirs), 1), 4),
        "token_oov_rate": round(sum(theirs[w] for w in unseen) / max(sum(theirs.values()), 1), 4),
        "top_unseen": [w for w, _ in Counter({w: theirs[w] for w in unseen}).most_common(30)],
        "existing_corpus_clitics": {"joined_x'ap": joined, "bare_pronoun_tokens": spaced},
    }


def render_inspect_report(s: dict[str, Any]) -> str:
    cov, style = s["tokenizer_coverage"], s["style"]
    lines = [
        "# Radio Haiti-Inter — inspection",
        "",
        f"Source: Zenodo [{ZENODO_DOI}](https://doi.org/{ZENODO_DOI}), CC-BY-4.0. "
        "Transcripts are machine-generated.",
        "",
        f"- Recordings: **{s['recordings']}** {s['recordings_per_split']}",
        f"- Segments: **{s['segments']}** / **{s['segment_hours']} h**",
        f"- Distinct speaker labels: **{s['distinct_speaker_labels']}** — but "
        f"**{s['distinct_speaker_keys']}** once namespaced per recording. "
        f"The raw labels are per-recording diarization output, not identities.",
        "",
        "## Durations",
        "",
        f"- mean **{s['duration'].get('mean')} s**, median **{s['duration'].get('p50')} s**, "
        f"p95 **{s['duration'].get('p95')} s**",
        f"- Over {MAX_S} s: **{s['over_max_s']['share_of_segments']:.1%}** of segments but "
        f"**{s['over_max_s']['share_of_hours']:.1%}** of the hours",
        "",
        "| bucket (s) | segments |",
        "|---|---:|",
    ]
    lines += [f"| {b} | {c} |" for b, c in s["duration_histogram"] if c]

    wa = s["word_alignment"]
    lines += [
        "",
        "## Word alignment on over-long segments",
        "",
        f"- Checked **{wa['long_segments_checked']}**, usable alignment on "
        f"**{wa['with_usable_alignment']}** (**{wa['share']}**)",
        "",
        "> This share is how much of the over-long hours are recoverable. Anything without "
        "a usable alignment is dropped rather than cut mid-word.",
        "",
        "## Confidence (the corpus's own, per segment)",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in s["confidence"].items()]
    lines += ["", "> `ctc_score` is an unnormalized log-score that grows with length — per "
                  "second it reads:", ""]
    lines += [f"- {k}: {v}" for k, v in s["ctc_score_per_second"].items()]

    if "audio" in s:
        a = s["audio"]
        lines += [
            "", "## Audio", "",
            f"- Sample rates across {a['measured']} recordings: `{a['sample_rates']}`",
            f"- Mean band energy (speech frames only): `{a['band_energy_mean']}`",
            f"- 95% spectral rolloff: `{a['rolloff95_hz']}`",
            "",
            "> A rolloff clustered near 4 kHz means the material is 8 kHz-sourced and merely "
            "stored at 16 kHz. That is usable — narrowband robustness is part of why this "
            "corpus is worth having — but if the existing 45.9 h is all wideband, the model "
            "can learn 'narrowband implies archival register' instead of learning the speech. "
            "Bimodality in this histogram is the thing to look for.",
            "",
            "| rolloff (Hz) | recordings |", "|---|---:|",
        ]
        lines += [f"| {b} | {c} |" for b, c in a["rolloff95_histogram"] if c]

    lines += [
        "",
        "## Text",
        "",
        f"- cased **{style['cased_ratio']:.1%}** · punctuated **{style['punctuated_ratio']:.1%}**",
        f"- apostrophes: **{s['apostrophes']}** · digits: **{s['digits']}**",
        f"- empty hypotheses: **{s['empty_hypothesis_rate']:.2%}**",
        f"- charset: `{s['charset']}`",
        "",
        "## Tokenizer coverage (pretrained BPE, vocab 13088)",
        "",
        f"- tokens/word **{cov['tokens_per_word']}** · `<unk>` rate **{cov['unk_rate']}**",
        f"- Verdict: **{s['tokenizer_verdict']}**",
        "",
    ]
    if "vs_existing_corpus" in s:
        v = s["vs_existing_corpus"]
        lines += [
            "## Against the existing corpus",
            "",
            f"- Type OOV **{v['type_oov_rate']:.1%}** · token OOV **{v['token_oov_rate']:.1%}** "
            f"({v['types_unseen_in_existing']} of {v['radio_vocab']} types unseen)",
            f"- Existing corpus clitic spelling: `{v['existing_corpus_clitics']}`",
            "",
            "> The clitic counts are the evidence for or against `--restyle clitics`. If the "
            "existing corpus writes `n'ap`, restyling aligns the two; if it writes `n ap`, "
            "restyling would introduce the mismatch it is meant to remove.",
            "",
            f"- Most frequent unseen words: `{', '.join(v['top_unseen'][:20])}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_ingest_report(s: dict[str, Any]) -> str:
    style, cov = s["style"], s.get("tokenizer_coverage") or {}
    lines = [
        "# Radio Haiti-Inter — ingest report",
        "",
        f"Source: Zenodo [{ZENODO_DOI}](https://doi.org/{ZENODO_DOI}), CC-BY-4.0.",
        "Transcripts are **machine-generated** (Havard et al., Interspeech 2025). Not trainable",
        "until `radio gate` writes `metadata.jsonl`.",
        "",
        f"- Recordings read: **{s['recordings_seen']}**",
        f"- Segment hours before re-segmentation: **{s['hours_before_resegment']} h**",
        f"- Clips out: **{s['clips']}** / **{s['hours']} h** across "
        f"**{s['speakers']}** namespaced speaker keys",
        f"- Clitic rewrites (`n ap` -> `n'ap`): **{s['restyled_clips']}**",
        "",
        "## Duration",
        "",
        f"- mean **{s['duration'].get('mean')} s**, median **{s['duration'].get('p50')} s**, "
        f"p95 **{s['duration'].get('p95')} s**",
        "",
        "| bucket (s) | clips |",
        "|---|---:|",
    ]
    lines += [f"| {b} | {c} |" for b, c in s["duration_histogram"] if c]
    lines += ["", "## Confidence (corpus's own, per segment)", ""]
    lines += [f"- {k}: {v}" for k, v in s["confidence"].items()]
    lines += ["", "## Provenance of each clip", ""]
    lines += [f"- {k}: {v}" for k, v in s["origin"].items()]
    lines += ["", "## Dropped", ""]
    lines += [f"- {k}: {v}" for k, v in s["dropped"].items()] or ["- none"]
    lines += [
        "",
        "## Transcript style vs the rest of the corpus",
        "",
        f"- cased: **{style['cased_ratio']:.1%}** · punctuated: **{style['punctuated_ratio']:.1%}**",
        f"- apostrophes / 1k words: **{s['apostrophes_per_1k_words']}** · "
        f"digits / 1k words: **{s['digits_per_1k_words']}**",
        f"- charset: `{s['charset']}`",
        "",
        "## Tokenizer coverage (pretrained BPE, vocab 13088)",
        "",
        f"- tokens/word: **{cov.get('tokens_per_word')}** · "
        f"`<unk>` rate: **{cov.get('unk_rate')}**",
        f"- Verdict: **{s.get('tokenizer_verdict')}**",
        "",
    ]
    if style["cased_ratio"] < 0.2 or style["punctuated_ratio"] < 0.2:
        lines.append(
            "> **Style mismatch, as expected.** These transcripts are CTC output: lowercase and "
            "unpunctuated. Case and punctuation are deliberately *not* invented here — these are "
            "diarization spans, many starting mid-sentence, so sentence-initial capitalization "
            "would teach the model to capitalize arbitrarily. Control the blend by capping "
            "accepted hours at gate time (`--max-hours`), not by `weight` (which cannot "
            "down-weight — see `_apply_weights`), and report `bench` both exact and "
            "`--normalize-scoring`."
        )
    return "\n".join(lines) + "\n"
