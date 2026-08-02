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
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from . import SAMPLE_RATE
from .config import DataConfig, Source
from .text import TokenizerCoverage, normalize, out_of_charset, style_stats

console = Console()

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


def _to_mono_16k(array: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:  # (samples, channels) or (channels, samples)
        arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[1] else 0)
    if sr != SAMPLE_RATE:
        import soxr  # band-limited resampling; np.interp would alias into the mel bins

        arr = soxr.resample(arr, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def _decode(entry: Any) -> tuple[np.ndarray, int] | None:
    """Decode one audio cell to (samples, sample_rate).

    We decode with soundfile rather than letting `datasets` do it: datasets>=4
    routes decoding through torchcodec (which needs ffmpeg), while datasets<4
    returns a plain array dict. Handling the raw bytes ourselves works on both
    and keeps resampling under our control.
    """
    import io

    if entry is None:
        return None
    # datasets<4 with decode=True
    if isinstance(entry, dict) and entry.get("array") is not None:
        return np.asarray(entry["array"]), int(entry.get("sampling_rate") or SAMPLE_RATE)
    # decode=False -> {"path": ..., "bytes": ...}
    if isinstance(entry, dict):
        if entry.get("bytes"):
            data, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32", always_2d=False)
            return data, sr
        if entry.get("path"):
            data, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
            return data, sr
        return None
    if isinstance(entry, (str, Path)):
        data, sr = sf.read(str(entry), dtype="float32", always_2d=False)
        return data, sr
    # datasets>=4 torchcodec AudioDecoder, if it is installed after all
    if hasattr(entry, "get_all_samples"):
        s = entry.get_all_samples()
        return np.asarray(s.data).squeeze(), int(s.sample_rate)
    return None


def _hf_home_problem() -> str | None:
    """Return why HF_HOME can't be used as a cache, or None if it's fine.

    Must actually attempt the write. `os.access()` consults the real uid's
    permission bits and returns True for root on paths root still cannot use —
    and containers run as root, which is exactly where a stale pod path like
    /workspace/.hf on a laptop needs to be caught.
    """
    home = os.environ.get("HF_HOME")
    if not home:
        return None
    probe = Path(home) / ".kreyol_write_test"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("x")
        probe.unlink()
        return None
    except OSError as e:
        return str(e)


def _probe_repo_access(repo_id: str, token: str | None) -> str:
    """Read one byte of one file, and report what actually goes wrong.

    `datasets` collapses every Hub failure into "Couldn't find ... check your
    connection", which hides 403s. Reading through HfFileSystem surfaces the real
    status line, so the message the user sees names the true cause.
    """
    try:
        from huggingface_hub import HfApi, HfFileSystem

        info = HfApi(token=token).dataset_info(repo_id, files_metadata=True)
        target = next((s.rfilename for s in info.siblings
                       if s.rfilename.endswith((".parquet", ".wav", ".mp3", ".csv"))), None)
        if not target:
            return ""
        with HfFileSystem(token=token).open(f"datasets/{repo_id}/{target}") as fh:
            fh.read(1)
        return ""  # reads are fine; the failure is something else
    except Exception as e:  # noqa: BLE001 - this IS the diagnostic
        return f"{type(e).__name__}: {e}"


def _explain_hub_error(repo_id: str, err: Exception, token: str | None = None) -> RuntimeError:
    """Turn Hub failures into something you can act on.

    Three failure modes cost real debugging time on this project:

    * A private-storage overage returns 403 on file *reads* while metadata calls
      keep working, so listings succeed and only the download fails — and
      `datasets` reports it as a connection problem.
    * An unwritable HF_HOME (e.g. a pod path like /workspace/.hf carried into a
      laptop .env) also surfaces as "check your connection", which it is not.
    * A duplicated HF_TOKEN line in .env silently shadows the real token.
    """
    text = f"{type(err).__name__}: {err}"
    if "storage limit" not in text.lower():
        text += " | probe -> " + _probe_repo_access(repo_id, token)
    if "storage limit" in text.lower() or "403" in text and "private" in text.lower():
        return RuntimeError(
            f"{repo_id}: Hugging Face is blocking file reads because the account's "
            f"private-repo storage limit is reached. Metadata still resolves, which is "
            f"why this looks like a permissions bug. Free up private storage or upgrade "
            f"the plan, then re-run. See https://huggingface.co/docs/hub/storage-limits"
        )
    unwritable = _hf_home_problem()
    if unwritable:
        return RuntimeError(
            f"{repo_id}: HF_HOME={os.environ.get('HF_HOME')!r} is not usable, so nothing "
            f"can be cached ({unwritable}). Pod paths like /workspace/.hf do not exist on "
            f"a laptop — comment HF_HOME out in .env when running locally. "
            f"(Underlying error: {text[:160]})"
        )
    if "401" in text or "RepositoryNotFound" in text:
        return RuntimeError(
            f"{repo_id}: not found or not readable with the current token. If it is "
            f"private, set HF_TOKEN in .env. Note a duplicated HF_TOKEN line in .env "
            f"silently shadows the real one. (Underlying error: {text[:160]})"
        )
    return RuntimeError(f"{repo_id}: could not load — {text[:300]}")


AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def _is_audiofolder(repo_id: str, token: str | None) -> bool:
    """True for repos that ship loose audio files instead of parquet shards."""
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception:  # noqa: BLE001 - fall back to the datasets path
        return False
    return (not any(f.endswith(".parquet") for f in files)
            and any(f.lower().endswith(AUDIO_EXT) for f in files))


def _iter_audiofolder(src: Source, token: str | None,
                      limit: int | None) -> Iterator[dict[str, Any]]:
    """Read a loose-file repo without going through `datasets`.

    `datasets` 5.x routes audiofolder repos through an *encode* step that needs
    torchcodec (`cast_column(Audio(decode=False))` raises ImportError). We only
    ever want the raw bytes — soundfile decodes them downstream — so read the
    metadata table and pull each file straight off the Hub instead.
    """
    import csv
    import io as _io

    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    files = api.list_repo_files(src.repo_id, repo_type="dataset")
    audio_files = [f for f in files if f.lower().endswith(AUDIO_EXT)]
    meta_name = next((f for f in files
                      if f.rsplit("/", 1)[-1] in ("metadata.csv", "metadata.jsonl")), None)
    if not meta_name:
        raise RuntimeError(
            f"{src.repo_id}: loose audio files but no metadata.csv/jsonl — cannot "
            f"recover transcripts. Convert the repo to parquet, or add a metadata table."
        )

    with fs.open(f"datasets/{src.repo_id}/{meta_name}") as fh:
        blob = fh.read().decode("utf-8", "replace")

    if meta_name.endswith(".jsonl"):
        recs = [json.loads(l) for l in blob.splitlines() if l.strip()]
        fields = list(recs[0]) if recs else []
    else:
        rdr = csv.DictReader(_io.StringIO(blob))
        recs, fields = list(rdr), list(rdr.fieldnames or [])

    file_key = next((c for c in fields
                     if c.lower() in ("file_name", "filename", "path", "audio", "file")), None)
    if not file_key:
        raise RuntimeError(f"{src.repo_id}: {meta_name} has no file-name column ({fields})")
    text_key = _pick(src.text_column, fields, TEXT_CANDIDATES, "text")
    spk_key = _pick(src.speaker_column, fields, SPEAKER_CANDIDATES, "speaker", required=False)
    console.print(f"  audiofolder -> file={file_key!r} text={text_key!r} "
                  f"speaker={spk_key!r}  ({len(recs)} rows, {len(audio_files)} files)")

    # metadata paths are relative to the file's directory; index by basename so
    # "data/x.wav" and "x.wav" both resolve.
    by_base = {f.rsplit("/", 1)[-1]: f for f in audio_files}
    for i, rec in enumerate(recs):
        if limit and i >= limit:
            return
        name = str(rec.get(file_key, "")).rsplit("/", 1)[-1]
        path = by_base.get(name)
        if not path:
            continue
        try:
            with fs.open(f"datasets/{src.repo_id}/{path}") as fh:
                raw = fh.read()
        except Exception:  # noqa: BLE001 - one unreadable clip must not kill the run
            continue
        yield {
            "index": i,
            "audio": {"bytes": raw, "path": path},
            "raw_text": rec.get(text_key),
            "speaker": str(rec[spk_key]) if spk_key and rec.get(spk_key) is not None else None,
        }


def _iter_source(src: Source, token: str | None, limit: int | None) -> Iterator[dict[str, Any]]:
    from datasets import Audio, load_dataset

    kind = "synthetic" if src.synthetic else "real"
    console.print(f"[bold]Loading[/bold] {src.repo_id} (split={src.split}, {kind})")

    if _is_audiofolder(src.repo_id, token):
        yield from _iter_audiofolder(src, token, limit)
        return

    # A bounded run must also bound the download. Non-streaming load_dataset
    # fetches the entire repo before .select() ever runs, so `--limit 20` would
    # still pull gigabytes — which is exactly what a smoke test must not do.
    streaming = limit is not None
    try:
        ds = load_dataset(src.repo_id, src.config, split=src.split, token=token,
                          streaming=streaming)
    except Exception as e:  # noqa: BLE001 - re-raised with a usable explanation
        raise _explain_hub_error(src.repo_id, e, token) from e

    if streaming:
        columns = list(ds.features) if ds.features else list(next(iter(ds)).keys())
        n_rows = "streaming"
    else:
        columns = list(ds.column_names)
        n_rows = f"{len(ds)} rows"

    audio_col = _pick(src.audio_column, columns, AUDIO_CANDIDATES, "audio")
    text_col = _pick(src.text_column, columns, TEXT_CANDIDATES, "text")
    speaker_col = _pick(src.speaker_column, columns, SPEAKER_CANDIDATES, "speaker",
                        required=False)
    console.print(f"  columns -> audio={audio_col!r} text={text_col!r} "
                  f"speaker={speaker_col!r}  ({n_rows})")

    ds = ds.cast_column(audio_col, Audio(decode=False))
    if limit:
        ds = ds.take(limit)

    for i, row in enumerate(ds):
        yield {
            "index": i,
            "audio": row[audio_col],
            "raw_text": row[text_col],
            "speaker": str(row[speaker_col]) if speaker_col else None,
        }


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
            task = bar.add_task(f"  {src.repo_id}", total=None)
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
    """Split, keeping TTS-generated clips out of val and test.

    A model scored on synthesized speech is scored on the narrow acoustic
    distribution it was trained on, which reports a WER the model will not
    reproduce on real recordings. Synthetic clips train; real clips evaluate.
    """
    synth = [r for r in records if r.get("_synthetic")]
    real = [r for r in records if not r.get("_synthetic")]

    if synth and not real:
        raise RuntimeError(
            "Every source is marked `synthetic: true`, so there is no real audio to "
            "evaluate on. Add at least one real-recording source, or drop the flag if "
            "these really are recordings."
        )
    if not synth:
        return _split_real(real, spec)

    out = _split_real(real, spec)
    out["train"].extend(synth)
    sh = sum(r["duration"] for r in synth) / 3600
    rh = sum(r["duration"] for r in real) / 3600
    console.print(f"Held {sh:.2f} h of synthetic audio to train only; "
                  f"val/test drawn from {rh:.2f} h of real audio")
    _split.synthetic_hours = sh
    _split.real_hours = rh
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
_split.real_hours = 0.0


def _render_report(s: dict[str, Any]) -> str:
    cov = s["tokenizer_coverage"]
    style = s["style"]
    lines = [
        "# Data report",
        "",
        f"- Language tag: `{s['language']}`",
        f"- Total: **{s['total_hours']} h** across **{s['total_clips']}** clips",
        f"- Split mode: {s['split_mode']}",
        (f"- Real: **{s['real_hours']} h** (trains and evaluates) · "
         f"Synthetic: **{s['synthetic_hours']} h** (trains only — TTS audio is kept "
         f"out of val/test so the WER reflects real recordings)"
         if s.get("synthetic_hours") else "- All sources are real recordings"),
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
