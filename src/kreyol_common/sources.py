"""Iterate a Hugging Face dataset repo, whether it ships parquet or loose audio files."""

from __future__ import annotations

import json
from typing import Any, Iterator

from . import console
from .audio import AUDIO_EXT
from .columns import (AUDIO_CANDIDATES, SPEAKER_CANDIDATES, TEXT_CANDIDATES, _pick)
from .hub import _explain_hub_error


def _is_audiofolder(repo_id: str, token: str | None) -> bool:
    """True for repos that ship loose audio files instead of parquet shards."""
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception:  # noqa: BLE001 - fall back to the datasets path
        return False
    return (not any(f.endswith(".parquet") for f in files)
            and any(f.lower().endswith(AUDIO_EXT) for f in files))


def _iter_audiofolder(src, token: str | None, limit: int | None) -> Iterator[dict[str, Any]]:
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


def _iter_localdir(src, limit: int | None) -> Iterator[dict[str, Any]]:
    """Read an audiofolder off local disk — same contract as `_iter_audiofolder`.

    Used for corpora too large or too licence-encumbered to round-trip through the
    Hub. `_decode` accepts `{"path": ...}`, so no audio is read here; the caller
    decodes lazily exactly as it does for Hub sources.
    """
    import csv
    from pathlib import Path

    root = Path(src.local_dir)
    if not root.is_dir():
        raise RuntimeError(f"{root} is not a directory — nothing to prepare.")

    meta = next((root / n for n in ("metadata.jsonl", "metadata.csv") if (root / n).exists()), None)
    if meta is None:
        # metadata.all.jsonl (every candidate) deliberately does NOT satisfy this:
        # ungated pseudo-labels must not be trainable by accident.
        raise RuntimeError(
            f"{root}: no metadata.jsonl/metadata.csv. If this is the Radio Haiti "
            f"corpus, run `kreyol-asr radio gate` — the ungated ingest writes "
            f"metadata.all.jsonl, which is not accepted here on purpose."
        )

    if meta.suffix == ".jsonl":
        recs = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
        fields = list(recs[0]) if recs else []
    else:
        rdr = csv.DictReader(meta.open())
        recs, fields = list(rdr), list(rdr.fieldnames or [])

    file_key = next((c for c in fields
                     if c.lower() in ("file_name", "filename", "path", "audio", "file")), None)
    if not file_key:
        raise RuntimeError(f"{meta}: no file-name column ({fields})")
    text_key = _pick(src.text_column, fields, TEXT_CANDIDATES, "text")
    spk_key = _pick(src.speaker_column, fields, SPEAKER_CANDIDATES, "speaker", required=False)
    console.print(f"  localdir -> file={file_key!r} text={text_key!r} "
                  f"speaker={spk_key!r}  ({len(recs)} rows)")

    for i, rec in enumerate(recs):
        if limit and i >= limit:
            return
        path = root / str(rec.get(file_key, ""))
        if not path.exists():
            continue
        yield {
            "index": i,
            "audio": {"path": str(path)},
            "raw_text": rec.get(text_key),
            "speaker": str(rec[spk_key]) if spk_key and rec.get(spk_key) is not None else None,
        }


def _iter_source(src, token: str | None, limit: int | None) -> Iterator[dict[str, Any]]:
    from datasets import Audio, load_dataset

    kind = src.kind
    # Branch before `_is_audiofolder`, which would call list_repo_files(None).
    if src.local_dir:
        console.print(f"[bold]Loading[/bold] {src.local_dir} (local audiofolder, {kind})")
        yield from _iter_localdir(src, limit)
        return

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
