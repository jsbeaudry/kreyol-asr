"""Local audiofolders as a dataset source, and pseudo-labeled data as a third bucket.

Radio Haiti is 6 GB of Zenodo zips segmented on the training pod. Round-tripping the
slices through a private Hub repo would cost an upload per gate iteration and buy
nothing, so `prepare` learned to read a directory. The safety property that comes with
it: only the *gated* metadata file is accepted, so machine transcripts cannot reach
training by accident.
"""

import json

import pytest

from kreyol_asr.config import DataConfig, Source
from kreyol_asr.datasets import _apply_weights, _split
from kreyol_common.sources import _iter_localdir

SPEC = {"train": 0.8, "val": 0.1, "test": 0.1, "speaker_disjoint": True, "seed": 1804}


def rec(i, *, synthetic=False, pseudo=False, speaker=None, dur=4.0):
    kind = "pseudo" if pseudo else ("syn" if synthetic else "real")
    return {
        "audio_filepath": f"/tmp/{kind}{i}.wav",
        "duration": dur,
        "text": "bonjou",
        "_source": kind,
        "_speaker": speaker or f"{kind}-spk{i % 12}",
        "_synthetic": synthetic,
        "_pseudo": pseudo,
        "_train_only": synthetic or pseudo,
    }


@pytest.fixture
def audiofolder(tmp_path):
    def build(name="metadata.jsonl", rows=None):
        root = tmp_path / "radio-haiti-inter"
        (root / "wav").mkdir(parents=True, exist_ok=True)
        rows = rows if rows is not None else [
            {"file_name": f"wav/clip{i}.wav", "text": f"fraz {i}",
             "speaker": f"rec-a:SPEAKER_0{i % 2}", "duration": 4.0}
            for i in range(5)
        ]
        for r in rows:
            (root / r["file_name"]).write_bytes(b"RIFF")
        with open(root / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return root
    return build


# --- Source ----------------------------------------------------------------

def test_source_needs_exactly_one_location():
    with pytest.raises(ValueError, match="exactly one"):
        Source()
    with pytest.raises(ValueError, match="exactly one"):
        Source(repo_id="me/ds", local_dir="/tmp/x")
    assert Source(repo_id="me/ds").repo_id == "me/ds"
    assert Source(local_dir="/tmp/x").local_dir == "/tmp/x"


def test_local_slug_is_the_directory_name():
    """`slug` labels the source in data_report.md and on the model card, and it used
    to be `repo_id.replace(...)` — which is None for a local source."""
    assert Source(local_dir="/workspace/corpora/radio-haiti/radio-haiti-inter").slug \
        == "radio-haiti-inter"
    assert Source(repo_id="me/ds").slug == "me__ds"


def test_train_only_covers_both_reasons():
    assert Source(repo_id="a/b").train_only is False
    assert Source(repo_id="a/b", synthetic=True).train_only is True
    assert Source(local_dir="/tmp/x", pseudo_labeled=True).train_only is True


def test_kind_distinguishes_the_two_train_only_reasons():
    assert Source(repo_id="a/b").kind == "real"
    assert Source(repo_id="a/b", synthetic=True).kind == "synthetic"
    assert Source(local_dir="/tmp/x", pseudo_labeled=True).kind == "pseudo-labeled"


def test_config_loads_a_local_source(tmp_path):
    cfg = tmp_path / "d.yaml"
    cfg.write_text(
        "language: ht-HT\n"
        "sources:\n"
        "  - repo_id: me/real\n"
        "  - local_dir: /workspace/corpora/radio-haiti/radio-haiti-inter\n"
        "    pseudo_labeled: true\n"
    )
    loaded = DataConfig.load(cfg)
    assert [s.kind for s in loaded.sources] == ["real", "pseudo-labeled"]


# --- _iter_localdir --------------------------------------------------------

def test_localdir_yields_the_iter_source_contract(audiofolder):
    root = audiofolder()
    items = list(_iter_localdir(Source(local_dir=str(root)), None))
    assert len(items) == 5
    first = items[0]
    assert set(first) == {"index", "audio", "raw_text", "speaker"}
    # A path, not bytes: `_decode` opens it lazily, exactly as for Hub sources.
    assert first["audio"]["path"].endswith("wav/clip0.wav")
    assert first["raw_text"] == "fraz 0"
    assert first["speaker"] == "rec-a:SPEAKER_00"


def test_localdir_honours_limit(audiofolder):
    root = audiofolder()
    assert len(list(_iter_localdir(Source(local_dir=str(root)), 2))) == 2


def test_localdir_skips_rows_whose_audio_is_missing(audiofolder):
    root = audiofolder()
    (root / "wav" / "clip2.wav").unlink()
    assert len(list(_iter_localdir(Source(local_dir=str(root)), None))) == 4


def test_ungated_ingest_is_refused_by_name(audiofolder):
    """metadata.all.jsonl is every candidate, including the ones the gate rejects.

    Accepting it here would let machine transcripts reach training without ever
    passing the confidence or agreement filters — so the error names the fix.
    """
    root = audiofolder(name="metadata.all.jsonl")
    with pytest.raises(RuntimeError, match="radio gate"):
        list(_iter_localdir(Source(local_dir=str(root)), None))


def test_missing_directory_is_reported_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="not a directory"):
        list(_iter_localdir(Source(local_dir=str(tmp_path / "nope")), None))


# --- splitting -------------------------------------------------------------

def test_pseudo_labeled_clips_never_reach_val_or_test():
    """Scoring against another model's transcripts measures agreement with that
    model, not accuracy — and here that model is the weaker of the two."""
    records = [rec(i) for i in range(60)] + [rec(i, pseudo=True) for i in range(200)]
    out = _split(records, SPEC)
    for name in ("val", "test"):
        assert out[name]
        assert all(not r["_pseudo"] for r in out[name]), f"pseudo-labels leaked into {name}"
    assert any(r["_pseudo"] for r in out["train"])


def test_every_pseudo_clip_is_kept_for_training():
    records = [rec(i) for i in range(40)] + [rec(i, pseudo=True) for i in range(75)]
    out = _split(records, SPEC)
    assert sum(1 for r in out["train"] if r["_pseudo"]) == 75
    assert sum(len(v) for v in out.values()) == len(records)


def test_synthetic_and_pseudo_are_counted_separately():
    """One combined number would hide which risk you are carrying: narrow acoustics
    or noisy labels."""
    records = ([rec(i, dur=10.0) for i in range(40)]
               + [rec(i, synthetic=True, dur=10.0) for i in range(20)]
               + [rec(i, pseudo=True, dur=10.0) for i in range(30)])
    _split(records, SPEC)
    assert _split.synthetic_hours == pytest.approx(200 / 3600)
    assert _split.pseudo_hours == pytest.approx(300 / 3600)
    assert _split.real_hours == pytest.approx(400 / 3600)


def test_a_corpus_with_no_human_labels_is_rejected():
    records = [rec(i, pseudo=True) for i in range(50)]
    with pytest.raises(RuntimeError, match="no real audio to evaluate"):
        _split(records, SPEC)


def test_records_predating_train_only_are_still_held_back():
    """`_train_only` is derived. A record built without it must not leak into test
    because a bookkeeping field was missing."""
    old = [{**rec(i, synthetic=True)} for i in range(50)]
    for r in old:
        del r["_train_only"]
    out = _split([rec(i) for i in range(40)] + old, SPEC)
    assert all(not r["_synthetic"] for r in out["test"])


# --- the weight trap -------------------------------------------------------

def test_weight_below_one_does_not_down_weight():
    """Documenting the current behaviour, not endorsing it.

    `_apply_weights` rounds and clamps to at least one copy, so `weight: 0.5` emits a
    full copy rather than half. Radio Haiti's share of the corpus therefore has to be
    controlled by `radio gate --max-hours`, not by this field. Asserted here so the
    trap is visible in the suite rather than discovered mid training run.
    """
    rows = [{"_source": "rh"} for _ in range(10)]
    assert len(_apply_weights(rows, {"rh": 0.5})) == 10
    assert len(_apply_weights(rows, {"rh": 2.0})) == 20
