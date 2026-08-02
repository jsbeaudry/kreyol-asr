"""Synthetic (TTS) audio trains but never evaluates.

Scoring an ASR model on synthesized speech measures it on the narrow acoustic
distribution it was trained on. The reported WER then does not survive contact
with real recordings.
"""

import pytest

from kreyol_asr.config import Source
from kreyol_asr.datasets import _explain_hub_error, _split

SPEC = {"train": 0.8, "val": 0.1, "test": 0.1, "speaker_disjoint": True, "seed": 1804}


def rec(i, synthetic=False, speaker=None, dur=4.0):
    return {
        "audio_filepath": f"/tmp/{'syn' if synthetic else 'real'}{i}.wav",
        "duration": dur,
        "text": "bonjou",
        "_source": "syn" if synthetic else "real",
        "_speaker": speaker or f"spk{i % 12}",
        "_synthetic": synthetic,
    }


def test_synthetic_clips_never_reach_val_or_test():
    records = [rec(i) for i in range(60)] + [rec(i, synthetic=True) for i in range(200)]
    out = _split(records, SPEC)

    for name in ("val", "test"):
        assert out[name], f"{name} split is empty"
        assert all(not r["_synthetic"] for r in out[name]), \
            f"synthetic audio leaked into {name}"
    assert any(r["_synthetic"] for r in out["train"]), "synthetic audio should still train"


def test_every_synthetic_clip_is_kept_for_training():
    records = [rec(i) for i in range(40)] + [rec(i, synthetic=True) for i in range(75)]
    out = _split(records, SPEC)
    kept = sum(1 for r in out["train"] if r["_synthetic"])
    assert kept == 75, "synthetic clips must not be silently dropped"

    total = sum(len(v) for v in out.values())
    assert total == len(records), "no clip may vanish in the split"


def test_all_synthetic_corpus_is_rejected():
    records = [rec(i, synthetic=True) for i in range(50)]
    with pytest.raises(RuntimeError, match="no real audio to evaluate"):
        _split(records, SPEC)


def test_split_without_synthetic_sources_is_unchanged():
    records = [rec(i) for i in range(60)]
    out = _split(records, SPEC)
    assert sum(len(v) for v in out.values()) == 60
    assert out["val"] and out["test"]


def test_source_defaults_to_real():
    assert Source(repo_id="x/y").synthetic is False
    assert Source(repo_id="x/y", synthetic=True).synthetic is True


# --- Hub error explanations ------------------------------------------------
# Both of these cost real debugging time; the messages must name the cause.

def test_storage_limit_error_is_explained():
    err = Exception("403 Forbidden: Private repository storage limit reached for someone")
    msg = str(_explain_hub_error("me/ds", err))
    assert "storage limit" in msg.lower()
    assert "metadata still resolves" in msg.lower()


def test_storage_limit_is_recovered_when_datasets_masks_it(monkeypatch):
    """The regression that actually bit: `datasets` reports a 403 as a connection error.

    Only a direct Hub read surfaces the real status, so the explainer probes and
    must still name the storage limit rather than blaming the network.
    """
    import kreyol_asr.datasets as D

    monkeypatch.setattr(D, "_probe_repo_access", lambda repo, token:
                        "HfHubHTTPError: 403 Forbidden: Private repository storage "
                        "limit reached for someone")
    masked = Exception("LocalEntryNotFoundError: An error happened while trying to "
                       "locate the file on the Hub. Please check your connection")
    msg = str(D._explain_hub_error("me/ds", masked, token="t"))
    assert "storage limit" in msg.lower()
    assert "connection" not in msg.lower(), "must not repeat the misleading advice"


def test_probe_is_skipped_when_the_error_already_says_it(monkeypatch):
    import kreyol_asr.datasets as D

    called = []
    monkeypatch.setattr(D, "_probe_repo_access", lambda r, t: called.append(1) or "")
    D._explain_hub_error("me/ds", Exception("403: storage limit reached"), token="t")
    assert not called, "no extra network round-trip when the cause is already known"


def test_unwritable_hf_home_is_explained(monkeypatch):
    """Must hold as root too — containers run as root, and os.access() lies there."""
    monkeypatch.setenv("HF_HOME", "/proc/cannot/create/here/.hf")
    msg = str(_explain_hub_error("me/ds", Exception("LocalEntryNotFoundError: check your connection")))
    assert "HF_HOME" in msg and "not usable" in msg


def test_writable_hf_home_is_not_blamed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr("kreyol_asr.datasets._probe_repo_access", lambda r, t: "")
    msg = str(_explain_hub_error("me/ds", Exception("some unrelated failure")))
    assert "HF_HOME" not in msg, "a healthy cache dir must not be reported as the cause"


def test_hf_home_check_leaves_no_litter(tmp_path, monkeypatch):
    from kreyol_asr.datasets import _hf_home_problem

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert _hf_home_problem() is None
    assert list((tmp_path / "hf").iterdir()) == [], "probe file must be cleaned up"


def test_401_mentions_the_shadowed_token_trap(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    # Stub the probe: it makes a live Hub call, so without this the assertion
    # depends on the account's current state rather than on the code under test.
    monkeypatch.setattr("kreyol_asr.datasets._probe_repo_access", lambda r, t: "")
    msg = str(_explain_hub_error("me/ds", Exception("401 Client Error RepositoryNotFound")))
    assert "HF_TOKEN" in msg and "shadows" in msg
