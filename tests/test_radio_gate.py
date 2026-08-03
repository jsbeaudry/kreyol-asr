"""Deciding which pseudo-labels are worth training on.

The gate exists because the corpus transcripts come from a model scoring ~21% CER on
its own test set — plausibly worse than the one being trained. Two properties matter
most here: the agreement CER must measure *errors* rather than spelling conventions,
and the thresholds must come from the measured distribution rather than from constants
that happened to look reasonable on one corpus.
"""

import json

import pytest

from kreyol_asr.radio_gate import (_band, _keep_fraction, band_accuracy, gate,
                                   norm_for_agreement, resolve_model, segment_cer)


def row(name, *, cer, conf, text="bonjou tout moun", dur=5.0, ours="bonjou tout moun"):
    return {"file_name": f"wav/{name}.wav", "text": text, "speaker": "rec-a:SPEAKER_00",
            "recording_id": "rec-a", "corpus_split": "train", "start_ms": 0,
            "end_ms": int(dur * 1000), "duration": dur, "confidence": conf,
            "cer": cer, "band": _band(cer), "ours": ours, "origin": "csv",
            "cer_reliable": bool(ours.strip())}


@pytest.fixture
def agreement(tmp_path):
    def build(rows):
        d = tmp_path / "agreement"
        d.mkdir(exist_ok=True)
        with open(d / "per_segment.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (tmp_path / "af").mkdir(exist_ok=True)
        return tmp_path / "af", d
    return build


# --- the normalizer, which is the whole ballgame ---------------------------

def test_clitic_spelling_is_not_counted_as_an_error():
    """We emit `n'ap`, the corpus writes `n ap`.

    Stripping punctuation alone turns `n'ap` into `nap`, which still mismatches
    `n ap` — so without un-joining the clitic first, the gate rejects perfectly
    good labels for a reason that has nothing to do with the audio. These are the
    most frequent tokens in the language, so the effect is not marginal.
    """
    assert segment_cer("n ap ale", "n'ap ale") == 0.0
    assert segment_cer("ki t ap pale", "ki t'ap pale") == 0.0


def test_case_and_punctuation_are_not_counted_as_errors():
    assert segment_cer("bonjou zanmi", "Bonjou, zanmi.") == 0.0


def test_real_disagreement_still_scores():
    assert segment_cer("bonjou tout moun", "bonswa tout moun") > 0.0


def test_empty_reference_is_maximally_bad_not_a_crash():
    assert segment_cer("", "yon bagay") == 1.0


def test_lang_tags_do_not_leak_into_the_score():
    """NeMo can append `<ht-HT>` to a hypothesis; that is formatting, not an error."""
    assert segment_cer("bonjou zanmi", "bonjou zanmi <ht-HT>") == 0.0


def test_normalizer_is_idempotent():
    t = norm_for_agreement("N'ap ale, zanmi.")
    assert norm_for_agreement(t) == t


# --- bands -----------------------------------------------------------------

def test_band_edges_are_half_open():
    assert _band(0.05) == "0.05-0.10", "an edge value must land in exactly one band"
    assert _band(0.0499) == "0.02-0.05"
    assert _band(0.0) == "0.00-0.02"
    assert _band(1.0) == "0.60-1.01"


# --- threshold derivation --------------------------------------------------

def test_thresholds_come_from_the_data_not_from_constants(agreement):
    """Two corpora with different confidence distributions must get different cuts."""
    tight = [row(f"t{i}", cer=0.1, conf=0.95 + i * 0.0005) for i in range(100)]
    loose = [row(f"l{i}", cer=0.1, conf=0.20 + i * 0.0070) for i in range(100)]

    af, ag = agreement(tight)
    a = gate(af, ag)
    af, ag = agreement(loose)
    b = gate(af, ag)

    assert a["thresholds"]["min_confidence"] != b["thresholds"]["min_confidence"]
    assert a["thresholds"]["min_confidence"] > b["thresholds"]["min_confidence"]


def test_low_confidence_and_high_disagreement_are_rejected(agreement):
    rows = ([row(f"good{i}", cer=0.20, conf=0.97) for i in range(50)]
            + [row("doubtful", cer=0.20, conf=0.10)]
            + [row("disagree", cer=0.90, conf=0.99)])
    af, ag = agreement(rows)
    s = gate(af, ag)
    kept = {json.loads(l)["file_name"]
            for l in (af / "metadata.jsonl").read_text().splitlines()}
    assert "wav/doubtful.wav" not in kept
    assert "wav/disagree.wav" not in kept
    assert s["tiers"]["reject_disagreement"]["clips"] == 1


def test_the_informative_band_is_what_survives(agreement):
    """Near-zero CER is safe but teaches nothing; the band is where the signal is."""
    rows = ([row(f"same{i}", cer=0.0, conf=0.97) for i in range(20)]
            + [row(f"info{i}", cer=0.25, conf=0.97) for i in range(20)])
    af, ag = agreement(rows)
    s = gate(af, ag, consensus_share=1.0)
    assert s["tiers"]["informative"]["clips"] == 20
    assert s["tiers"]["consensus"]["clips"] == 20


def test_consensus_is_capped_and_deterministic(agreement):
    rows = [row(f"same{i}", cer=0.0, conf=0.97) for i in range(200)]
    af, ag = agreement(rows)
    a = gate(af, ag, consensus_share=0.5)
    first = (af / "metadata.jsonl").read_text()
    b = gate(af, ag, consensus_share=0.5)
    assert (af / "metadata.jsonl").read_text() == first, "same seed, same subset"
    assert a["tiers"]["consensus"]["clips"] == b["tiers"]["consensus"]["clips"]
    assert 60 < a["tiers"]["consensus"]["clips"] < 140, "roughly half of 200"


def test_keep_fraction_is_stable_and_roughly_uniform():
    names = [f"wav/{i}.wav" for i in range(2000)]
    kept = [n for n in names if _keep_fraction(n, 0.25, "1804")]
    assert 0.20 < len(kept) / len(names) < 0.30
    assert kept == [n for n in names if _keep_fraction(n, 0.25, "1804")]


def test_garbage_transcripts_never_reach_the_gpu_budget(agreement):
    """Empty text and repetition loops are `prepare`'s rules, applied early."""
    loop = "pou " * 60  # ~240 chars over 5 s = 48 cps
    rows = ([row(f"ok{i}", cer=0.2, conf=0.97) for i in range(30)]
            + [row("empty", cer=0.2, conf=0.99, text="  ")]
            + [row("loop", cer=0.2, conf=0.99, text=loop)])
    af, ag = agreement(rows)
    s = gate(af, ag)
    assert s["tiers"]["reject_empty"]["clips"] == 1
    assert s["tiers"]["reject_chars_per_second"]["clips"] == 1


def test_max_hours_caps_the_blend(agreement):
    rows = [row(f"r{i}", cer=0.2, conf=0.97, dur=60.0) for i in range(100)]
    af, ag = agreement(rows)
    s = gate(af, ag, max_hours=0.5)
    assert s["accepted_hours"] <= 0.55
    assert s["tiers"]["reject_over_max_hours"]["clips"] > 0


def test_gate_refuses_to_write_an_empty_corpus(agreement):
    af, ag = agreement([row(f"r{i}", cer=0.99, conf=0.99) for i in range(20)])
    with pytest.raises(RuntimeError, match="accepted nothing"):
        gate(af, ag)


def test_gate_writes_metadata_jsonl_not_metadata_all(agreement):
    """`_iter_localdir` only accepts metadata.jsonl — that is the safety interlock."""
    af, ag = agreement([row(f"r{i}", cer=0.2, conf=0.97) for i in range(20)])
    gate(af, ag)
    assert (af / "metadata.jsonl").exists()
    assert not (af / "metadata.all.jsonl").exists()


# --- human verdicts --------------------------------------------------------

def test_band_accuracy_counts_only_checked_clips():
    rows = [row("a", cer=0.03, conf=0.9), row("b", cer=0.03, conf=0.9),
            row("c", cer=0.50, conf=0.9)]
    verdicts = {"wav/a.wav": "corpus", "wav/b.wav": "ours"}
    acc = band_accuracy(rows, verdicts)
    assert acc["0.02-0.05"] == {"checked": 2, "label_ok": 1, "accuracy": 0.5}
    assert "0.45-0.60" not in acc, "an unchecked band has no measurement"


# --- our decoder returning nothing is not a label error --------------------

def test_empty_hypothesis_is_not_counted_as_disagreement(agreement):
    """Measured on the first 5 h: our decoder emits nothing on 11.7% of clips, and
    23.6% of clips under 2 s versus 1.5% above 10 s — a cache-aware streaming
    warm-up artifact. Scored as CER 1.0 it would strip short clips wholesale."""
    rows = ([row(f"ok{i}", cer=0.20, conf=0.97) for i in range(40)]
            + [row(f"mute{i}", cer=1.0, conf=0.97, ours="", dur=1.5) for i in range(20)])
    af, ag = agreement(rows)
    s = gate(af, ag)
    assert s["tiers"]["unscored"]["clips"] == 20
    assert "reject_disagreement" not in s["tiers"]
    kept = {json.loads(l)["file_name"]
            for l in (af / "metadata.jsonl").read_text().splitlines()}
    assert "wav/mute0.wav" in kept


def test_unscored_clips_still_need_confidence(agreement):
    """With no CER evidence, confidence is the only signal left — so it must bind."""
    rows = ([row(f"ok{i}", cer=0.20, conf=0.97) for i in range(40)]
            + [row("mute_bad", cer=1.0, conf=0.30, ours="")])
    af, ag = agreement(rows)
    s = gate(af, ag)
    assert s["tiers"].get("unscored", {"clips": 0})["clips"] == 0
    assert ("reject_unscored_low_confidence" in s["tiers"]
            or "reject_low_confidence" in s["tiers"])


def test_reliability_is_derived_when_the_field_is_absent(agreement):
    """Predictions written before this field existed must still be handled."""
    rows = [row(f"ok{i}", cer=0.20, conf=0.97) for i in range(40)]
    rows += [row("mute", cer=1.0, conf=0.97, ours="")]
    for r in rows:
        del r["cer_reliable"]
    af, ag = agreement(rows)
    s = gate(af, ag)
    assert s["tiers"]["unscored"]["clips"] == 1


# --- model resolution ------------------------------------------------------

def test_local_nemo_path_is_passed_through(tmp_path):
    p = tmp_path / "model.nemo"
    p.write_bytes(b"x")
    assert resolve_model(str(p)) == str(p)


def test_missing_local_path_is_not_mistaken_for_a_repo(tmp_path):
    with pytest.raises(RuntimeError, match="not a Hub repo id"):
        resolve_model(str(tmp_path / "absent.nemo"))


def test_private_repo_failure_names_the_token(monkeypatch):
    """NeMo's own error for a repo id points nowhere near the cause, and the
    published checkpoint is private by default — so say so."""
    import kreyol_asr.lang_slot as ls

    monkeypatch.setattr(ls, "fetch_base_nemo",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("401")))
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        resolve_model("me/private-model", token=None)


def test_token_present_means_no_misleading_token_hint(monkeypatch):
    import kreyol_asr.lang_slot as ls

    monkeypatch.setattr(ls, "fetch_base_nemo",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    with pytest.raises(RuntimeError) as e:
        resolve_model("me/model", token="hf_x")
    assert "HF_TOKEN is not set" not in str(e.value)


def test_both_ok_counts_as_a_usable_label():
    """Spelling-only differences are not label noise."""
    rows = [row("a", cer=0.03, conf=0.9)]
    assert band_accuracy(rows, {"wav/a.wav": "both-ok"})["0.02-0.05"]["accuracy"] == 1.0
