"""Parsing the Radio Haiti-Inter release.

Every case here is a real property of the shipped files, verified by reading the
Zenodo zips. The EAF fixture in particular reproduces four traps that a naive
ElementTree walk falls into, each of which corrupts the corpus silently rather
than raising.
"""

import pytest

from kreyol_asr.radio_haiti import (Segment, Word, attach_words, read_eaf_words,
                                    read_segments_csv, restyle_clitics,
                                    unstyle_clitics, words_match_text)

CSV = """hypothesis,ctc_score,confidence,recording_id,segment_start,segment_end,speaker
 depi plis pase dizan lajistis,2300.61,0.96908,rec-a,3440,7946,SPEAKER_00
 si gen ti pa ki fet ane sa a,5397.25,0.97306,rec-a,8047,18459,SPEAKER_01
,101.0,0.51,rec-a,19000,20000,SPEAKER_00
 bad span,1.0,0.9,rec-a,30000,30000,SPEAKER_00
"""

# Traps, in order of how much damage they do:
#   1. `words@SPEAKER_00` / `characters@SPEAKER_00` are CHILD tiers of the
#      transcription tier. Read as transcriptions they duplicate every utterance
#      at word and character granularity.
#   2. TIME_SLOTs are referenced by id and are NOT declared in time order.
#   3. REF_ANNOTATIONs carry no time slots of their own.
#   4. An ANNOTATION_VALUE can be empty.
EAF = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="ASR transcription - model-unknown" FORMAT="3.0">
  <TIME_ORDER>
    <TIME_SLOT TIME_SLOT_ID="ts9" TIME_VALUE="3000"/>
    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="1000"/>
    <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>
    <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="2000"/>
    <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="2600"/>
    <TIME_SLOT TIME_SLOT_ID="ts10" TIME_VALUE="3600"/>
  </TIME_ORDER>
  <TIER TIER_ID="SPEAKER_00" LINGUISTIC_TYPE_REF="transcription">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts4">
        <ANNOTATION_VALUE>bonjou tout moun</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
  <TIER TIER_ID="words@SPEAKER_00" PARENT_REF="SPEAKER_00" LINGUISTIC_TYPE_REF="alignments">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="w1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
        <ANNOTATION_VALUE>bonjou</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="w2" TIME_SLOT_REF1="ts2" TIME_SLOT_REF2="ts3">
        <ANNOTATION_VALUE>tout</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="w3" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
        <ANNOTATION_VALUE>moun</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="w4" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
        <ANNOTATION_VALUE></ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
  <TIER TIER_ID="characters@SPEAKER_00" PARENT_REF="words@SPEAKER_00"
        LINGUISTIC_TYPE_REF="alignments">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="c1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
        <ANNOTATION_VALUE>b</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
  <TIER TIER_ID="word_confidence@SPEAKER_00" PARENT_REF="words@SPEAKER_00"
        LINGUISTIC_TYPE_REF="confidence">
    <ANNOTATION>
      <REF_ANNOTATION ANNOTATION_ID="r1" ANNOTATION_REF="w1">
        <ANNOTATION_VALUE>0.98</ANNOTATION_VALUE>
      </REF_ANNOTATION>
    </ANNOTATION>
  </TIER>
  <TIER TIER_ID="words@SPEAKER_01" PARENT_REF="SPEAKER_01" LINGUISTIC_TYPE_REF="alignments">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="w5" TIME_SLOT_REF1="ts9" TIME_SLOT_REF2="ts10">
        <ANNOTATION_VALUE>mesi</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
</ANNOTATION_DOCUMENT>
"""


@pytest.fixture
def csv_path(tmp_path):
    d = tmp_path / "transcriptions" / "train"
    d.mkdir(parents=True)
    p = d / "rec-a.csv"
    p.write_text(CSV)
    return p


@pytest.fixture
def eaf_path(tmp_path):
    p = tmp_path / "rec-a.eaf"
    p.write_text(EAF)
    return p


# --- CSV -------------------------------------------------------------------

def test_csv_columns_and_units(csv_path):
    segs = read_segments_csv(csv_path)
    a = next(s for s in segs if s.start_ms == 3440)
    assert a.recording_id == "rec-a"
    assert a.corpus_split == "train"
    # ms in the file, seconds on the object — mixing these up shifts every clip.
    assert a.duration == pytest.approx(4.506)
    assert a.confidence == pytest.approx(0.96908)
    # Every hypothesis in the release carries a leading space.
    assert a.text == "depi plis pase dizan lajistis"


def test_zero_length_and_empty_rows_are_not_dropped_silently(csv_path):
    segs = read_segments_csv(csv_path)
    starts = {s.start_ms for s in segs}
    assert 30000 not in starts, "a zero-length span cannot be sliced"
    # An empty hypothesis IS kept here — `resegment` and the gate decide, so the
    # drop is counted in one place rather than vanishing during parsing.
    assert 19000 in starts


def test_segments_are_ordered_by_speaker_then_time(csv_path):
    segs = read_segments_csv(csv_path)
    keys = [(s.speaker, s.start_ms) for s in segs]
    assert keys == sorted(keys), "merge_short depends on this ordering"


# --- speaker namespacing ---------------------------------------------------

def test_speaker_keys_are_namespaced_per_recording():
    """`SPEAKER_00` appears in all 219 recordings; these are diarization labels.

    Unnamespaced, `_split_real` would treat 219 different people as one speaker
    and place that "speaker" in both train and test — destroying the disjointness
    guarantee without any error.
    """
    kw = dict(corpus_split="train", speaker="SPEAKER_00", start_ms=0, end_ms=1000,
              text="a", confidence=0.9)
    a = Segment(recording_id="rec-a", **kw)
    b = Segment(recording_id="rec-b", **kw)
    assert a.speaker == b.speaker
    assert a.speaker_key != b.speaker_key
    assert a.speaker_key == "rec-a:SPEAKER_00"


# --- EAF -------------------------------------------------------------------

def test_word_tiers_only(eaf_path):
    words = read_eaf_words(eaf_path)
    assert set(words) == {"SPEAKER_00", "SPEAKER_01"}
    assert [w.text for w in words["SPEAKER_00"]] == ["bonjou", "tout", "moun"], \
        "character tier or empty annotation leaked into the word list"


def test_time_slots_resolve_out_of_declaration_order(eaf_path):
    """ts9/ts10 are declared first but sit latest in time — real files do this."""
    words = read_eaf_words(eaf_path)
    assert words["SPEAKER_01"][0].start_ms == 3000
    assert words["SPEAKER_00"][0].start_ms == 1000


def test_words_are_time_sorted(eaf_path):
    words = read_eaf_words(eaf_path)
    for spk in words:
        assert words[spk] == sorted(words[spk], key=lambda w: w.start_ms)


def test_attach_words_scopes_to_the_segment(eaf_path):
    words = read_eaf_words(eaf_path)
    seg = Segment("rec-a", "train", "SPEAKER_00", 1000, 2000, "bonjou tout", 0.9)
    (attached,) = attach_words([seg], words)
    assert [w.text for w in attached.words] == ["bonjou", "tout"], \
        "a word past the segment end must not be attached"


def test_words_match_text_guards_against_misaligned_transcripts():
    """If the alignment describes different text, splitting on it pairs audio with
    a transcript that does not belong to it. Better to refuse."""
    ws = (Word("bonjou", 0, 500), Word("tout", 500, 900))
    assert words_match_text(Segment("r", "train", "S", 0, 900, "bonjou tout", 0.9, words=ws))
    assert not words_match_text(
        Segment("r", "train", "S", 0, 900, "yon lot fraz nèt", 0.9, words=ws))
    assert not words_match_text(Segment("r", "train", "S", 0, 900, "bonjou tout", 0.9))


# --- clitic restyling ------------------------------------------------------

def test_clitic_restyle_is_narrow():
    """Only the `ap` progressive — the case text.py documents.

    `m te` and `pou l tande` are space-separated in IPN too, so rewriting them
    would invent an orthography rather than match the rest of the corpus.
    """
    assert restyle_clitics("n ap ale") == "n'ap ale"
    assert restyle_clitics("ki t ap pale") == "ki t'ap pale"
    assert restyle_clitics("m te ale") == "m te ale"
    assert restyle_clitics("pou l tande") == "pou l tande"
    assert restyle_clitics("gwo kap la") == "gwo kap la", "must not fire mid-word"


def test_restyle_round_trips():
    for t in ("n ap ale", "k ap fè", "y ap vini"):
        assert unstyle_clitics(restyle_clitics(t)) == t


# --- spectral bandwidth ----------------------------------------------------

def test_band_energy_finds_the_real_bandwidth(tmp_path):
    """This is how we learn whether the archive is 8 kHz material stored at 16 kHz.

    A whole-file band ratio is uninterpretable on this material: most of the energy
    sits below 1 kHz (rumble and mains hum on 40-year-old tape) and swamps whatever
    happens at 4-8 kHz. Framing and keeping only the loudest frames restricts the
    measurement to speech, which is the thing whose bandwidth we care about.
    """
    import numpy as np
    import soundfile as sf

    from kreyol_asr.radio_haiti import band_energy

    sr, t = 16000, np.arange(16000 * 2) / 16000
    narrow = tmp_path / "narrow.wav"
    wide = tmp_path / "wide.wav"
    sf.write(narrow, (0.2 * np.sin(2 * np.pi * 300 * t)).astype("float32"), sr)
    sf.write(wide, (0.2 * np.sin(2 * np.pi * 6000 * t)).astype("float32"), sr)

    n, w = band_energy(narrow), band_energy(wide)
    assert n["sample_rate"] == sr
    # Not ~1.0: a 25 ms Hann frame is 40 Hz wide, so a pure tone leaks into its
    # neighbours. Dominance is the claim, and the rolloff is the sharper test.
    assert n["e_0_1000"] > 0.8 and n["rolloff95_hz"] < 1000
    assert w["e_6000_8000"] > 0.8 and w["rolloff95_hz"] > 4000


def test_band_energy_declines_to_guess_on_a_stub(tmp_path):
    import numpy as np
    import soundfile as sf

    from kreyol_asr.radio_haiti import band_energy

    p = tmp_path / "tiny.wav"
    sf.write(p, np.zeros(100, dtype="float32"), 16000)
    assert band_energy(p) == {}, "too short to frame — must return nothing, not zeros"


# --- dataset card ----------------------------------------------------------

def test_dataset_card_states_the_licence_and_the_changes(tmp_path, monkeypatch):
    """CC-BY-4.0 requires credit, a licence, AND an indication of changes. The
    corpus is re-segmented, filtered and restyled, so the last one is not optional."""
    from kreyol_asr import radio_haiti as rh

    folder = tmp_path / "radio-haiti-inter"
    folder.mkdir()
    (folder / "metadata.all.jsonl").write_text("{}\n")

    captured = {}

    class FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def create_repo(self, repo_id, **kw):
            captured["repo"] = repo_id
            captured["private"] = kw.get("private")

        def upload_large_folder(self, **kw):
            captured["folder"] = kw["folder_path"]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    rh.push_dataset(folder, "me/radio", private=True, token="hf_x")

    card = (folder / "README.md").read_text()
    assert "cc-by-4.0" in card
    assert "10.5281/zenodo.17818122" in card
    assert "10.21437/Interspeech.2025-1852" in card
    assert "Changes from the source release" in card
    assert "machine-generated" in card
    assert captured["private"] is True


def test_card_says_whether_the_clips_were_gated(tmp_path, monkeypatch):
    """An ungated dump must not read as if it had passed the quality gate."""
    from kreyol_asr import radio_haiti as rh

    class FakeApi:
        def __init__(self, token=None): pass
        def create_repo(self, *a, **k): pass
        def upload_large_folder(self, **k): pass

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    folder = tmp_path / "af"
    folder.mkdir()
    (folder / "metadata.all.jsonl").write_text("{}\n")

    rh.push_dataset(folder, "me/r", token="hf_x")
    assert "Nothing here has passed a quality gate" in (folder / "README.md").read_text()

    (folder / "metadata.jsonl").write_text("{}\n")
    rh.push_dataset(folder, "me/r", token="hf_x")
    assert "passed the two-signal gate" in (folder / "README.md").read_text()


def test_push_refuses_without_a_token_or_an_ingest(tmp_path):
    import pytest as _pytest

    from kreyol_asr import radio_haiti as rh

    empty = tmp_path / "empty"
    empty.mkdir()
    with _pytest.raises(RuntimeError, match="radio ingest"):
        rh.push_dataset(empty, "me/r", token="hf_x")

    (empty / "metadata.all.jsonl").write_text("{}\n")
    with _pytest.raises(RuntimeError, match="HF_TOKEN"):
        rh.push_dataset(empty, "me/r", token=None)
