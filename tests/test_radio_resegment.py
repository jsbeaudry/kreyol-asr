"""Re-segmenting Radio Haiti into the window `prepare` will accept.

Two failure modes are worth a test suite of their own. A mid-word cut pairs half a
word of audio with a whole word of text on *both* sides of the seam, quietly teaching
the model a wrong alignment. A merge across a long pause fabricates an utterance that
was never spoken as one. Neither raises; both would just make the corpus worse.
"""

import pytest

from kreyol_asr.radio_haiti import (MAX_S, Segment, Word, merge_short, resegment,
                                    split_long)


def seg(start_ms, end_ms, *, speaker="SPEAKER_00", recording="rec-a", text="",
        conf=0.95, words=()):
    return Segment(recording, "train", speaker, start_ms, end_ms,
                   text or " ".join(w.text for w in words), conf, words=tuple(words))


def evenly_spaced(n, *, start=0, word_ms=1000, gap_ms=0):
    """n words back to back, each `word_ms` long."""
    out, t = [], start
    for i in range(n):
        out.append(Word(f"w{i}", t, t + word_ms))
        t += word_ms + gap_ms
    return out


# --- split -----------------------------------------------------------------

def test_short_segments_pass_through_untouched():
    s = seg(0, 5000, text="bonjou")
    assert split_long(s) == [s]


def test_split_respects_max_and_min():
    words = evenly_spaced(40)  # 40 s
    out = split_long(seg(0, 40_000, words=words), max_s=15.0, min_s=1.0)
    assert out, "a fully aligned segment must be splittable"
    for part in out:
        assert part.duration <= 15.0
        assert part.duration >= 1.0


def test_cuts_land_only_on_word_boundaries():
    words = evenly_spaced(40)
    ends = {w.end_ms for w in words}
    starts = {w.start_ms for w in words}
    out = split_long(seg(0, 40_000, words=words), max_s=15.0)
    for part in out[:-1]:
        # The boundary is the midpoint of an inter-word gap; with zero-width gaps
        # that is exactly a word end, which is also the next word's start.
        assert part.end_ms in ends or part.end_ms in starts


def test_split_preserves_the_transcript():
    words = evenly_spaced(40)
    original = " ".join(w.text for w in words)
    out = split_long(seg(0, 40_000, words=words), max_s=15.0)
    assert " ".join(p.text for p in out) == original, "no word may be lost at a seam"


def test_split_prefers_the_widest_pause():
    """A real pause is a better seam than an arbitrary one, even off-centre."""
    words = [Word("a", 0, 4000), Word("b", 4000, 8000),
             Word("c", 11_000, 15_000), Word("d", 15_000, 19_000)]
    out = split_long(seg(0, 19_000, words=words), max_s=10.0, min_s=1.0)
    assert len(out) == 2
    assert out[0].text == "a b" and out[1].text == "c d"


def test_unsplittable_segment_is_dropped_not_guessed():
    """No alignment means no safe cut. Losing the segment beats a mid-word seam."""
    assert split_long(seg(0, 40_000, text="yon fraz long anpil")) == []


def test_split_refuses_when_words_describe_other_text():
    mismatched = seg(0, 40_000, words=evenly_spaced(40))
    mismatched = Segment(mismatched.recording_id, "train", mismatched.speaker, 0, 40_000,
                         "tèks ki pa koresponn", 0.95, words=mismatched.words)
    assert split_long(mismatched) == []


# --- merge -----------------------------------------------------------------

def test_merge_joins_short_neighbours():
    segs = [seg(0, 2000, text="bonjou"), seg(2100, 4000, text="tout moun")]
    (merged,) = merge_short(segs, target_s=12.0, max_gap_ms=400)
    assert merged.text == "bonjou tout moun"
    assert (merged.start_ms, merged.end_ms) == (0, 4000)
    assert merged.origin == "merged"


def test_merge_stops_at_a_speaker_change():
    segs = [seg(0, 2000, text="a"), seg(2100, 4000, text="b", speaker="SPEAKER_01")]
    assert len(merge_short(segs)) == 2


def test_merge_stops_at_a_long_pause():
    """Gluing across a pause fabricates an utterance nobody spoke as one."""
    segs = [seg(0, 2000, text="a"), seg(9000, 11_000, text="b")]
    assert len(merge_short(segs, max_gap_ms=400)) == 2


def test_merge_never_exceeds_target():
    segs = [seg(i * 3000, i * 3000 + 2900, text=f"w{i}") for i in range(10)]
    for m in merge_short(segs, target_s=12.0, max_gap_ms=400):
        assert m.duration <= 12.0


def test_merged_confidence_is_duration_weighted():
    """A long confident span must not be dragged down by a short doubtful one."""
    segs = [seg(0, 9000, text="a", conf=1.0), seg(9000, 10_000, text="b", conf=0.0)]
    (merged,) = merge_short(segs, target_s=12.0, max_gap_ms=400)
    assert merged.confidence == pytest.approx(0.9, abs=1e-6)


def test_merge_does_not_cross_recordings():
    segs = [seg(0, 2000, text="a"), seg(2100, 4000, text="b", recording="rec-b")]
    assert len(merge_short(segs)) == 2


# --- resegment -------------------------------------------------------------

def test_resegment_reports_what_it_lost():
    segs = [seg(0, 40_000, text="pa gen aliyman"), seg(40_000, 45_000, text="bon")]
    kept, drops = resegment(segs, merge=False)
    assert [k.text for k in kept] == ["bon"]
    assert drops["too_long_no_word_alignment"] == 1
    assert drops["_hours_lost"] == pytest.approx(40 / 3600)


def test_resegment_output_always_fits_the_prepare_window():
    segs = [seg(0, 40_000, words=evenly_spaced(40)),
            seg(40_000, 40_300, text="ti"),
            seg(40_400, 52_000, words=evenly_spaced(12, start=40_400))]
    kept, _ = resegment(segs, max_s=MAX_S, min_s=1.0, target_s=12.0)
    for k in kept:
        assert 1.0 <= k.duration <= MAX_S, f"{k.duration}s escapes the window"


def test_resegment_conserves_audio_time():
    """Time may be dropped deliberately, never duplicated."""
    words = evenly_spaced(30)
    segs = [seg(0, 30_000, words=words)]
    kept, _ = resegment(segs, max_s=15.0, merge=False)
    assert sum(k.duration for k in kept) == pytest.approx(30.0, abs=0.01)


def test_empty_text_is_counted_not_kept():
    kept, drops = resegment([seg(0, 3000, text="")], merge=False)
    assert kept == []
    assert drops["empty_text"] == 1
