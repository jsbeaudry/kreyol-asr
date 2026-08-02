import json

from kreyol_asr.evaluate import read_predictions, render_report, score, worst_clips


def test_exact_vs_normalized_scoring():
    pairs = [("Mwen renmen sa.", "mwen renmen sa")]
    assert score(pairs)["wer"] > 0  # punctuation + casing count
    assert score(pairs, aggressive=True)["wer"] == 0.0


def test_score_skips_empty_references():
    r = score([("", "anything"), ("Bonjou tout moun", "Bonjou tout moun")])
    assert r["clips"] == 1 and r["wer"] == 0.0


def test_read_predictions_accepts_jsonl_and_array(tmp_path):
    rows = [{"audio_filepath": "a.wav", "text": "Bonjou", "pred_text": "Bonjou"}]
    jsonl = tmp_path / "a.json"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    arr = tmp_path / "b.json"
    arr.write_text(json.dumps(rows))
    assert read_predictions(jsonl) == read_predictions(arr) == rows


def test_read_predictions_maps_alternate_prediction_key(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"audio_filepath": "a.wav", "text": "Bonjou",
                              "transcript": "Bonjou tout"}]))
    assert read_predictions(p)[0]["pred_text"] == "Bonjou tout"


def test_worst_clips_sorted_descending():
    rows = [
        {"audio_filepath": "good.wav", "text": "Bonjou tout moun", "pred_text": "Bonjou tout moun"},
        {"audio_filepath": "bad.wav", "text": "Bonjou tout moun", "pred_text": "zzz"},
    ]
    worst = worst_clips(rows, n=2, aggressive=False)
    assert worst[0]["audio_filepath"] == "bad.wav"


def test_report_shows_delta_between_baseline_and_finetune():
    report = {
        "scoring": "exact",
        "test_clips": 10,
        "results": [
            {"model": "baseline", "att_context_size": [56, 3], "latency_ms": 320,
             "target_lang": "fr-FR", "wer": 0.80, "cer": 0.50, "clips": 10},
            {"model": "finetuned", "att_context_size": [56, 3], "latency_ms": 320,
             "target_lang": "ht-HT", "wer": 0.25, "cer": 0.10, "clips": 10},
        ],
    }
    md = render_report(report)
    assert "-0.5500" in md and "0.8000" in md and "320 ms" in md


def test_report_handles_baseline_only():
    report = {
        "scoring": "exact", "test_clips": 5,
        "results": [{"model": "baseline", "att_context_size": [56, 0], "latency_ms": 80,
                     "target_lang": "fr-FR", "wer": 0.9, "cer": 0.6, "clips": 5}],
    }
    md = render_report(report)
    assert "0.9000" in md and "—" in md


def test_locate_predictions_finds_nemo_chosen_filename(tmp_path):
    """NeMo picks the output filename itself, inside the directory we hand it.

    Passing a file path made it mkdir that path and write inside, so reading it
    back raised IsADirectoryError after a *successful* inference run.
    """
    from kreyol_asr.evaluate import locate_predictions

    out_root = tmp_path / "baseline_56-3"
    out_root.mkdir()
    produced = out_root / "streaming_out_nemotron-3.5-asr-streaming-0_test.json"
    produced.write_text('{"audio_filepath": "a.wav", "text": "bonjou", "pred_text": "bonjou"}\n')

    assert locate_predictions(out_root) == produced


def test_locate_predictions_accepts_a_plain_file(tmp_path):
    from kreyol_asr.evaluate import locate_predictions

    f = tmp_path / "preds.json"
    f.write_text("{}")
    assert locate_predictions(f) == f


def test_locate_predictions_errors_when_nothing_written(tmp_path):
    import pytest as _pytest

    from kreyol_asr.evaluate import locate_predictions

    empty = tmp_path / "empty"
    empty.mkdir()
    with _pytest.raises(RuntimeError, match="wrote no JSON"):
        locate_predictions(empty)


def test_language_tag_is_stripped_before_scoring():
    """NeMo emits the prompt tag inline: "... contribution. <fr-FR>".

    It is never in the reference, so counting it costs one substitution on every
    clip that carries it and quietly inflates the reported WER.
    """
    from kreyol_asr.evaluate import score

    clean = score([("bonjou zanmi", "bonjou zanmi")])
    tagged = score([("bonjou zanmi", "bonjou zanmi <fr-FR>")])
    assert tagged["wer"] == clean["wer"] == 0.0

    assert score([("bonjou zanmi", "bonjou zanmi <ht-HT>")])["wer"] == 0.0


def test_worst_clips_survives_missing_audio_filepath():
    """NeMo streaming output carries only {pred_text, text, wer}."""
    from kreyol_asr.evaluate import worst_clips

    rows = [{"text": "bonjou zanmi mwen", "pred_text": "bonjour ami"},
            {"text": "mwen renmen ou", "pred_text": "mwen renmen ou"}]
    out = worst_clips(rows, n=5, aggressive=False)
    assert len(out) == 2
    assert out[0]["wer"] > out[1]["wer"]
    assert out[0]["audio_filepath"].startswith("<row ")


def test_worst_clips_uses_path_when_present():
    from kreyol_asr.evaluate import worst_clips

    rows = [{"audio_filepath": "/a/b.wav", "text": "bonjou", "pred_text": "bonjour"}]
    assert worst_clips(rows, 5, False)[0]["audio_filepath"] == "/a/b.wav"


def test_tags_are_counted_even_though_stripped_from_scoring():
    """Stripping tags fixes WER but destroys the evidence — so count them.

    A run prompted with ht-HT that comes back emitting <fr-FR> means the prompt
    conditioning never engaged. Without this, that failure is silently normalized
    away and the WER looks respectable.
    """
    from kreyol_asr.evaluate import score, tags_emitted

    assert tags_emitted(["bonjou <fr-FR>", "sa k ap fet <fr-FR>", "no tag"]) == {"<fr-FR>": 2}
    assert tags_emitted([]) == {}
    assert tags_emitted([None, ""]) == {}

    m = score([("bonjou", "bonjou <fr-FR>")])
    assert m["wer"] == 0.0, "tag must not count against WER"
    assert m["tags_emitted"] == {"<fr-FR>": 1}, "but must still be reported"


def test_mixed_tags_are_all_counted():
    from kreyol_asr.evaluate import tags_emitted

    assert tags_emitted(["a <ht-HT>", "b <fr-FR>", "c <ht-HT>"]) == {"<ht-HT>": 2, "<fr-FR>": 1}
