from kreyol_asr.text import CREOLE_CHARSET, normalize, out_of_charset, style_stats


def test_curly_apostrophe_becomes_ascii():
    # Creole elisions are the highest-frequency tokens in the language; a curly
    # apostrophe fragments them in the BPE and inflates WER.
    assert normalize("L’ap vini") == "L'ap vini"
    assert normalize("Mʼap ale, k’ap fè") == "M'ap ale, k'ap fè"


def test_nfc_normalization():
    decomposed = "fèt"  # e + combining grave
    assert normalize(decomposed) == "fèt"


def test_bracketed_annotations_stripped():
    assert normalize("Bonjou [noise] tout moun") == "Bonjou tout moun"
    assert normalize("(inaudible) sa k ap fèt?") == "sa k ap fèt?"


def test_bracketed_left_alone_when_disabled():
    assert normalize("Bonjou [noise]", strip_bracketed=False) == "Bonjou [noise]"


def test_whitespace_and_space_before_punct():
    assert normalize("  Mwen   renmen  sa .") == "Mwen renmen sa."


def test_casing_preserved_by_default():
    assert normalize("Ayiti Cheri") == "Ayiti Cheri"
    assert normalize("Ayiti Cheri", lowercase=True) == "ayiti cheri"


def test_quotes_and_dashes_normalized():
    assert normalize("“Ayiti” — peyi m") == '"Ayiti" - peyi m'


def test_creole_charset_accepts_accented_vowels():
    assert not out_of_charset("Mwen renmen w anpil, frè m! Sa k ap fèt? Bò lakay.")


def test_out_of_charset_flags_foreign_script():
    assert out_of_charset("Bonjou 你好") == {"你", "好"}


def test_charset_contains_ipn_accents():
    for c in "èòé":
        assert c in CREOLE_CHARSET


def test_style_stats_detects_lowercase_unpunctuated_corpus():
    s = style_stats(["mwen renmen sa", "sa k ap fet"])
    assert s["cased_ratio"] == 0.0
    assert s["punctuated_ratio"] == 0.0

    s2 = style_stats(["Mwen renmen sa.", "Sa k ap fèt?"])
    assert s2["cased_ratio"] == 1.0
    assert s2["punctuated_ratio"] == 1.0


def test_empty_and_noise_only_transcripts():
    assert normalize("") == ""
    assert normalize("[noise]") == ""


# --- implausible transcript filter -----------------------------------------

def test_chars_per_second_filter_default_is_wired():
    """A 644 chars/sec transcript OOM-killed a training run and teaches loops."""
    from kreyol_asr.config import DataConfig

    import tempfile, pathlib, yaml
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c.yaml"
        p.write_text(yaml.safe_dump({"sources": [{"repo_id": "a/b"}]}))
        cfg = DataConfig.load(p)
    assert cfg.filters["max_chars_per_second"] == 25.0


def test_chars_per_second_filter_is_overridable():
    from kreyol_asr.config import DataConfig

    import tempfile, pathlib, yaml
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c.yaml"
        p.write_text(yaml.safe_dump({"sources": [{"repo_id": "a/b"}],
                                     "filters": {"max_chars_per_second": 40}}))
        cfg = DataConfig.load(p)
    assert cfg.filters["max_chars_per_second"] == 40


def test_repetition_loop_is_implausibly_fast():
    """Sanity-check the threshold against the real offender and real speech."""
    loop = "Seyè, bondye, " + "pou " * 430          # the actual 1761-char clip
    assert len(loop) / 2.7 > 25, "must be caught"

    normal = "Mwen wè nou kanpe isit kòm lòt bò dlo, pou mwen kanpe avè nou isite."
    assert len(normal) / 5.0 < 25, "normal speech must survive"
