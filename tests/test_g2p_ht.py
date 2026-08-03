"""Haitian Creole G2P.

Many cases below are exactly where espeak-ng's `ht` voice fails (measured against
1.52.0, 2026-08-02). They are kept as regressions because that failure mode is silent:
wrong phonemes still scan as fluent speech, so nothing but a Creole listener — or
this file — would catch a reintroduction.
"""

import pytest

from kreyol_tts.g2p import (PhonemeCoverage, expand_numbers, g2p,
                            number_to_creole, preprocess)

NASAL = "̃"       # COMBINING TILDE
SCRIPT_G = "ɡ"    # ɡ, not ASCII g


# --- the contrasts espeak-ng collapses --------------------------------------

@pytest.mark.parametrize("a,b", [
    ("pe", "pè"),      # calm / priest
    ("se", "sè"),      # is / sister
    ("bo", "bò"),      # kiss / side
    ("po", "pò"),      # skin / port
    ("mete", "mèt"),   # to put / master
])
def test_accented_vowels_stay_distinct(a, b):
    """espeak-ng maps è->e and ò->o, merging phonemic contrasts."""
    assert g2p(a) != g2p(b), f"{a!r} and {b!r} collapsed to {g2p(a)!r}"


def test_open_vowel_qualities():
    assert g2p("pè") == "pɛ"
    assert g2p("bò") == "bɔ"
    assert g2p("pe") == "pe"
    assert g2p("bo") == "bo"


# --- nasal vowels: absent entirely from espeak-ng ---------------------------

@pytest.mark.parametrize("word,expected", [
    ("an", "ɑ" + NASAL),
    ("tan", "tɑ" + NASAL),
    ("manje", "mɑ" + NASAL + "ʒe"),
    ("pen", "pɛ" + NASAL),
    ("plen", "plɛ" + NASAL),
    ("bon", "bɔ" + NASAL),
    ("non", "nɔ" + NASAL),
    ("tonton", "tɔ" + NASAL + "tɔ" + NASAL),
    ("gason", SCRIPT_G + "asɔ" + NASAL),
])
def test_nasal_vowels(word, expected):
    assert g2p(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("ane", "ane"),          # n opens the next syllable -> oral
    ("kann", "kan"),         # doubled n denasalizes
    ("moun", "u" + NASAL),   # `oun` is a nasal /ũ/...
    ("founi", "funi"),       # ...but not when a vowel follows the n
])
def test_denasalizing_contexts(word, expected):
    got = g2p(word)
    assert got.endswith(expected) or got == expected, f"{word!r} -> {got!r}"


def test_lang_is_nasal_plus_g_not_eng():
    """`lang` is /lɑ̃ɡ/. Treating `ng` as a /ŋ/ digraph would give /laŋ/."""
    assert g2p("lang") == "lɑ" + NASAL + SCRIPT_G


def test_ny_digraph_blocks_nasalization():
    """`benyen`: the first n belongs to `ny`, so only the final `en` nasalizes."""
    assert g2p("benyen") == "beɲɛ" + NASAL
    assert g2p("peny") == "peɲ"


# --- consonants espeak-ng gets wrong ----------------------------------------

def test_j_is_ezh_not_yod():
    """espeak-ng gives /j/, making `jou` homophonous with English "you"."""
    assert g2p("jou") == "ʒu"
    assert g2p("jodi") == "ʒodi"


def test_r_is_gamma_not_h():
    """espeak-ng maps r -> /h/."""
    assert g2p("rele") == "ɣele"
    assert g2p("rive") == "ɣive"
    assert "h" not in g2p("travay")


def test_ou_is_single_vowel():
    """espeak-ng expands `ou` into a three-segment /ouˈu/."""
    assert g2p("nou") == "nu"
    assert g2p("tout") == "tut"


def test_ch_has_no_trailing_h():
    """espeak-ng emits /ʃh/."""
    assert g2p("chita") == "ʃita"
    assert g2p("machin") == "maʃin"


def test_ui_is_labiopalatal():
    assert g2p("uit") == "ɥit"


def test_g_is_script_g_not_ascii():
    """ASCII g (U+0067) is absent from Kokoro's vocab and would be dropped silently."""
    out = g2p("gade")
    assert out == SCRIPT_G + "ade"
    assert "g" not in out, "ASCII g leaked into the output"


def test_y_and_w():
    assert g2p("yo") == "jo"
    assert g2p("ayiti") == "ajiti"
    assert g2p("wè") == "wɛ"
    assert g2p("kwè") == "kwɛ"


# --- elision ----------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("l'ap", "lap"), ("m'ap", "map"), ("k'ap", "kap"),
    ("n'ap", "nap"), ("sa'k", "sak"),
])
def test_elision_deletes_apostrophe_without_splitting(word, expected):
    assert preprocess(word) == expected
    assert "'" not in g2p(word)
    assert " " not in g2p(word), "apostrophe left a word boundary that isn't spoken"


def test_curly_apostrophes_normalize_first():
    assert preprocess("l’ap") == "lap"


# --- numbers ----------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, "zewo"), (1, "en"), (7, "sèt"), (10, "dis"), (11, "onz"), (19, "diznèf"),
    (20, "ven"), (21, "venteyen"), (22, "vennde"), (30, "trant"), (31, "tranteyen"),
    (40, "karant"), (60, "swasant"), (70, "swasanndis"), (71, "swasannonz"),
    (80, "katreven"), (81, "katreven en"), (90, "katrevendis"), (91, "katrevenonz"),
    (100, "san"), (200, "desan"), (120, "san ven"), (1000, "mil"),
    (2026, "de mil vennsis"), (999999, "nèfsan katrevendiznèf mil nèfsan katrevendiznèf"),
])
def test_number_to_creole(n, expected):
    assert number_to_creole(n) == expected


def test_number_expansion_in_text():
    assert "san ven" in expand_numbers("Li gen 120 liv")
    assert not any(c.isdigit() for c in expand_numbers("Mwen gen 25 an"))


def test_thousands_separators():
    assert expand_numbers("1,250") == expand_numbers("1250")


def test_numbers_reach_g2p_as_phonemes():
    out = g2p("Li gen 120 liv")
    assert not any(c.isdigit() for c in out)
    assert out, "digits produced no phonemes at all"


def test_number_out_of_range_is_read_digitwise():
    assert "zewo" in expand_numbers("1000000")


# --- sentences and coverage -------------------------------------------------

def test_sentence():
    # `koman` is spelled with plain `o`, so it is /o/ not /ɔ/ — IPN reserves `ò`
    # for /ɔ/, and the G2P must not "helpfully" open the vowel.
    assert g2p("Bonjou, koman ou ye?") == "bɔ" + NASAL + "ʒu, komɑ" + NASAL + " u je?"


def test_punctuation_preserved_and_droppable():
    assert g2p("Bonjou, monchè!").endswith("!")
    assert "," not in g2p("Bonjou, monchè!", keep_punct=False)


@pytest.mark.parametrize("text", [
    "Bonjou, koman ou ye?",
    "Mwen kontan wè ou jodi a.",
    "Timoun yo ap jwe nan lakou a.",
    "Kreyòl se lang manman nou.",
    "Li gen ventuit an.",
    "L'ap travay nan jaden an.",
    "Nou gen 250 goud.",
])
def test_every_phoneme_is_in_kokoro_vocab(text):
    """An out-of-vocab phoneme cannot be embedded, so it trains against nothing."""
    cov = PhonemeCoverage()
    cov.add(text)
    assert not cov.out_of_vocab, f"{text!r} -> {cov.out_of_vocab}"


def test_coverage_report_shape():
    cov = PhonemeCoverage()
    cov.add_all(["Bonjou tout moun.", "Kijan ou rele?"])
    r = cov.report()
    assert r["verdict"] == "PASS"
    assert r["texts"] == 2
    assert r["distinct_phonemes"] > 5


def test_empty_and_junk_input():
    assert g2p("") == ""
    assert g2p("   ") == ""
    assert g2p("[noise]") == ""
