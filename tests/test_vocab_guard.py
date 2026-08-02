"""The pretrained BPE cannot encode every character our transcripts contain.

Anything it can't encode must never reach the manifest — it would become an
<unk> training target and teach the RNN-T to emit <unk>.
"""

from kreyol_asr.text import VOCAB_FALLBACKS, TokenizerCoverage


class FakeCoverage(TokenizerCoverage):
    """Stands in for the real tokenizer: only ASCII letters, space and . , ? ! ' - are encodable."""

    ENCODABLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "èòé" ".,?!'- ")

    def __init__(self):  # deliberately does not call super().__init__
        self.unk_id = 0
        self.n_tokens = self.n_words = self.n_unk = 0
        from collections import Counter

        self.unk_chars = Counter()
        self.oov_chars = Counter()
        self.substituted = Counter()
        self.dropped = Counter()
        self._encodable = {}

    def encodable(self, ch):
        return ch in self.ENCODABLE


def test_colon_and_semicolon_become_commas():
    c = FakeCoverage()
    assert c.sanitize("Sa se yon pawòl ki vre: Si yon moun") == "Sa se yon pawòl ki vre, Si yon moun"
    assert c.substituted[":"] == 1


def test_parentheses_removed_without_gluing_words():
    c = FakeCoverage()
    assert c.sanitize("Bonjou (tout) moun") == "Bonjou tout moun"


def test_slash_becomes_space():
    c = FakeCoverage()
    assert c.sanitize("wi/non") == "wi non"


def test_unmapped_unencodable_char_is_dropped_and_counted():
    c = FakeCoverage()
    assert c.sanitize("Bonjou 你好 zanmi") == "Bonjou zanmi"
    assert c.dropped["你"] == 1 and c.dropped["好"] == 1


def test_encodable_punctuation_survives():
    c = FakeCoverage()
    text = "Sa k ap fèt? Mwen renmen sa. L'ap vini, wi!"
    assert c.sanitize(text) == text


def test_every_fallback_target_is_itself_encodable():
    # A fallback that maps to an unencodable character would defeat the purpose.
    c = FakeCoverage()
    for src, dst in VOCAB_FALLBACKS.items():
        assert all(ch in c.ENCODABLE or ch.isspace() for ch in dst), (src, dst)


def test_sanitized_output_has_no_unencodable_characters():
    c = FakeCoverage()
    messy = "Note: 50% (a/b) & [c] — ˆ 你 «Ayiti»"
    out = c.sanitize(messy)
    assert all(ch.isspace() or ch in c.ENCODABLE for ch in out), out
