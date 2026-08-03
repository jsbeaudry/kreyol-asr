"""Haitian Creole grapheme-to-phoneme, targeting Kokoro's IPA vocabulary.

espeak-ng ships a `ht` voice and it cannot be used. Measured against 1.52.0 on
2026-08-02, it collapses `è/e` and `ò/o` (so `pè` "priest" and `pe` "calm" both come
out `/pˈe/`), maps `r` to `/h/`, renders `j` as `/j/` instead of `/ʒ/` (making `jou`
"day" homophonous with English "you"), expands `ou` into a spurious three-segment
`/ouˈu/`, and has no nasal vowels at all — `bon` becomes `/bˈon/` rather than `/bɔ̃/`.
Nasals are among the commonest vowels in the language.

That failure mode is worse than it looks: the output still scans as fluent speech, so
a model trained on it sounds confident and is systematically wrong. Nothing in a loss
curve would catch it; only a Creole speaker would.

Writing our own is cheap because IPN orthography is close to perfectly phonemic — the
whole system is the table below plus four context rules. Every phoneme it emits is
already in Kokoro's vocabulary, so no embedding surgery is needed.

    >>> g2p("Bonjou, koman ou ye?")
    'bɔ̃ʒu, kɔmɑ̃ u je?'
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from collections import Counter
from typing import Iterable

from kreyol_asr.text import normalize

# --- inventory --------------------------------------------------------------
# NOTE: /ɡ/ is U+0261 LATIN SMALL LETTER SCRIPT G, *not* ASCII "g" (U+0067).
# ASCII g is absent from Kokoro's vocab, so emitting it silently drops every /g/
# with no error anywhere. This is the single easiest way to corrupt the corpus.
G = "ɡ"
NASAL = "̃"          # COMBINING TILDE; Kokoro spells nasal vowels as V + this

VOWEL_LETTERS = set("aeiouàèéòù")

# Ordered longest-first. Context guards live in `_word_to_ipa`, not here.
DIGRAPHS = {
    "ou": "u",
    "ui": "ɥi",      # ɥi
    "ch": "ʃ",       # ʃ
    "ny": "ɲ",       # ɲ  — "peny" /pɛɲ/, "benyen" /beɲɛ̃/
}

NASAL_DIGRAPHS = {
    "an": "ɑ" + NASAL,   # ɑ̃
    "en": "ɛ" + NASAL,   # ɛ̃
    "on": "ɔ" + NASAL,   # ɔ̃
}

# Doubling the n denasalizes: `an` /ɑ̃/ but `ann` /an/.
DENASALIZED = {
    "ann": "an",
    "enn": "ɛn",     # ɛn
    "onn": "ɔn",     # ɔn
}

SINGLES = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "à": "a", "é": "e", "è": "ɛ", "ò": "ɔ", "ù": "u",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": G,
    "f": "f", "v": "v", "s": "s", "z": "z", "h": "h",
    "m": "m", "n": "n", "l": "l",
    "j": "ʒ",        # ʒ — French-style, NOT /j/. espeak-ng gets this wrong.
    "r": "ɣ",        # ɣ — velar approximant. IPN already writes `w` where [w]
                          #     is spoken ("wout", "gwo"), so no allophony rule needed.
    "w": "w",
    "y": "j",             # /j/ — "pye" /pje/, "ayiti" /ajiti/
    "c": "k",             # bare `c` is not native IPN; loanword fallback
    "q": "k", "x": "ks",
}

PUNCT_KEPT = set(".,!?;:")

# `oun` is /ũ/ ("moun" /mũ/), but only when the n closes the syllable.
_OUN = "oun"


def _is_vowel(ch: str) -> bool:
    return ch in VOWEL_LETTERS


def _word_to_ipa(word: str) -> str:
    """Convert one already-normalized, lowercased, apostrophe-free word."""
    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        nxt = word[i + 3:i + 4]
        if word[i:i + 3] == _OUN and not _is_vowel(nxt):
            out.append("u" + NASAL)
            i += 3
            continue

        tri = word[i:i + 3]
        if tri in DENASALIZED:
            out.append(DENASALIZED[tri])
            i += 3
            continue

        bi = word[i:i + 2]
        if bi in NASAL_DIGRAPHS:
            after = word[i + 2:i + 3]
            # The n stays a consonant when it opens the next syllable (`ane`), when
            # it is doubled (handled above), or when it belongs to `ny` (`benyen`).
            if not (_is_vowel(after) or after in ("n", "y")):
                out.append(NASAL_DIGRAPHS[bi])
                i += 2
                continue
            # fall through: emit the bare vowel, let the n be matched on its own
            out.append(SINGLES[bi[0]])
            i += 1
            continue

        if bi in DIGRAPHS:
            out.append(DIGRAPHS[bi])
            i += 2
            continue

        ch = word[i]
        if ch in SINGLES:
            out.append(SINGLES[ch])
        elif ch in PUNCT_KEPT or ch == " ":
            out.append(ch)
        # anything else is dropped: it is not Creole and has no phoneme
        i += 1
    return "".join(out)


# --- numbers ----------------------------------------------------------------
# `learn-the-numbers` is 1225 clips of numerals and it is the Tier B backbone for
# three of the four shipping voices, so digits reaching the G2P is not hypothetical.
# G2P on "120" produces nothing at all — silently, and only for the clips that
# matter most.

_UNITS = ["zewo", "en", "de", "twa", "kat", "senk", "sis", "sèt", "uit", "nèf",
          "dis", "onz", "douz", "trèz", "katòz", "kenz", "sèz", "disèt", "dizuit",
          "diznèf"]
_TENS = {2: "ven", 3: "trant", 4: "karant", 5: "senkant", 6: "swasant", 8: "katreven"}
# Liaison forms used before a unit: 22 "vennde", 33 "tranntwa".
_LIAISON = {2: "venn", 3: "trann", 4: "karann", 5: "senkann", 6: "swasann",
            8: "katreven"}
# The "-eyen" (X+1) forms take the underlying t-final stem, which only 20 lacks in
# its citation form: "ven" but "venteyen".
_STEM_T = {2: "vent", 3: "trant", 4: "karant", 5: "senkant", 6: "swasant"}


def _two_digit(v: int) -> str:
    if v < 20:
        return _UNITS[v]
    t, u = divmod(v, 10)
    # Haitian counts 70-79 and 90-99 vigesimally, like French: 71 is "sixty-eleven".
    if t == 7:
        return "swasann" + _UNITS[10 + u]
    if t == 9:
        return "katreven" + _UNITS[10 + u]
    base = _TENS[t]
    if u == 0:
        return base
    if u == 1:
        return "katreven en" if t == 8 else _STEM_T[t] + "eyen"
    return _LIAISON[t] + _UNITS[u]


def _three_digit(v: int) -> str:
    h, r = divmod(v, 100)
    if h == 0:
        return _two_digit(r)
    head = "san" if h == 1 else _UNITS[h] + "san"
    return head if r == 0 else f"{head} {_two_digit(r)}"


def number_to_creole(v: int) -> str:
    """Spell an integer in Haitian Creole. Supports -999,999 .. 999,999."""
    if v < 0:
        return "mwens " + number_to_creole(-v)
    if v < 1000:
        return _three_digit(v)
    if v >= 1_000_000:
        raise ValueError(f"{v}: only magnitudes below 1,000,000 are supported")
    th, r = divmod(v, 1000)
    head = "mil" if th == 1 else f"{_three_digit(th)} mil"
    return head if r == 0 else f"{head} {_three_digit(r)}"


_NUMBER = re.compile(r"\d[\d\s.,]*\d|\d")
_CURRENCY = {"$": "dola", "€": "ero", "HTG": "goud", "USD": "dola"}


def expand_numbers(text: str) -> str:
    """Replace digit runs with their Creole spelling, thousands separators and all."""
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        # "1,250" and "1 250" are separators; "1.5" is a decimal.
        if "." in raw and not re.fullmatch(r"[\d.]+", raw.replace(",", "")):
            pass
        cleaned = raw.replace(",", "").replace(" ", "")
        if "." in cleaned:
            whole, _, frac = cleaned.partition(".")
            parts = [number_to_creole(int(whole))] if whole else []
            parts.append("vigil")
            parts += [_UNITS[int(d)] for d in frac if d.isdigit()]
            return " ".join(parts)
        try:
            return number_to_creole(int(cleaned))
        except ValueError:
            # Too large to spell — read it digit by digit rather than dropping it.
            return " ".join(_UNITS[int(d)] for d in cleaned)

    for sym, word in _CURRENCY.items():
        text = text.replace(sym, f" {word} ")
    return _NUMBER.sub(repl, text)


# --- top level --------------------------------------------------------------

_APOSTROPHE_ELISION = re.compile(r"(?<=[a-zàèéòù])'(?=[a-zàèéòù])")


def preprocess(text: str) -> str:
    """Normalize, lowercase, expand numbers, resolve elision. Runs before G2P.

    Lowercasing is deliberate and is the opposite of the ASR config, where
    `lowercase: false` matches the base model's cased output. Case carries no
    phonetic information and would double the rule surface for nothing.

    Elision is deleted, not tokenized: the apostrophe in `l'ap` marks an elided `li`,
    it is not a segment, and it is absent from Kokoro's vocab. Deleting it *between
    letters* keeps `lap` contiguous — splitting on non-letters first would leave a
    word boundary that does not exist in the speech.
    """
    text = normalize(text, lowercase=True, strip_bracketed=True,
                     normalize_apostrophes=True)
    text = expand_numbers(text)
    text = _APOSTROPHE_ELISION.sub("", text)
    return unicodedata.normalize("NFC", text)


def g2p(text: str, *, keep_punct: bool = True) -> str:
    """Haitian Creole text -> IPA string in Kokoro's phoneme inventory."""
    text = preprocess(text)
    out: list[str] = []
    for token in re.findall(r"[a-zàèéòù]+|[.,!?;:]|\s+", text):
        if token.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        elif token in PUNCT_KEPT:
            if keep_punct:
                out.append(token)
        else:
            out.append(_word_to_ipa(token))
    return "".join(out).strip()


# --- coverage ---------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def kokoro_vocab() -> dict[str, int]:
    from huggingface_hub import hf_hub_download

    from . import KOKORO_REPO

    return json.load(open(hf_hub_download(KOKORO_REPO, "config.json")))["vocab"]


class PhonemeCoverage:
    """Check that every phoneme the corpus produces exists in Kokoro's vocabulary.

    The analogue of `kreyol_asr.text.TokenizerCoverage`, which is BPE-specific and
    not reusable. An out-of-vocab phoneme is a hard stop, not a warning: it cannot be
    embedded, so those clips would train against a silently dropped segment.
    """

    def __init__(self) -> None:
        self.vocab = kokoro_vocab()
        self.counts: Counter[str] = Counter()
        self.texts = 0

    def add(self, text: str) -> str:
        ipa = g2p(text)
        self.counts.update(c for c in ipa if not c.isspace())
        self.texts += 1
        return ipa

    def add_all(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    @property
    def out_of_vocab(self) -> dict[str, int]:
        return {c: n for c, n in self.counts.items() if c not in self.vocab}

    def report(self) -> dict[str, object]:
        oov = self.out_of_vocab
        # A phoneme the model sees only a handful of times is present but untrained;
        # it will be produced badly rather than not at all, which is harder to spot.
        # Punctuation is excluded — it is in the vocab but it is not a phoneme, and
        # counting it here buries real findings under "! appeared 32 times".
        rare = {c: n for c, n in self.counts.items()
                if c not in oov and c not in PUNCT_KEPT and n < 50}
        return {
            "texts": self.texts,
            "distinct_phonemes": len(self.counts),
            "vocab_size": len(self.vocab),
            "out_of_vocab": {
                c: {"count": n, "name": unicodedata.name(c, "?"),
                    "codepoint": f"U+{ord(c):04X}"}
                for c, n in sorted(oov.items(), key=lambda kv: -kv[1])
            },
            "rare_phonemes": dict(sorted(rare.items(), key=lambda kv: kv[1])),
            "most_common": self.counts.most_common(20),
            "verdict": "PASS" if not oov else f"FAIL: {len(oov)} out-of-vocab phoneme(s)",
        }
