"""Haitian Creole transcript normalization + tokenizer coverage diagnostics.

The base model emits punctuated, properly-cased text. NVIDIA's fine-tuning guidance
is explicit that transcript style should match the base model's, so the default here
is conservative: fix encoding-level noise, leave casing and punctuation alone.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Haitian Creole (IPN orthography) letters, plus the accented vowels it uses.
# `é` and `à` are not standard IPN but appear in real-world text and French loanwords.
CREOLE_LETTERS = set("abcdefghijklmnopqrstuvwxyzàèéòù")
CREOLE_LETTERS |= {c.upper() for c in CREOLE_LETTERS}
ALLOWED_PUNCT = set(" .,?!'-:;\"()%&/")
DIGITS = set("0123456789")
CREOLE_CHARSET = CREOLE_LETTERS | ALLOWED_PUNCT | DIGITS

# Curly quotes, primes and lookalikes -> ASCII apostrophe. Critical for Creole
# elisions (l'ap, m'ap, k'ap, n'ap): a mismatched apostrophe splits the BPE token
# and inflates WER for the most common words in the language.
_APOSTROPHES = dict.fromkeys(map(ord, "‘’ʼʹ′´`"), "'")
_QUOTES = dict.fromkeys(map(ord, "“”„«»"), '"')
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")

_BRACKETED = re.compile(r"[\[\(<](?:noise|inaudible|unk|silence|music|laugh"
                        r"|applause|foreign|uh+|um+)[^\]\)>]*[\]\)>]", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")


def normalize(
    text: str,
    *,
    lowercase: bool = False,
    strip_bracketed: bool = True,
    normalize_apostrophes: bool = True,
) -> str:
    """Clean a single transcript. Returns "" for transcripts that are all noise."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    if strip_bracketed:
        text = _BRACKETED.sub(" ", text)
    if normalize_apostrophes:
        text = text.translate(_APOSTROPHES)
    text = text.translate(_QUOTES).translate(_DASHES)
    text = text.replace(" ", " ").replace("​", "")
    text = _WS.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    if lowercase:
        text = text.lower()
    return text


def out_of_charset(text: str) -> set[str]:
    """Characters not expected in Haitian Creole — worth eyeballing before a long run."""
    return {c for c in text if c not in CREOLE_CHARSET}


def style_stats(texts: list[str]) -> dict[str, float]:
    """Detect a transcript-style mismatch with the base model (cased + punctuated).

    A corpus that is 100% lowercase and unpunctuated will fight the base model's
    prior; the caller surfaces this so the user can decide before training.
    """
    if not texts:
        return {"cased_ratio": 0.0, "punctuated_ratio": 0.0}
    cased = sum(1 for t in texts if any(c.isupper() for c in t))
    punct = sum(1 for t in texts if any(c in ".!?," for c in t))
    return {
        "cased_ratio": cased / len(texts),
        "punctuated_ratio": punct / len(texts),
    }


# The Nemotron BPE covers . , ? ! ' - " but NOT : ; ( ) % & / — verified by
# encoding each character against tokenizer.json. Left alone, those characters
# become <unk> *training targets*, which teaches the RNN-T to emit <unk>.
# Map them to the nearest character the vocab can actually produce.
VOCAB_FALLBACKS = {
    ":": ",", ";": ",", "…": "...",
    "(": "", ")": "", "[": "", "]": "", "{": "", "}": "",
    "/": " ", "\\": " ", "|": " ", "_": " ",
    "%": "", "&": "", "*": "", "+": "", "=": "", "@": "", "#": "", "~": "", "^": "",
}


class TokenizerCoverage:
    """Runs the *pretrained* BPE over the corpus to decide whether reusing it is safe.

    Retraining the tokenizer would reset the RNN-T joint, so we want evidence that
    the existing 13088-token vocab covers Creole before we commit to reusing it.
    """

    def __init__(self, repo_id: str, token: str | None = None):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(repo_id, "tokenizer.json", token=token)
        self.tok = Tokenizer.from_file(path)
        self.unk_id = None
        vocab = self.tok.get_vocab()
        for cand in ("<unk>", "[UNK]", "<UNK>"):
            if cand in vocab:
                self.unk_id = vocab[cand]
                break
        self.n_tokens = 0
        self.n_words = 0
        self.n_unk = 0
        self.unk_chars: Counter[str] = Counter()
        self.oov_chars: Counter[str] = Counter()
        self.substituted: Counter[str] = Counter()
        self.dropped: Counter[str] = Counter()
        self._encodable: dict[str, bool] = {}

    def encodable(self, ch: str) -> bool:
        """Can the pretrained vocab represent this character at all?"""
        if ch not in self._encodable:
            ids = self.tok.encode(f"a{ch}a").ids
            self._encodable[ch] = self.unk_id is None or self.unk_id not in ids
        return self._encodable[ch]

    def sanitize(self, text: str, fallbacks: dict[str, str] | None = None) -> str:
        """Rewrite text so every character survives the pretrained tokenizer.

        Anything the vocab cannot encode is substituted via `fallbacks`, or dropped.
        Both outcomes are counted and surfaced in the data report.
        """
        fb = VOCAB_FALLBACKS if fallbacks is None else fallbacks
        out: list[str] = []
        for ch in text:
            if ch.isspace() or self.encodable(ch):
                out.append(ch)
                continue
            if ch in fb:
                self.substituted[ch] += 1
                out.append(fb[ch])
            else:
                self.dropped[ch] += 1
        return _WS.sub(" ", "".join(out)).strip()

    def add(self, text: str) -> None:
        enc = self.tok.encode(text)
        self.n_tokens += len(enc.ids)
        self.n_words += len(text.split())
        for c in out_of_charset(text):
            self.oov_chars[c] += 1
        if self.unk_id is not None:
            for i, tid in enumerate(enc.ids):
                if tid != self.unk_id:
                    continue
                self.n_unk += 1
                # Report the *source* substring, not the literal "<unk>" — the
                # whole point is to see which characters the vocab can't encode.
                start, end = enc.offsets[i]
                piece = text[start:end] if end > start else ""
                self.unk_chars[piece or "<empty-span>"] += 1

    def report(self) -> dict:
        return {
            "tokens": self.n_tokens,
            "words": self.n_words,
            "tokens_per_word": round(self.n_tokens / self.n_words, 3) if self.n_words else 0.0,
            "unk_tokens": self.n_unk,
            "unk_rate": round(self.n_unk / self.n_tokens, 6) if self.n_tokens else 0.0,
            "unk_pieces": dict(self.unk_chars.most_common(20)),
            "out_of_charset_chars": dict(self.oov_chars.most_common(30)),
            "substituted_chars": dict(self.substituted.most_common(30)),
            "dropped_chars": dict(self.dropped.most_common(30)),
        }

    @staticmethod
    def verdict(rep: dict) -> str:
        """Plain-language go/no-go on reusing the pretrained tokenizer."""
        if rep["unk_rate"] > 0.001:
            return ("REVIEW: >0.1% of tokens are <unk>. Inspect `unk_pieces` — usually a "
                    "stray script or encoding artifact worth cleaning, not a reason to "
                    "retrain the tokenizer.")
        if rep["tokens_per_word"] > 4.0:
            return ("REVIEW: >4 tokens/word means the vocab fragments Creole badly. "
                    "Reuse still beats resetting the RNN-T joint, but expect slower "
                    "convergence.")
        return "OK: reuse the pretrained tokenizer."
