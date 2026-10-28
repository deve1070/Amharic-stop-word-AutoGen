"""Tokenizers that work for any space-delimited script.

The default :class:`WordTokenizer` splits on every Unicode punctuation,
symbol, separator and control character, so it handles Latin, Cyrillic, Arabic,
Ethiopic (including the Ethiopic word space and full stop), Devanagari (combining
marks stay inside words) and so on. Languages written without spaces (Chinese,
Japanese, Thai, Khmer, Burmese...) need a real segmenter: pass any
``Callable[[str], Iterable[str]]`` as ``tokenizer``, or use
:class:`CharNgramTokenizer` as a crude fallback.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache
from typing import List

# Apostrophes are kept inside words (don't) but stripped from token edges.
_INNER_KEEP = "'\u2019\u02bc"


def _is_separator(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return (cat[0] in "PSZ" or cat == "Cc") and ch not in _INNER_KEEP


@lru_cache(maxsize=1)
def _word_regex() -> "re.Pattern[str]":
    ranges = []
    start = None
    for cp in range(sys.maxunicode + 1):
        sep = _is_separator(chr(cp))
        if sep and start is None:
            start = cp
        elif not sep and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, sys.maxunicode))
    cls = "".join(
        f"\\U{a:08x}" if a == b else f"\\U{a:08x}-\\U{b:08x}" for a, b in ranges
    )
    return re.compile(f"[^{cls}]+")


class WordTokenizer:
    """Split text into word tokens on Unicode punctuation, symbols and spaces."""

    def __init__(self, drop_numeric: bool = True, min_len: int = 1):
        self.drop_numeric = drop_numeric
        self.min_len = min_len
        self._rx = _word_regex()

    def __call__(self, text: str) -> List[str]:
        out = []
        for tok in self._rx.findall(text):
            tok = tok.strip(_INNER_KEEP)
            if len(tok) < self.min_len:
                continue
            if self.drop_numeric and tok.isnumeric():
                continue
            out.append(tok)
        return out


class CharNgramTokenizer:
    """Fallback for unsegmented scripts: character n-grams inside each word run.

    Crude, but lets the statistics run on Chinese/Japanese/Thai text without an
    external segmenter. Prefer a proper word segmenter when you have one.
    """

    def __init__(self, n: int = 1, drop_numeric: bool = True):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._base = WordTokenizer(drop_numeric=drop_numeric)

    def __call__(self, text: str) -> List[str]:
        out: List[str] = []
        n = self.n
        for run in self._base(text):
            if len(run) <= n:
                out.append(run)
            else:
                out.extend(run[i : i + n] for i in range(len(run) - n + 1))
        return out


default_tokenizer = WordTokenizer()
