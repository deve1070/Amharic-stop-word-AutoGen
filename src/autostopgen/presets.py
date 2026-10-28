"""Per-language defaults (normalizer + tokenizer).

The statistics are language-agnostic; presets only supply sensible text
handling. Unknown language codes fall back to the generic preset, so any
space-delimited language works out of the box.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from .normalizers import Normalizer, amharic_normalizer, default_normalizer
from .tokenizers import WordTokenizer

Tokenizer = Callable[[str], Iterable[str]]


@dataclass(frozen=True)
class Preset:
    name: str
    normalizer: Normalizer
    tokenizer: Tokenizer


PRESETS: Dict[str, Preset] = {
    "default": Preset("generic", default_normalizer, WordTokenizer()),
    "am": Preset("Amharic", amharic_normalizer, WordTokenizer(min_len=2)),
}


def register_preset(code: str, preset: Preset) -> None:
    PRESETS[code.lower()] = preset


def get_preset(code: str) -> Preset:
    key = (code or "default").lower()
    if key in PRESETS:
        return PRESETS[key]
    warnings.warn(
        f"No preset for language {code!r}; using the generic one "
        "(Unicode NFKC + casefold + Unicode word splitting). For languages written "
        "without spaces, pass your own tokenizer.",
        stacklevel=3,
    )
    return PRESETS["default"]
