"""Text normalizers applied before tokenization."""
from __future__ import annotations

import unicodedata
from typing import Callable, Dict

Normalizer = Callable[[str], str]


def default_normalizer(text: str) -> str:
    """NFKC normalization followed by case folding."""
    return unicodedata.normalize("NFKC", text).casefold()


def make_translation_normalizer(mapping: Dict[str, str]) -> Normalizer:
    """NFKC + character mapping + case folding."""
    table = str.maketrans(mapping)

    def normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text).translate(table).casefold()

    return normalize


# ---------------------------------------------------------------- Amharic
# Ethiopic has several letters that sound identical in modern Amharic and are
# used interchangeably in real text. Mapping each variant series onto one
# canonical series stops a single word from being counted as several types.
#   ሐ/ኀ -> ሀ   ሠ -> ሰ   ዐ -> አ   ፀ -> ጸ   ዉ -> ው  (e.g. ነዉ -> ነው)   and  ኣ/ዓ -> አ  (ኣመት/ዓመት -> አመት)
# Each series has seven orders (ሀ ሁ ሂ ሃ ሄ ህ ሆ), laid out contiguously.
_AMHARIC_SERIES = (
    (0x1210, 0x1200),  # ሐ series -> ሀ series
    (0x1280, 0x1200),  # ኀ series -> ሀ series
    (0x1220, 0x1230),  # ሠ series -> ሰ series
    (0x12D0, 0x12A0),  # ዐ series -> አ series
    (0x1340, 0x1338),  # ፀ series -> ጸ series
)

AMHARIC_CHAR_MAP: Dict[str, str] = {}
for _variant, _canon in _AMHARIC_SERIES:
    for _k in range(7):
        AMHARIC_CHAR_MAP[chr(_variant + _k)] = chr(_canon + _k)
AMHARIC_CHAR_MAP["\u12c9"] = "\u12cd"  # ዉ -> ው
AMHARIC_CHAR_MAP["\u12a3"] = "\u12a0"  # ኣ -> አ
AMHARIC_CHAR_MAP["\u12d3"] = "\u12a0"  # ዓ -> አ (set after the series loop so it overrides ዓ -> ኣ)

amharic_normalizer: Normalizer = make_translation_normalizer(AMHARIC_CHAR_MAP)
