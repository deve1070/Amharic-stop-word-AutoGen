"""Text normalizers applied before tokenization."""
from __future__ import annotations

import re
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

# Slash abbreviations, written after letter merging (so ፅ/ቤት is matched as ጽ/ቤት). The tokenizer
# splits on "/" and drops single letters, so without expansion ዶ/ር (doctor) vanishes and the word
# after it cannot be recognized as a name. A clitic prefix (የ, በ, ለ, ...) may be attached.
AMHARIC_ABBREVIATIONS: Dict[str, str] = {
    "ዶ/ር": "ዶክተር", "ፕ/ር": "ፕሮፌሰር", "ወ/ሮ": "ወይዘሮ", "ወ/ሪት": "ወይዘሪት", "ጠ/ሚ": "ጠቅላይ ሚኒስትር",
    "ጠ/ሚኒስትር": "ጠቅላይ ሚኒስትር", "አ/አ": "አዲስ አበባ", "ት/ቤት": "ትምህርት ቤት", "ም/ቤት": "ምክር ቤት",
    "ጽ/ቤት": "ጽህፈት ቤት", "መ/ቤት": "መስሪያ ቤት", "ቤ/ክ": "ቤተ ክርስቲያን",
}
_ETHIOPIC = "\u1200-\u137f"
_AMHARIC_ABBR_RX = re.compile(
    f"(?<![{_ETHIOPIC}])((?:የ|በ|ለ|ከ|ስለ|እንደ|ወደ)?)("
    + "|".join(re.escape(k) for k in sorted(AMHARIC_ABBREVIATIONS, key=len, reverse=True))
    + f")(?![{_ETHIOPIC}])"
)
_amharic_letters = make_translation_normalizer(AMHARIC_CHAR_MAP)


def amharic_normalizer(text: str) -> str:
    """Merge interchangeable letters, then expand slash abbreviations (ዶ/ር -> ዶክተር)."""
    text = _amharic_letters(text)
    if "/" not in text:
        return text
    return _AMHARIC_ABBR_RX.sub(lambda m: m.group(1) + AMHARIC_ABBREVIATIONS[m.group(2)], text)
