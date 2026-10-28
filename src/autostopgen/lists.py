"""Bundled stopword lists and helpers for applying a list to text."""
from __future__ import annotations

from importlib import resources
from typing import Callable, Collection, Iterable, List, Optional, Union

from .normalizers import Normalizer, default_normalizer
from .presets import get_preset
from .tokenizers import default_tokenizer


def _data_dir():
    return resources.files("autostopgen").joinpath("data")


def available_languages() -> List[str]:
    return sorted(p.name[:-4] for p in _data_dir().iterdir() if p.name.endswith(".txt") and p.name != "README.txt")


def read_stopword_file(path) -> List[str]:
    with open(path, "rt", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def load_stopwords(language: str) -> List[str]:
    """Return the bundled stopword list for ``language`` (ISO 639-1 code, e.g. ``"am"``)."""
    f = _data_dir().joinpath(f"{language.lower()}.txt")
    if not f.is_file():
        raise LookupError(
            f"No bundled stopword list for {language!r}. Available: {available_languages() or 'none'}. "
            "Generate one with autostopgen.StopwordGenerator or the `autostopgen generate` command."
        )
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def remove_stopwords(
    text_or_tokens: Union[str, Iterable[str]],
    stopwords: Collection[str],
    *,
    language: Optional[str] = None,
    tokenizer: Optional[Callable[[str], Iterable[str]]] = None,
    normalizer: Optional[Normalizer] = None,
) -> List[str]:
    """Drop stopwords. A string is normalized and tokenized first; a token list is filtered as is.

    Matching uses normalized forms on both sides, so Amharic ነው and ነዉ are treated alike with
    ``language="am"``.
    """
    if language:
        preset = get_preset(language)
        tokenizer = tokenizer or preset.tokenizer
        normalizer = normalizer or preset.normalizer
    tokenizer = tokenizer or default_tokenizer
    normalizer = normalizer or default_normalizer
    stop = {normalizer(w) for w in stopwords}
    tokens = tokenizer(normalizer(text_or_tokens)) if isinstance(text_or_tokens, str) else text_or_tokens
    return [t for t in tokens if normalizer(t) not in stop]
