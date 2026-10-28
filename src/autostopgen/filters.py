"""Remove nouns and names from a ranked candidate list.

Corpus statistics cannot separate an evenly spread noun (Amharic ሰው "person", ጉዳይ "matter") from
a function word, so ranked candidates pass through a filter before the final list is cut:

* an **exclude** list loaded from word-list files (each entry remembers the file it came from);
* a **residual IDF** threshold: a word whose observed IDF exceeds the IDF expected under random
  placement by more than ``ridf_max`` in any corpus is clustered by topic;
* a **title rule** (:class:`TitleContext`): words that often follow a person title (አቶ X, ዶክተር X Y)
  or precede a place word (X ከተማ, X ክልል) are names. The title and place words are excluded too;
* an **include** list (whitelist) that overrides every exclusion. It protects words; it does not
  add words the statistics did not rank.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .core import StopwordGenerator
from .io import PathLike
from .lists import read_stopword_file
from .normalizers import Normalizer, default_normalizer

_CONTEXT_FORMAT = "autostopgen-title-context-1"


def load_lexicon(paths: Iterable[PathLike], normalizer: Optional[Normalizer] = None) -> Dict[str, str]:
    """Read word-list files (one word per line, ``#`` comments) into ``{normalized word: file name}``."""
    norm = normalizer or default_normalizer
    out: Dict[str, str] = {}
    for path in paths:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        for w in read_stopword_file(path):
            out.setdefault(norm(w), name)
    return out


class TitleContext:
    """Count the tokens next to person titles and place words while a corpus is counted.

    Pass an instance as an observer: ``StopwordGenerator(..., observers=[ctx])``. For each person
    title it counts the next two tokens (given name, then father's name); for each place word it
    counts the previous token. ``prefixes`` are clitics written attached to the title (የአቶ, በአቶ).
    """

    def __init__(self, person_titles: Iterable[str], place_words: Iterable[str],
                 prefixes: Sequence[str] = ()):
        self.person_titles = set(person_titles)
        self.place_words = set(place_words)
        self.prefixes = tuple(prefixes)
        self._titles = {p + t for t in self.person_titles for p in ("",) + self.prefixes}
        self._places = {p + t for t in self.place_words for p in ("",) + self.prefixes}
        self.hits: Counter = Counter()

    def __call__(self, tokens: List[str]) -> None:
        titles, places, hits = self._titles, self._places, self.hits
        n = len(tokens)
        for i, t in enumerate(tokens):
            if t in titles:
                if i + 1 < n and tokens[i + 1] not in titles:
                    hits[tokens[i + 1]] += 1
                    if i + 2 < n and tokens[i + 2] not in titles:
                        hits[tokens[i + 2]] += 1
            elif t in places and i > 0 and tokens[i - 1] not in places:
                hits[tokens[i - 1]] += 1

    def merge(self, other: "TitleContext") -> "TitleContext":
        """Add another corpus's counts (the trigger lists must match)."""
        if (other.person_titles, other.place_words, other.prefixes) != (
                self.person_titles, self.place_words, self.prefixes):
            raise ValueError("cannot merge title contexts built from different trigger lists")
        self.hits.update(other.hits)
        return self

    @property
    def trigger_words(self) -> set:
        return self._titles | self._places

    def names(self, gen: StopwordGenerator, min_hits: int = 20, min_ratio: float = 0.2) -> Dict[str, float]:
        """``{word: share of its occurrences next to a title or place word}`` for words with at least
        ``min_hits`` such occurrences making up at least ``min_ratio`` of the word's count in ``gen``."""
        out = {}
        for w, h in self.hits.items():
            if h < min_hits:
                continue
            i = gen._idx.get(w)
            if i is not None and h / gen._count[i] >= min_ratio:
                out[w] = h / gen._count[i]
        return out

    def save(self, path: PathLike) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"format": _CONTEXT_FORMAT, "person_titles": sorted(self.person_titles),
                                 "place_words": sorted(self.place_words), "prefixes": list(self.prefixes)},
                                ensure_ascii=False) + "\n")
            for w, h in self.hits.items():
                fh.write(f"{w}\t{h}\n")

    @classmethod
    def load(cls, path: PathLike) -> "TitleContext":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("format") != _CONTEXT_FORMAT:
                raise ValueError(f"{path} is not a title-context file")
            ctx = cls(header["person_titles"], header["place_words"], header["prefixes"])
            for line in fh:
                w, h = line.rstrip("\n").split("\t")
                ctx.hits[w] = int(h)
        return ctx


@dataclass
class StopwordFilter:
    """Exclusion rules applied to a ranked candidate list; see the module docstring.

    ``exclude`` and ``include`` map a normalized word to the source it came from (a file name or a
    rule name). ``ridf_max`` is compared with the ``ridf`` of the word in every generator passed to
    :meth:`select`; ``None`` disables the threshold.
    """

    exclude: Dict[str, str] = field(default_factory=dict)
    include: Dict[str, str] = field(default_factory=dict)
    ridf_max: Optional[float] = None

    def add_names(self, ctx: TitleContext, gen: StopwordGenerator, min_hits: int = 20,
                  min_ratio: float = 0.2) -> int:
        """Exclude the title and place words and the names found next to them. Returns how many were added."""
        before = len(self.exclude)
        for w in ctx.trigger_words:
            self.exclude.setdefault(w, "title or place word")
        for w, share in ctx.names(gen, min_hits, min_ratio).items():
            self.exclude.setdefault(w, f"title rule ({share:.0%} next to a title or place word)")
        return len(self.exclude) - before

    def reason(self, word: str, ridf: Sequence[Tuple[str, float]] = ()) -> Optional[str]:
        """Why ``word`` is excluded, or ``None`` if it is kept. ``ridf`` holds ``(corpus name, value)``."""
        if word in self.include:
            return None
        if word in self.exclude:
            return f"exclude: {self.exclude[word]}"
        if self.ridf_max is not None:
            over = [f"{name} {v:.2f}" for name, v in ridf if v > self.ridf_max]
            if over:
                return f"ridf > {self.ridf_max:g} ({', '.join(over)})"
        return None

    def select(
        self,
        ranked: Sequence[str],
        top_k: int,
        generators: Dict[str, StopwordGenerator],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Walk ``ranked`` (best first) and keep up to ``top_k`` words that pass the filter.

        ``generators`` maps a corpus name to its generator and supplies each word's ``ridf``.
        Returns ``(kept, rejected)`` where ``rejected`` lists ``(word, reason)`` in rank order.
        """
        ridf = {name: {r.word: r.ridf for r in g.scores_} for name, g in generators.items()}
        kept: List[str] = []
        rejected: List[Tuple[str, str]] = []
        for w in ranked:
            if len(kept) >= top_k:
                break
            why = self.reason(w, [(name, t[w]) for name, t in ridf.items() if w in t])
            if why is None:
                kept.append(w)
            else:
                rejected.append((w, why))
        return kept, rejected
