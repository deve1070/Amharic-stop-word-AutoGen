"""Combine stopword scores from several corpora (or sub-corpora).

Function words rank high in every source; topic- or genre-bound words (e.g. news vocabulary)
rank high only where the topic dominates. Taking the minimum score across corpora keeps the
former and drops the latter (cf. Turki et al. 2024, who find that most stopwords are shared
across news categories).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .core import StopwordGenerator


@dataclass(frozen=True)
class ConsensusScore:
    word: str
    score: float                 # min or mean of the per-corpus scores
    scores: Tuple[float, ...]    # one score per corpus, in the order given


def consensus_scores(generators: Sequence[StopwordGenerator], mode: str = "min") -> List[ConsensusScore]:
    """Score each word that is a candidate in *every* corpus; best first.

    ``mode="min"``: a word must look like a stopword in all corpora. ``mode="mean"``: average.
    Each generator ranks words with its own metrics and candidate filters.
    """
    if len(generators) < 2:
        raise ValueError("need at least two corpora/generators")
    if mode not in ("min", "mean"):
        raise ValueError("mode must be 'min' or 'mean'")
    tables = [{r.word: r.score for r in g.scores_} for g in generators]
    common = set(tables[0])
    for t in tables[1:]:
        common &= set(t)
    rows = []
    for w in common:
        per = tuple(t[w] for t in tables)
        rows.append(ConsensusScore(w, min(per) if mode == "min" else sum(per) / len(per), per))
    rows.sort(key=lambda r: (-r.score, -sum(r.scores), r.word))
    return rows


def consensus_stopwords(generators: Sequence[StopwordGenerator], top_k: int = 300, mode: str = "min") -> List[str]:
    return [r.word for r in consensus_scores(generators, mode)[:top_k]]
