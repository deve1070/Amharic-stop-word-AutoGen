"""Human review of candidates (the automatic statistics cannot tell a generic noun from a
function word; Sarica & Luo 2021 add expert review for the same reason).

Workflow: ``export_review`` writes a CSV with a ``stopword`` column pre-filled ``y`` for the
automatic selection and ``n`` for the rest. Change cells in a spreadsheet, then
``load_review`` returns the final list.
"""
from __future__ import annotations

import csv
from typing import Collection, Dict, Optional, Sequence

from .core import StopwordGenerator
from .io import PathLike

_YES = {"y", "yes", "1", "true", "stop", "stopword"}


def export_review(
    gen: StopwordGenerator,
    path: PathLike,
    selected: Optional[Sequence[str]] = None,
    top: int = 600,
    excluded: Optional[Dict[str, str]] = None,
) -> int:
    """Write up to ``top`` candidate rows. ``selected`` (ordered) defaults to ``gen.stopwords_``;
    pass a consensus list to review that instead. ``excluded`` maps a word to the reason a
    :class:`~autostopgen.filters.StopwordFilter` rejected it; it fills the ``excluded_by`` column.
    Returns the number of rows written."""
    excluded = excluded or {}
    selected = list(gen.stopwords_ if selected is None else selected)
    by_word = {r.word: r for r in gen.scores_}
    ordered = [w for w in selected if w in by_word]
    seen = set(ordered)
    ordered += [r.word for r in gen.scores_ if r.word not in seen]
    ordered = ordered[: max(top, len(selected))]
    sel = set(selected)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stopword", "word", "rank", "count", "df", "tf", "idf", "entropy",
                     "ridf", "deficit", "dispersion", "excluded_by", "note"])
        for rank, w in enumerate(ordered, 1):
            r = by_word[w]
            wr.writerow(["y" if w in sel else "n", w, rank, r.count, r.df, f"{r.tf:.8g}", f"{r.idf:.4f}",
                         f"{r.entropy:.4f}", f"{r.ridf:.4f}", f"{r.deficit:.4f}", f"{r.dispersion:.4f}", excluded.get(w, ""), ""])
    return len(ordered)


def load_review(path: PathLike) -> list:
    """Return the words whose ``stopword`` cell is y/yes/1/true, in file order."""
    out = []
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if (row.get("stopword") or "").strip().lower() in _YES and row.get("word"):
                out.append(row["word"].strip())
    return out
