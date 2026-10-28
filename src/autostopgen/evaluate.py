"""Compare a generated list against a reference list."""
from __future__ import annotations

from typing import Collection, Dict, Optional, Sequence

from .normalizers import Normalizer, default_normalizer


def compare(
    generated: Sequence[str],
    reference: Collection[str],
    ks: Sequence[int] = (50, 100, 200, 300),
    normalizer: Optional[Normalizer] = None,
) -> Dict[str, float]:
    """Precision@k, recall@k, overlap and Jaccard of ``generated`` (ranked) vs ``reference``.

    Both sides are normalized first (use the language's normalizer, e.g. the Amharic one).
    """
    norm = normalizer or default_normalizer
    ref = {norm(w) for w in reference}
    gen = []
    seen = set()
    for w in generated:
        n = norm(w)
        if n not in seen:
            seen.add(n)
            gen.append(n)
    out: Dict[str, float] = {"generated": len(gen), "reference": len(ref)}
    inter = len(set(gen) & ref)
    out["overlap"] = inter
    out["jaccard"] = inter / len(set(gen) | ref) if (gen or ref) else 0.0
    for k in ks:
        top = gen[:k]
        hit = sum(1 for w in top if w in ref)
        out[f"precision@{k}"] = hit / len(top) if top else 0.0
        out[f"recall@{k}"] = hit / len(ref) if ref else 0.0
    return out


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    """Kendall's tau between the orders of the words that ``a`` and ``b`` share (each ranked best first).

    1 = shared words in the same order, -1 = reversed, 0 = unrelated. Words found in only one list
    do not count; report the overlap next to it (see :func:`list_agreement`).
    """
    rank_b = {w: i for i, w in enumerate(b)}
    common = [rank_b[w] for w in a if w in rank_b]
    concordant = discordant = 0
    for i in range(len(common)):
        ri = common[i]
        for rj in common[i + 1:]:
            if rj > ri:
                concordant += 1
            elif rj < ri:
                discordant += 1
    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else 0.0


def list_agreement(a: Sequence[str], b: Sequence[str], ks: Sequence[int] = (150, 300)) -> Dict[str, float]:
    """Overlap and Kendall's tau of the top ``k`` of two ranked lists, for each ``k``."""
    out: Dict[str, float] = {}
    for k in ks:
        ta, tb = list(a[:k]), list(b[:k])
        out[f"overlap@{k}"] = len(set(ta) & set(tb))
        out[f"tau@{k}"] = kendall_tau(ta, tb)
    return out
