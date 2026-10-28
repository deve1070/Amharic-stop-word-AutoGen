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
