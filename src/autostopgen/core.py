"""Streaming corpus statistics and the stopword generator."""
from __future__ import annotations

import csv
import gzip
import json
import math
import warnings
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .io import PathLike
from .normalizers import Normalizer, default_normalizer
from .presets import Tokenizer, get_preset
from .tokenizers import default_tokenizer

METRICS_PAPER = ("tf", "idf", "entropy")
METRICS_ROBUST = ("tf", "idf", "burst", "deficit", "dispersion")  # experimental; do not use for ranking
METRICS = ("tf", "idf", "entropy", "burst", "deficit", "dispersion")
# True: a higher raw value makes a word look more like a stopword.
_HIGH_IS_STOP = {"tf": True, "idf": False, "entropy": True, "burst": False, "deficit": False, "dispersion": False}
_BANDED = ("burst", "deficit", "dispersion")  # ranked within frequency bands
_STATE_FORMAT = "autostopgen-state-2"

Aggregation = Union[str, Callable[[Dict[str, List[float]]], List[float]]]


@dataclass(frozen=True)
class WordScore:
    word: str
    count: int         # total occurrences
    df: int            # number of documents/blocks containing the word
    tf: float          # count / total tokens
    idf: float         # log(N / df)
    entropy: float     # normalized to [0, 1]; 1 = spread perfectly evenly over all documents
    burst: float       # observed IDF minus the IDF expected if the word were placed at random (Church & Gale)
    deficit: float     # (expected - observed entropy under random placement) / log N (Gerlach et al.)
    dispersion: float  # variance/mean of per-document counts; about 1 for a randomly placed word
    score: float       # aggregated stopword-likeness in [0, 1]


def percentile_ranks(values: Sequence[float], high_is_stop: bool) -> List[float]:
    """Rank in [0, 1] (ties share the average rank); 1 = most stopword-like."""
    n = len(values)
    order = sorted(range(n), key=values.__getitem__, reverse=not high_is_stop)
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[order[i]]
        while j + 1 < n and values[order[j + 1]] == v:
            j += 1
        p = ((i + j) / 2) / (n - 1) if n > 1 else 1.0
        for k in range(i, j + 1):
            out[order[k]] = p
        i = j + 1
    return out


def banded_percentile_ranks(
    values: Sequence[float], counts: Sequence[int], high_is_stop: bool, band_size: int
) -> List[float]:
    """Percentile ranks computed *within bands of words with similar frequency*.

    Burstiness grows with frequency, so a frequent function word looks bursty next to a rare
    word that is merely too rare to show any clustering. Comparing each word only with words
    of similar frequency (Church & Gale's point) removes that bias.
    """
    n = len(values)
    order = sorted(range(n), key=counts.__getitem__)
    starts = list(range(0, n, band_size))
    if len(starts) > 1 and n - starts[-1] < band_size // 2:
        starts.pop()  # merge a small last band into the previous one
    out = [0.0] * n
    for b, st in enumerate(starts):
        idxs = order[st : (starts[b + 1] if b + 1 < len(starts) else n)]
        ranks = percentile_ranks([values[k] for k in idxs], high_is_stop)
        for k, r in zip(idxs, ranks):
            out[k] = r
    return out


def poisson_xlogx(lam: float) -> float:
    """E[X log X] for X ~ Poisson(lam); exact series for small lam, expansion for large."""
    if lam <= 0:
        return 0.0
    if lam > 50:
        return lam * math.log(lam) + 0.5 - 1.0 / (6.0 * lam)
    p = math.exp(-lam)
    total = 0.0
    for k in range(1, 2000):
        p *= lam / k
        if k >= 2:
            total += p * k * math.log(k)
        if k > lam and p * k * math.log(k + 1) < 1e-15:
            break
    return total


def expected_entropy(count: int, n_docs: int) -> float:
    """Expected entropy (nats) of a word with ``count`` occurrences placed at random in
    ``n_docs`` equal-size documents (Poisson approximation of the multinomial)."""
    return math.log(count) - n_docs * poisson_xlogx(count / n_docs) / count


class StopwordGenerator:
    """Generate stopwords from a corpus.

    Counting is streaming and uses a few arrays per vocabulary type, so very large
    corpora (tens of millions of words) are fine. Scoring is separate from counting:
    after ``fit`` you can change ``metrics``, ``top_k``, ``weights``, ``min_count``...
    and read ``stopwords_`` again without recounting.

    Metrics (stopword-like when...)
    -------------------------------
    ``tf`` high; ``idf`` low; ``entropy`` high: the three metrics of Miretie & Khedkar (2018).
    These are the default and the only ones recommended for ranking.

    ``burst`` (observed IDF minus the IDF expected if the word were placed at random; Church &
    Gale 1995), ``deficit`` (entropy shortfall versus random placement; Gerlach et al. 2019) and
    ``dispersion`` (variance/mean of per-document counts) are **diagnostics**. They show whether a
    frequent word is clustered by topic, and are exported to ``scores.csv`` and the review CSV.
    Do not use them to *rank* candidates: rare words are too sparse to show any clustering, so
    they all look "random" and outrank real stopwords (on a 140-million-word Amharic corpus the
    ranking was filled with rare inflected forms; on a synthetic corpus the paper metrics found 100%
    of the true stopwords and adding these dropped it to 13-37%). They assume equal-size blocks
    (``fit_stream``) and at least ~100 blocks.

    Parameters
    ----------
    tokenizer, normalizer
        ``normalizer(text) -> text`` then ``tokenizer(text) -> tokens``.
    metrics
        Subset of ``METRICS``. Default ``METRICS_PAPER`` = tf, idf, entropy. ``METRICS_ROBUST`` is
        kept for experiments only (see above).
    weights
        Optional ``{metric: weight}``; missing metrics get weight 1.
    aggregation
        ``"rank_mean"`` (weighted mean of percentile ranks, default), ``"rank_min"``
        (a word must rank high on *every* metric), or a callable taking
        ``{metric: [raw values]}`` and returning one score per candidate.
    top_k, threshold
        Selection: keep the ``top_k`` best words and/or words with ``score >= threshold``.
    min_count, min_df
        Candidate filters; rare words have unreliable statistics.
    length_normalize
        Entropy and dispersion use per-document relative frequencies. Fixed once counting starts.
    band_size
        ``burst``, ``deficit`` and ``dispersion`` are ranked within bands of this many candidates
        of similar frequency (default: max(100, candidates / 200)); ``tf``, ``idf`` and
        ``entropy`` are ranked over all candidates.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[Normalizer] = None,
        metrics: Sequence[str] = METRICS_PAPER,
        weights: Optional[Dict[str, float]] = None,
        aggregation: Aggregation = "rank_mean",
        top_k: Optional[int] = 300,
        threshold: Optional[float] = None,
        min_count: int = 10,
        min_df: int = 5,
        length_normalize: bool = True,
        band_size: Optional[int] = None,
        language: Optional[str] = None,
    ):
        self.tokenizer = tokenizer or default_tokenizer
        self.normalizer = normalizer or default_normalizer
        self.metrics = tuple(metrics)
        self.weights = dict(weights) if weights else {}
        self.aggregation = aggregation
        self.top_k = top_k
        self.threshold = threshold
        self.min_count = min_count
        self.min_df = min_df
        self.language = language
        self.band_size = band_size
        self._length_normalize = bool(length_normalize)
        self._validate()
        self._reset()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_preset(cls, language: str, **kwargs) -> "StopwordGenerator":
        """Use the normalizer/tokenizer preset for ``language`` (e.g. ``"am"``)."""
        preset = get_preset(language)
        kwargs.setdefault("tokenizer", preset.tokenizer)
        kwargs.setdefault("normalizer", preset.normalizer)
        kwargs.setdefault("language", language)
        return cls(**kwargs)

    def _validate(self):
        bad = [m for m in self.metrics if m not in METRICS]
        if bad or not self.metrics:
            raise ValueError(f"metrics must be a non-empty subset of {METRICS}, got {self.metrics}")
        if isinstance(self.aggregation, str) and self.aggregation not in ("rank_mean", "rank_min"):
            raise ValueError("aggregation must be 'rank_mean', 'rank_min' or a callable")

    def _reset(self):
        self._idx: Dict[str, int] = {}
        self._words: List[str] = []
        self._count = array("q")
        self._df = array("q")
        self._R = array("d")  # sum over docs of r_i (relative or raw frequency)
        self._S = array("d")  # sum over docs of r_i * log(r_i)
        self._Q = array("d")  # sum over docs of r_i ** 2
        self._lens: Dict[int, int] = {}  # block length -> number of blocks
        self.n_docs_ = 0
        self.n_tokens_ = 0
        self._raw_cache = None
        self._score_cache = None

    # ---------------------------------------------------------------- counting
    @property
    def length_normalize(self) -> bool:
        return self._length_normalize

    @property
    def vocab_size_(self) -> int:
        return len(self._words)

    def _tokens(self, text: str) -> List[str]:
        return list(self.tokenizer(self.normalizer(text)))

    def _new_word(self, w: str) -> int:
        i = len(self._words)
        self._idx[w] = i
        self._words.append(w)
        self._count.append(0)
        self._df.append(0)
        self._R.append(0.0)
        self._S.append(0.0)
        self._Q.append(0.0)
        return i

    def _add_block(self, tokens: Sequence[str]) -> None:
        length = len(tokens)
        if length == 0:
            return
        idx = self._idx
        count, df, R, S, Q = self._count, self._df, self._R, self._S, self._Q
        ln = self._length_normalize
        log = math.log
        for w, c in Counter(tokens).items():
            i = idx.get(w)
            if i is None:
                i = self._new_word(w)
            r = c / length if ln else float(c)
            count[i] += c
            df[i] += 1
            R[i] += r
            S[i] += r * log(r)
            Q[i] += r * r
        self._lens[length] = self._lens.get(length, 0) + 1
        self.n_docs_ += 1
        self.n_tokens_ += length
        self._raw_cache = self._score_cache = None

    def partial_fit(self, docs: Iterable[str]) -> "StopwordGenerator":
        """Add documents; each string is one document (entropy/IDF are per document)."""
        for doc in docs:
            self._add_block(self._tokens(doc))
        return self

    def fit(self, docs: Iterable[str]) -> "StopwordGenerator":
        """Reset, then count ``docs`` (one string per document)."""
        self._reset()
        return self.partial_fit(docs)

    def partial_fit_stream(self, texts: Iterable[str], block_tokens: int = 1000) -> "StopwordGenerator":
        """Add text whose document boundaries are unknown or meaningless.

        Tokens from consecutive strings are concatenated and cut into pseudo-documents
        of ``block_tokens`` tokens. A final block shorter than half a block is dropped.
        Use this for line-per-sentence corpora.
        """
        if block_tokens < 10:
            raise ValueError("block_tokens must be >= 10")
        buf: List[str] = []
        for text in texts:
            buf.extend(self._tokens(text))
            if len(buf) >= block_tokens:
                n_full = len(buf) // block_tokens
                for b in range(n_full):
                    self._add_block(buf[b * block_tokens : (b + 1) * block_tokens])
                buf = buf[n_full * block_tokens :]
        if len(buf) >= block_tokens // 2:
            self._add_block(buf)
        return self

    def fit_stream(self, texts: Iterable[str], block_tokens: int = 1000) -> "StopwordGenerator":
        self._reset()
        return self.partial_fit_stream(texts, block_tokens)

    def merge(self, other: "StopwordGenerator") -> "StopwordGenerator":
        """Add another generator's counts (e.g. corpora counted in parallel processes)."""
        if other._length_normalize != self._length_normalize:
            raise ValueError("cannot merge generators with different length_normalize")
        for j, w in enumerate(other._words):
            i = self._idx.get(w)
            if i is None:
                i = self._new_word(w)
            self._count[i] += other._count[j]
            self._df[i] += other._df[j]
            self._R[i] += other._R[j]
            self._S[i] += other._S[j]
            self._Q[i] += other._Q[j]
        for length, n in other._lens.items():
            self._lens[length] = self._lens.get(length, 0) + n
        self.n_docs_ += other.n_docs_
        self.n_tokens_ += other.n_tokens_
        self._raw_cache = self._score_cache = None
        return self

    # ----------------------------------------------------------------- scoring
    def _length_bins(self, max_bins: int = 64) -> List[Tuple[int, float]]:
        items = sorted(self._lens.items())
        if len(items) <= max_bins:
            return [(n, float(length)) for length, n in items]
        per = self.n_docs_ / max_bins
        bins, acc_n, acc_l = [], 0, 0.0
        for length, n in items:
            acc_n += n
            acc_l += length * n
            if acc_n >= per:
                bins.append((acc_n, acc_l / acc_n))
                acc_n, acc_l = 0, 0.0
        if acc_n:
            bins.append((acc_n, acc_l / acc_n))
        return bins

    def _compute_raw(self):
        """Raw metric values for every candidate word (independent of metric choice)."""
        if self.n_docs_ < 2:
            raise ValueError("need at least 2 documents/blocks to compute IDF and entropy; call fit first")
        key = (self.min_count, self.min_df)
        if self._raw_cache is not None and self._raw_cache[0] == key:
            return self._raw_cache[1], self._raw_cache[2]
        self._warn_if_unsegmented()
        N, T = self.n_docs_, self.n_tokens_
        log_n = math.log(N)
        count, df, R, S, Q = self._count, self._df, self._R, self._S, self._Q
        cand = [i for i in range(len(self._words)) if count[i] >= self.min_count and df[i] >= self.min_df]
        bins = self._length_bins()
        mean_len = T / N
        len_factor = mean_len if self._length_normalize else 1.0
        xlogx_cache: Dict[int, float] = {}
        raw: Dict[str, List[float]] = {m: [] for m in METRICS}
        for i in cand:
            F, d = count[i], df[i]
            idf = math.log(N / d)
            h_obs = math.log(R[i]) - S[i] / R[i]
            df_exp = sum(n_b * -math.expm1(-F * l_b / T) for n_b, l_b in bins)
            x = xlogx_cache.get(F)
            if x is None:
                x = xlogx_cache[F] = poisson_xlogx(F / N)
            h_null = math.log(F) - N * x / F
            mean_r = R[i] / N
            var_r = max(0.0, Q[i] / N - mean_r * mean_r)
            raw["tf"].append(F / T)
            raw["idf"].append(idf)
            # Rounded so that summation order (e.g. merged vs single-pass counts) cannot reorder ties.
            raw["entropy"].append(round(min(1.0, max(0.0, h_obs / log_n)), 12))
            raw["burst"].append(round(idf - math.log(N / df_exp), 10))
            raw["deficit"].append(round((h_null - h_obs) / log_n, 10))
            raw["dispersion"].append(round(len_factor * var_r / mean_r, 10))
        self._raw_cache = (key, cand, raw)
        return cand, raw

    def _cache_key(self):
        agg = self.aggregation if isinstance(self.aggregation, str) else id(self.aggregation)
        return (self.metrics, tuple(sorted(self.weights.items())), agg, self.min_count, self.min_df, self.band_size)

    def _compute_scores(self) -> List[WordScore]:
        cand, raw = self._compute_raw()
        if not cand:
            return []
        n = len(cand)
        if callable(self.aggregation):
            scores = list(self.aggregation({m: raw[m] for m in self.metrics}))
            if len(scores) != n:
                raise ValueError("custom aggregation must return one score per candidate")
        else:
            counts = [self._count[i] for i in cand]
            band = self.band_size or max(100, n // 200)
            ranks = {
                m: (banded_percentile_ranks(raw[m], counts, _HIGH_IS_STOP[m], band)
                    if m in _BANDED else percentile_ranks(raw[m], _HIGH_IS_STOP[m]))
                for m in self.metrics
            }
            if self.aggregation == "rank_min":
                scores = [min(ranks[m][k] for m in self.metrics) for k in range(n)]
            else:
                w = {m: float(self.weights.get(m, 1.0)) for m in self.metrics}
                wsum = sum(w.values())
                if wsum <= 0:
                    raise ValueError("weights must sum to a positive number")
                scores = [sum(w[m] * ranks[m][k] for m in self.metrics) / wsum for k in range(n)]
        rows = [
            WordScore(self._words[i], self._count[i], self._df[i], raw["tf"][k], raw["idf"][k],
                      raw["entropy"][k], raw["burst"][k], raw["deficit"][k], raw["dispersion"][k], scores[k])
            for k, i in enumerate(cand)
        ]
        rows.sort(key=lambda r: (-r.score, -r.tf, r.word))
        return rows

    def _warn_if_unsegmented(self):
        long_tokens = sum(self._count[i] for i, w in enumerate(self._words) if len(w) > 15)
        if self.n_tokens_ and long_tokens / self.n_tokens_ > 0.2:
            warnings.warn(
                "More than 20% of tokens are longer than 15 characters; this language may be "
                "written without spaces. Pass a word segmenter as `tokenizer`, or try "
                "autostopgen.CharNgramTokenizer.",
                stacklevel=4,
            )

    @property
    def scores_(self) -> List[WordScore]:
        """All candidate words with their metrics, best stopword candidates first."""
        key = self._cache_key()
        if self._score_cache is None or self._score_cache[0] != key:
            self._score_cache = (key, self._compute_scores())
        return self._score_cache[1]

    def get_stopwords(self, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]:
        top_k = self.top_k if top_k is None else top_k
        threshold = self.threshold if threshold is None else threshold
        rows = self.scores_
        if threshold is not None:
            rows = [r for r in rows if r.score >= threshold]
        if top_k is not None:
            rows = rows[:top_k]
        return [r.word for r in rows]

    @property
    def stopwords_(self) -> List[str]:
        return self.get_stopwords()

    def coverage(self, words: Optional[Iterable[str]] = None) -> float:
        """Fraction of all corpus tokens accounted for by ``words`` (default: ``stopwords_``)."""
        words = self.stopwords_ if words is None else words
        hit = sum(self._count[self._idx[w]] for w in set(words) if w in self._idx)
        return hit / self.n_tokens_ if self.n_tokens_ else 0.0

    # --------------------------------------------------------------- persistence
    def to_csv(self, path: PathLike, top: Optional[int] = None) -> None:
        rows = self.scores_[:top] if top else self.scores_
        selected = set(self.stopwords_)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["rank", "word", "selected", "score", "count", "df", "tf", "idf",
                         "entropy", "burst", "deficit", "dispersion"])
            for rank, r in enumerate(rows, 1):
                wr.writerow([rank, r.word, int(r.word in selected), f"{r.score:.6f}", r.count, r.df,
                             f"{r.tf:.8g}", f"{r.idf:.6f}", f"{r.entropy:.6f}", f"{r.burst:.6f}",
                             f"{r.deficit:.6f}", f"{r.dispersion:.6f}"])

    def save_stopwords(self, path: PathLike) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for w in self.stopwords_:
                fh.write(w + "\n")

    def save_state(self, path: PathLike) -> None:
        """Save raw counts (gzip TSV) so you can re-score later without recounting."""
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "format": _STATE_FORMAT,
                "n_docs": self.n_docs_,
                "n_tokens": self.n_tokens_,
                "length_normalize": self._length_normalize,
                "lengths": {str(k): v for k, v in self._lens.items()},
            }) + "\n")
            for i, w in enumerate(self._words):
                if "\t" in w or "\n" in w:
                    raise ValueError(f"token contains a tab or newline and cannot be saved: {w!r}")
                fh.write(f"{w}\t{self._count[i]}\t{self._df[i]}\t{self._R[i]!r}\t{self._S[i]!r}\t{self._Q[i]!r}\n")

    @classmethod
    def load_state(cls, path: PathLike, **kwargs) -> "StopwordGenerator":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("format") != _STATE_FORMAT:
                raise ValueError(
                    "state file was written by an older autostopgen (or is not a state file); "
                    "recount the corpus to create a new one")
            kwargs["length_normalize"] = header["length_normalize"]
            gen = cls(**kwargs)
            for line in fh:
                w, c, d, r, s, q = line.rstrip("\n").split("\t")
                i = gen._new_word(w)
                gen._count[i] = int(c)
                gen._df[i] = int(d)
                gen._R[i] = float(r)
                gen._S[i] = float(s)
                gen._Q[i] = float(q)
            gen._lens = {int(k): v for k, v in header["lengths"].items()}
            gen.n_docs_ = header["n_docs"]
            gen.n_tokens_ = header["n_tokens"]
        return gen
