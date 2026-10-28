"""Command line interface: ``autostopgen generate|rescore|apply-review|compare|list``."""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from .core import METRICS_PAPER, METRICS_ROBUST, StopwordGenerator
from .evaluate import compare
from .io import iter_documents, iter_lines
from .lists import available_languages, read_stopword_file
from .presets import get_preset
from .review import export_review, load_review


def _parse_weights(text: Optional[str]) -> Optional[Dict[str, float]]:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def _parse_metrics(text: str):
    if text == "paper":
        return METRICS_PAPER
    if text == "robust":
        return METRICS_ROBUST
    return tuple(m.strip() for m in text.split(","))


def _make_gen(a: argparse.Namespace, state: Optional[str] = None) -> StopwordGenerator:
    kw = dict(top_k=a.top, threshold=a.threshold, min_count=a.min_count, min_df=a.min_df,
              metrics=_parse_metrics(a.metrics), weights=_parse_weights(a.weights), aggregation=a.aggregation)
    if state:
        preset = get_preset(a.lang or "default")
        return StopwordGenerator.load_state(state, tokenizer=preset.tokenizer, normalizer=preset.normalizer,
                                            language=a.lang, **kw)
    return StopwordGenerator.from_preset(a.lang or "default", **kw)


def _emit(gen: StopwordGenerator, a: argparse.Namespace) -> None:
    words = gen.stopwords_
    if a.out:
        gen.save_stopwords(a.out)
    else:
        print("\n".join(words))
    if a.scores:
        gen.to_csv(a.scores)
    if a.review:
        n = export_review(gen, a.review, top=a.review_size)
        print(f"wrote {n} review rows to {a.review}", file=sys.stderr)
    print(f"selected {len(words)} stopwords covering {gen.coverage():.1%} of tokens", file=sys.stderr)


def _cmd_generate(a: argparse.Namespace) -> int:
    gen = _make_gen(a)
    units = iter_documents(a.inputs) if a.units == "blank" else iter_lines(a.inputs)
    if a.block_tokens:
        gen.partial_fit_stream(units, block_tokens=a.block_tokens)
    else:
        gen.partial_fit(units)
    print(f"counted {gen.n_tokens_:,} tokens, {gen.n_docs_:,} documents/blocks, "
          f"{gen.vocab_size_:,} types", file=sys.stderr)
    if a.save_state:
        gen.save_state(a.save_state)
    _emit(gen, a)
    return 0


def _cmd_rescore(a: argparse.Namespace) -> int:
    _emit(_make_gen(a, state=a.state), a)
    return 0


def _cmd_apply_review(a: argparse.Namespace) -> int:
    words = load_review(a.review_csv)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(words) + "\n")
    print(f"wrote {len(words)} reviewed stopwords to {a.out}", file=sys.stderr)
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    norm = get_preset(a.lang or "default").normalizer
    res = compare(read_stopword_file(a.generated), read_stopword_file(a.reference), normalizer=norm)
    for k, v in res.items():
        print(f"{k}\t{v:.4f}" if isinstance(v, float) else f"{k}\t{v}")
    return 0


def _cmd_list(a: argparse.Namespace) -> int:
    langs = available_languages()
    print("\n".join(langs) if langs else "(no bundled lists)")
    return 0


def _scoring_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lang", help="language code for the normalizer/tokenizer preset (e.g. am)")
    p.add_argument("--top", type=int, default=300, help="number of stopwords to keep (default 300)")
    p.add_argument("--threshold", type=float, default=None, help="also/instead keep words with score >= threshold")
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--metrics", default="paper",
                   help="paper = tf,idf,entropy (default, recommended); robust = experimental; or a comma list")
    p.add_argument("--weights", help="e.g. tf=1,idf=1,ridf=2")
    p.add_argument("--aggregation", default="rank_mean", choices=["rank_mean", "rank_min"])
    p.add_argument("--out", help="write the list here (default: stdout)")
    p.add_argument("--scores", help="write all candidate scores as CSV")
    p.add_argument("--review", help="write a review CSV (edit the stopword column y/n)")
    p.add_argument("--review-size", type=int, default=600)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autostopgen", description="Generate stopword lists from a corpus.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a stopword list from text files (.txt/.gz/.xz/.bz2)")
    g.add_argument("inputs", nargs="+")
    _scoring_args(g)
    g.add_argument("--units", choices=["lines", "blank"], default="lines",
                   help="input units: lines (default) or blank-line separated documents (the CC-100 layout)")
    g.add_argument("--block-tokens", type=int, default=1000,
                   help="pack units into blocks of this many to twice this many tokens, ending at unit "
                        "boundaries (default 1000); 0 = each unit is one document")
    g.add_argument("--save-state", help="save raw counts (gzip) to re-score later")
    g.set_defaults(func=_cmd_generate)

    r = sub.add_parser("rescore", help="re-score saved counts without recounting")
    r.add_argument("state")
    _scoring_args(r)
    r.set_defaults(func=_cmd_rescore)

    ar = sub.add_parser("apply-review", help="build the final list from an edited review CSV")
    ar.add_argument("review_csv")
    ar.add_argument("--out", required=True)
    ar.set_defaults(func=_cmd_apply_review)

    c = sub.add_parser("compare", help="compare a generated list with a reference list")
    c.add_argument("generated")
    c.add_argument("reference")
    c.add_argument("--lang")
    c.set_defaults(func=_cmd_compare)

    l = sub.add_parser("list", help="list bundled languages")
    l.set_defaults(func=_cmd_list)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
