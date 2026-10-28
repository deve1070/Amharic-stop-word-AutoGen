#!/usr/bin/env python3
"""Build the bundled Amharic stopword list from your corpora.

    python scripts/build_amharic_list.py --cc100 am.txt --gpac GPAC.txt --top 300 --work-dir build_out
    python scripts/build_amharic_list.py --cc100 am.txt --gpac GPAC.txt --reuse-state --work-dir build_out   # no recount

Each corpus is counted separately in blocks of --block-tokens to twice that many tokens. cc100 blocks
end at document boundaries (blank lines); GPAC has no boundaries, so its blocks end at arbitrary
whitespace. Words are ranked by tf + idf + entropy (the 2018 paper), then:
  * combined.txt   ranked on the two corpora merged
  * consensus.txt  ranked per corpus, then the minimum across corpora, then filtered    (becomes am.txt)
  * review.csv     candidates with a y/n "stopword" column to edit by hand and an "excluded_by" column
The noun filter (autostopgen.filters) removes words in data/am.exclude.txt, words whose residual IDF
exceeds --ridf-max in either corpus, and names found next to the titles and place words in
data/am.person_titles.txt and data/am.place_words.txt; data/am.include.txt overrides all of these.
Writes src/autostopgen/data/am.txt (+ am.meta.json). Review the list before releasing it.
"""
import argparse
import datetime
import json
import pathlib

import autostopgen
from autostopgen import (METRICS_PAPER, StopwordFilter, StopwordGenerator, TitleContext, consensus_scores,
                         data_file, export_review, get_preset, iter_documents, iter_lines, list_agreement,
                         load_lexicon, read_stopword_file)

CORPORA = (("cc100", "blank"), ("gpac", "lines"))  # (name, input units)
CLITIC_PREFIXES = ("የ", "በ", "ለ", "ከ", "ስለ", "እንደ", "ወደ")
KNOWN_FUNCTION = "ነው ላይ ውስጥ ወደ ጋር እና ግን ይህ ወይም ስለ እንደ ብቻ ጊዜ ነገር".split()
KNOWN_NOUNS = "ቤት ሰው ህዝብ አቶ አበባ ኢትዮጵያ የኢትዮጵያ መንግስት ሰዎች አመት ስራ ጉዳይ ሁኔታ መንገድ ምክንያት".split()


def write_list(path, words):
    pathlib.Path(path).write_text("\n".join(words) + "\n", encoding="utf-8")


def new_title_context(normalizer):
    titles = [normalizer(w) for w in read_stopword_file(data_file("am.person_titles.txt"))]
    places = [normalizer(w) for w in read_stopword_file(data_file("am.place_words.txt"))]
    return TitleContext(titles, places, prefixes=CLITIC_PREFIXES)


def count_corpus(name, path, units, block_tokens, work, reuse, kw):
    """Count one corpus (or load its saved counts). Returns (generator, title context)."""
    preset = get_preset("am")
    state = work / f"state_{name}_b{block_tokens}.tsv.gz"
    context = work / f"context_{name}_b{block_tokens}.tsv.gz"
    if reuse:
        g = StopwordGenerator.load_state(state, tokenizer=preset.tokenizer, normalizer=preset.normalizer,
                                         language="am", **kw)
        ctx = TitleContext.load(context)
    else:
        ctx = new_title_context(preset.normalizer)
        g = StopwordGenerator.from_preset("am", observers=[ctx], **kw)
        texts = iter_documents(path) if units == "blank" else iter_lines(path)
        g.partial_fit_stream(texts, block_tokens=block_tokens)
        g.save_state(state)
        ctx.save(context)
    print(f"{name}: {g.n_tokens_:,} tokens, {g.n_docs_:,} blocks of {block_tokens}+ tokens, {g.vocab_size_:,} types")
    return g, ctx


def print_agreement(label, a, b):
    res = list_agreement(a, b)
    print(f"{label}: " + ", ".join(f"top-{k} overlap {res[f'overlap@{k}']:.0f}/{k}, tau {res[f'tau@{k}']:+.3f}"
                                   for k in (150, 300)))


def print_known_words(gens, flt, normalizer):
    print("\nknown words (rank in each corpus's unfiltered ranking; filter verdict):")
    print(f"{'word':10} " + " ".join(f"{n + ' rank':>11} {'df/N':>5} {'ent':>5} {'ridf':>5}" for n in gens) + "  verdict")
    tables = {n: {r.word: (k, r) for k, r in enumerate(g.scores_, 1)} for n, g in gens.items()}
    for label, words in (("function words", KNOWN_FUNCTION), ("nouns and names", KNOWN_NOUNS)):
        print(f"- {label}")
        for w in map(normalizer, words):
            cells, ridf = [], []
            for n, g in gens.items():
                if w in tables[n]:
                    k, r = tables[n][w]
                    cells.append(f"{k:11} {r.df / g.n_docs_:5.2f} {r.entropy:5.2f} {r.ridf:5.2f}")
                    ridf.append((n, r.ridf))
                else:
                    cells.append(f"{'-':>11} {'':5} {'':5} {'':5}")
            print(f"{w:10} " + " ".join(cells) + f"  {flt.reason(w, ridf) or 'kept'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc100", required=True, help="CC-100 Amharic file (.txt/.xz/.gz), blank-line separated documents")
    ap.add_argument("--gpac", required=True, help="GPAC file (no document boundaries)")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--block-tokens", type=int, default=1000)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--consensus", choices=["min", "mean"], default="min")
    ap.add_argument("--ridf-max", type=float, default=0.40,
                    help="exclude words whose residual IDF exceeds this in either corpus (default 0.40)")
    ap.add_argument("--title-min-hits", type=int, default=20)
    ap.add_argument("--title-min-ratio", type=float, default=0.2,
                    help="title rule: share of a word's occurrences next to a title or place word")
    ap.add_argument("--review-size", type=int, default=600)
    ap.add_argument("--reuse-state", action="store_true", help="load saved counts from --work-dir instead of recounting")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).resolve().parents[1] / "src/autostopgen/data"))
    ap.add_argument("--work-dir", default="build_out")
    a = ap.parse_args()

    work = pathlib.Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    kw = dict(top_k=a.top, min_count=a.min_count, min_df=a.min_df, metrics=METRICS_PAPER)
    paths = {"cc100": a.cc100, "gpac": a.gpac}
    gens, ctxs = {}, {}
    for name, units in CORPORA:
        gens[name], ctxs[name] = count_corpus(name, paths[name], units, a.block_tokens, work, a.reuse_state, kw)

    combined = StopwordGenerator.from_preset("am", **kw)
    ctx = new_title_context(get_preset("am").normalizer)
    for name in gens:
        combined.merge(gens[name])
        ctx.merge(ctxs[name])
    combined_words = combined.stopwords_
    ranked = [r.word for r in consensus_scores(list(gens.values()), mode=a.consensus)]

    normalizer = get_preset("am").normalizer
    flt = StopwordFilter(
        exclude=load_lexicon([data_file("am.exclude.txt")], normalizer),
        include=load_lexicon([data_file("am.include.txt")], normalizer),
        ridf_max=a.ridf_max,
    )
    n_names = flt.add_names(ctx, combined, a.title_min_hits, a.title_min_ratio)
    final, rejected = flt.select(ranked, a.top, gens)
    depth = ranked.index(final[-1]) + 1 if final else 0

    per_corpus = {n: g.get_stopwords(top_k=None) for n, g in gens.items()}
    filtered = {n: flt.select(w, a.top, {n: gens[n]})[0] for n, w in per_corpus.items()}
    print()
    print_agreement("cc100 vs gpac, unfiltered", per_corpus["cc100"], per_corpus["gpac"])
    print_agreement("cc100 vs gpac, filtered  ", filtered["cc100"], filtered["gpac"])
    print_agreement("consensus unfiltered vs filtered", ranked, final)
    print(f"\nfilter: {len(flt.exclude)} excluded words ({n_names} from the title rule), {len(flt.include)} protected; "
          f"kept {len(final)} of the top {depth} consensus candidates, rejected {len(rejected)}")
    print_known_words(gens, flt, normalizer)
    print("\nrejected (rank order):")
    for w, why in rejected:
        print(f"  {w:14} {why}")

    write_list(work / "combined.txt", combined_words)
    write_list(work / "consensus.txt", final)
    combined.to_csv(work / "scores_combined.csv")
    export_review(combined, work / "review.csv", selected=final, top=a.review_size, excluded=dict(rejected))

    out = pathlib.Path(a.out_dir)
    (out / "am.txt").write_text(
        f"# Amharic stopwords generated by autostopgen (tf + idf + entropy, consensus={a.consensus}, "
        f"noun filter: exclude/include lists, ridf <= {a.ridf_max:g}, title rule)\n"
        + "\n".join(final) + "\n", encoding="utf-8")
    meta = {
        "language": "am", "autostopgen_version": autostopgen.__version__,
        "built": datetime.date.today().isoformat(),
        "corpora": {n: {"file": pathlib.Path(paths[n]).name, "units": u, "tokens": gens[n].n_tokens_,
                        "blocks": gens[n].n_docs_} for n, u in CORPORA},
        "block_tokens": a.block_tokens, "blocks_end_at_unit_boundaries": True,
        "min_count": a.min_count, "min_df": a.min_df,
        "metrics": list(METRICS_PAPER), "consensus": a.consensus, "top_k": a.top,
        "filter": {"ridf_max": a.ridf_max, "title_min_hits": a.title_min_hits,
                   "title_min_ratio": a.title_min_ratio, "excluded_words": len(flt.exclude),
                   "protected_words": len(flt.include), "rejected_from_ranking": len(rejected),
                   "ranking_depth": depth},
        "token_coverage_combined": round(combined.coverage(final), 4),
        "reviewed_by_hand": False,
    }
    (out / "am.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {out / 'am.txt'} ({len(final)} words); work files in {work}/")
    print(f"first 60: {' '.join(final[:60])}")


if __name__ == "__main__":
    main()
