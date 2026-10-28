#!/usr/bin/env python3
"""Evaluate the bundled Amharic list against published lists and out-of-genre text.

    python scripts/evaluate_amharic_list.py --work-dir build_out \\
        --spacy spacy/lang/am/stop_words.py --uhhlt amstopwords.txt --irit StopWord-list.txt \\
        --wiki amwiki.txt

Reference lists (any subset):
  --spacy  spaCy's spacy/lang/am/stop_words.py (Teshome Kassie; Tihitina Petros)
  --uhhlt  amstopwords.txt from the Hugging Face dataset uhhlt/amharic-stopwords
  --irit   StopWord-list.txt from irit.fr/AmharicResources (Yeshambel, Mothe & Assabie 2021); a
           morpheme-based list, so entries that are affixes or never occur as standalone tokens are skipped
--wiki is plain text with blank-line-separated documents (scripts/extract_wikipedia.py).

For every reference word missing from the bundled list, the report says whether the hand review
rejected it (with the reason) or never saw it. Word frequencies come from the saved cc100 and GPAC
counts in --work-dir (build_amharic_list.py).
"""
import argparse
import csv
import pathlib
from collections import Counter

from autostopgen import METRICS_PAPER, StopwordGenerator, get_preset, iter_documents, list_agreement, load_stopwords

PRESET = get_preset("am")
NORM, TOK = PRESET.normalizer, PRESET.tokenizer


def per_million(gen, word):
    i = gen._idx.get(word)
    return 1e6 * gen._count[i] / gen.n_tokens_ if i is not None else 0.0


def read_spacy(path):
    src = pathlib.Path(path).read_text(encoding="utf-8")
    return src[src.index('"""') + 3:src.rindex('"""')].split()


def read_lines(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [line.strip() for line in fh if line.strip()]


def words_of(entries, news, morphemes=False):
    """Normalized single-token Amharic words of a reference list, and counts of skipped entries."""
    out, skipped = [], Counter()
    for e in entries:
        if morphemes and ("-" in e or len(e) < 2):
            skipped["affix or morpheme"] += 1
            continue
        t = TOK(NORM(e))
        if len(t) != 1:
            skipped["punctuation or several words"] += 1
        elif not any("ሀ" <= c <= "፿" for c in t[0]):
            skipped["not Amharic"] += 1
        elif morphemes and per_million(news, t[0]) < 2:
            skipped["affix or morpheme"] += 1  # (almost) never a standalone token
        else:
            out.append(t[0])
    return list(dict.fromkeys(out)), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="build_out")
    ap.add_argument("--block-tokens", type=int, default=1000)
    ap.add_argument("--review", default=str(pathlib.Path(__file__).resolve().parents[1] / "reviews/am_review.csv"))
    ap.add_argument("--spacy")
    ap.add_argument("--uhhlt")
    ap.add_argument("--irit")
    ap.add_argument("--wiki")
    a = ap.parse_args()

    ours = load_stopwords("am")
    ours_set = set(ours)
    with open(a.review, encoding="utf-8-sig") as fh:
        review = {r["word"]: r for r in csv.DictReader(fh)}
    kw = dict(tokenizer=PRESET.tokenizer, normalizer=NORM, min_count=20, min_df=10, top_k=None)
    work = pathlib.Path(a.work_dir)
    news = StopwordGenerator.load_state(work / f"state_cc100_b{a.block_tokens}.tsv.gz", **kw)
    news.merge(StopwordGenerator.load_state(work / f"state_gpac_b{a.block_tokens}.tsv.gz", **kw))

    refs = {}
    if a.spacy:
        refs["spaCy"] = words_of(read_spacy(a.spacy), news)
    if a.uhhlt:
        refs["uhhlt"] = words_of(read_lines(a.uhhlt), news)
    if a.irit:
        refs["IRIT"] = words_of(read_lines(a.irit), news, morphemes=True)

    for name, (ref, skipped) in refs.items():
        inter = [w for w in ref if w in ours_set]
        rejected = [w for w in ref if w not in ours_set and w in review]
        unseen = sorted((w for w in ref if w not in ours_set and w not in review), key=lambda w: -per_million(news, w))
        print(f"\n== {name}: {len(ref)} words (skipped: {dict(skipped)})")
        print(f"in the bundled list: {len(inter)}/{len(ref)} ({len(inter) / len(ref):.0%})")
        reasons = Counter(review[w]["note"].split(" [")[0].replace(" – check", "") for w in rejected)
        print(f"rejected in the review: {len(rejected)} {dict(reasons.most_common())}")
        print("  " + " ".join(rejected))
        print(f"never reviewed: {len(unseen)} (per million tokens in cc100+gpac)")
        print("  " + " ".join(f"{w}({per_million(news, w):.0f})" for w in unseen))
    names = list(refs)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            shared = set(refs[x][0]) & set(refs[y][0])
            print(f"\n{x} vs {y}: {len(shared)} shared ({len(shared) / len(refs[x][0]):.0%} of {x}, "
                  f"{len(shared) / len(refs[y][0]):.0%} of {y})")
    if refs:
        in_all = set.intersection(*(set(r) for r, _ in refs.values()))
        print(f"words in every reference list: {len(in_all)}; missing from the bundled list: "
              f"{' '.join(sorted(in_all - ours_set)) or 'none'}")
        in_any = set.union(*(set(r) for r, _ in refs.values()))
        print(f"bundled words in no reference list: {sum(w not in in_any for w in ours)}/{len(ours)}")

    if a.wiki:
        wiki = StopwordGenerator.from_preset("am", top_k=None, min_count=20, min_df=10, metrics=METRICS_PAPER)
        wiki.partial_fit_stream(iter_documents(a.wiki), block_tokens=a.block_tokens)
        print(f"\n== out of genre: {a.wiki} ({wiki.n_tokens_:,} tokens, {wiki.n_docs_:,} blocks)")
        print(f"token coverage of the bundled list: cc100+gpac {news.coverage(ours):.1%}, here {wiki.coverage(ours):.1%}")
        ratio = {w: (per_million(wiki, w) + 1) / (per_million(news, w) + 1) for w in ours}
        rare = sorted((w for w in ours if ratio[w] < 0.2), key=ratio.get)
        print(f"bundled words at least 5x rarer here than in cc100+gpac: {len(rare)}")
        print("  " + " ".join(f"{w}({per_million(news, w):.0f}->{per_million(wiki, w):.0f})" for w in rare))
        top = [r.word for r in wiki.scores_[:400]]
        unseen = [w for w in top if w not in ours_set and w not in review]
        print(f"top-400 candidates here that were never reviewed: {len(unseen)} (per million here / in cc100+gpac)")
        print("  " + " ".join(f"{w}({per_million(wiki, w):.0f}/{per_million(news, w):.0f})" for w in unseen))
        res = list_agreement(wiki.get_stopwords(top_k=300), news.get_stopwords(top_k=300), ks=(300,))
        print(f"automatic top-300 here vs cc100+gpac: overlap {res['overlap@300']:.0f}/300, tau {res['tau@300']:+.3f}")


if __name__ == "__main__":
    main()
