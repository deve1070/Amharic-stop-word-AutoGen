#!/usr/bin/env python3
"""How much does the stopword ranking change with the block size?

    python scripts/block_size_sensitivity.py --cc100 am.txt --gpac GPAC.txt --work-dir build_out
    python scripts/block_size_sensitivity.py --cc100 am.txt --gpac GPAC.txt --work-dir build_out --reuse-state

Counts each corpus at every --sizes block size (saved as state_<corpus>_b<size>.tsv.gz, the files
build_amharic_list.py --reuse-state reads), ranks by tf + idf + entropy, and prints top-150 and
top-300 overlap and Kendall's tau between adjacent sizes, plus cc100 vs gpac agreement at each size.
Choose a block size where adjacent sizes agree (the plateau). The noun filter is not applied, so the
comparison measures the statistics alone.
"""
import argparse
import pathlib

from autostopgen import METRICS_PAPER, list_agreement
from build_amharic_list import CORPORA, count_corpus


def fmt(res):
    return "  ".join(f"top-{k} {res[f'overlap@{k}']:3.0f}/{k} tau {res[f'tau@{k}']:+.3f}" for k in (150, 300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc100", required=True)
    ap.add_argument("--gpac", required=True)
    ap.add_argument("--sizes", default="250,500,1000,2000,5000")
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--reuse-state", action="store_true")
    ap.add_argument("--work-dir", default="build_out")
    a = ap.parse_args()
    work = pathlib.Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in a.sizes.split(",")]
    paths = {"cc100": a.cc100, "gpac": a.gpac}
    kw = dict(top_k=None, min_count=a.min_count, min_df=a.min_df, metrics=METRICS_PAPER)

    lists = {}  # (corpus, size) -> top-300 ranking; generators are dropped to keep memory low
    for size in sizes:
        for name, units in CORPORA:
            state = work / f"state_{name}_b{size}.tsv.gz"
            g, _ = count_corpus(name, paths[name], units, size, work, a.reuse_state and state.exists(), kw)
            lists[name, size] = g.get_stopwords(top_k=300)
            del g

    print("\nadjacent block sizes:")
    for name, _ in CORPORA:
        for s1, s2 in zip(sizes, sizes[1:]):
            print(f"  {name:6} {s1:>5} vs {s2:<5} {fmt(list_agreement(lists[name, s1], lists[name, s2]))}")
    print("\ncc100 vs gpac at each block size:")
    for s in sizes:
        print(f"  {s:>5}         {fmt(list_agreement(lists['cc100', s], lists['gpac', s]))}")


if __name__ == "__main__":
    main()
