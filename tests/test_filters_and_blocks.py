import csv

import pytest

import autostopgen as a


class Recorder:
    def __init__(self):
        self.blocks = []

    def __call__(self, tokens):
        self.blocks.append(list(tokens))


def test_stream_blocks_end_at_text_boundaries():
    texts = [" ".join(f"t{i}w{j}" for j in range(n)) for i, n in enumerate([30, 50, 40, 10, 120, 15, 60])]
    rec = Recorder()
    a.StopwordGenerator(min_count=1, min_df=1, observers=[rec]).fit_stream(texts, block_tokens=50)
    # 30+50 -> 80; 40+10 -> 50; 120 -> cut 50, keep 70; 15+60 -> 75
    assert [len(b) for b in rec.blocks] == [80, 50, 50, 70, 75]
    assert all(50 <= len(b) < 100 for b in rec.blocks)
    assert rec.blocks[0][-1] == "t1w49" and rec.blocks[1][-1] == "t3w9"


def test_iter_lines_splits_overlong_lines_at_whitespace(tmp_path):
    words = [f"w{i}" for i in range(5000)]
    f = tmp_path / "one_line.txt"
    f.write_text(" ".join(words) + "\nshort line\n", encoding="utf-8")
    pieces = list(a.iter_lines(f, max_chars=1000))
    assert max(len(p) for p in pieces) <= 1000
    assert " ".join(pieces).split() == words + ["short", "line"]


def test_amharic_normalizer_expands_slash_abbreviations():
    n, tok = a.amharic_normalizer, a.get_preset("am").tokenizer
    assert tok(n("ዶ/ር አብይ")) == ["ዶክተር", "አብይ"]
    assert tok(n("የወ/ሮ አልማዝ")) == ["የወይዘሮ", "አልማዝ"]
    assert tok(n("በጠ/ሚ መለስ")) == ["በጠቅላይ", "ሚኒስትር", "መለስ"]
    assert n("ፅ/ቤት") == "ጽህፈት ቤት"
    assert n("ሀ/ለ") == "ሀ/ለ"


def test_kendall_tau_and_list_agreement():
    assert a.kendall_tau(list("abcd"), list("abcd")) == 1.0
    assert a.kendall_tau(list("abcd"), list("dcba")) == -1.0
    assert a.kendall_tau(list("abcx"), list("acby")) == pytest.approx(1 / 3)
    res = a.list_agreement(list("abcdef"), list("abcxyz"), ks=(3, 6))
    assert res["overlap@3"] == 3 and res["overlap@6"] == 3 and res["tau@3"] == 1.0


def _title_corpus():
    func = "ነው ላይ ወደ ጋር".split()
    docs = []
    for i in range(200):
        toks = []
        for j in range(50):
            toks += [func[(i + j) % 4], func[(i + 2 * j) % 4]]
        toks += ["አቶ", "አበበ", "ከበደ", "ነው", "አዲስ", "አበባ", "ከተማ", "ላይ", "ቀበሌ"]
        if i % 2:
            toks += ["ወደ", "ቀበሌ"]
        docs.append(" ".join(toks))
    return docs


def test_title_rule_flags_names_but_not_function_words(tmp_path):
    ctx = a.TitleContext(["አቶ"], ["ከተማ", "ቀበሌ"])
    g = a.StopwordGenerator(min_count=1, min_df=1, top_k=None, observers=[ctx]).fit(_title_corpus())
    names = ctx.names(g, min_hits=20, min_ratio=0.2)
    assert {"አበበ", "ከበደ", "አበባ"} <= set(names)
    assert not {"ነው", "ላይ", "ወደ"} & set(names)  # follow titles too, but are far more frequent elsewhere
    ctx.save(tmp_path / "c.gz")
    assert a.TitleContext.load(tmp_path / "c.gz").hits == ctx.hits
    both = a.TitleContext.load(tmp_path / "c.gz").merge(ctx)
    assert both.hits["አበበ"] == 2 * ctx.hits["አበበ"]


def test_filter_select_applies_exclude_ridf_and_include():
    flt = a.StopwordFilter(exclude={"x": "nouns.txt", "y": "nouns.txt"}, include={"y": "keep.txt", "z": "keep.txt"},
                           ridf_max=0.4)
    assert flt.reason("x") == "exclude: nouns.txt"
    assert flt.reason("y") is None  # include overrides exclude
    assert flt.reason("w", [("c1", 0.1), ("c2", 0.5)]) == "ridf > 0.4 (c2 0.50)"
    assert flt.reason("z", [("c1", 0.9)]) is None

    docs = _title_corpus()
    g = a.StopwordGenerator(min_count=1, min_df=1, top_k=None).fit(docs)
    ranked = g.get_stopwords(top_k=None)
    flt = a.StopwordFilter(exclude={"ነው": "test"})
    kept, rejected = flt.select(ranked, 3, {"c": g})
    assert "ነው" not in kept and len(kept) == 3
    assert ("ነው", "exclude: test") in rejected


def test_add_names_excludes_titles_and_names():
    ctx = a.TitleContext(["አቶ"], ["ከተማ"], prefixes=["የ"])
    g = a.StopwordGenerator(min_count=1, min_df=1, top_k=None, observers=[ctx]).fit(_title_corpus())
    flt = a.StopwordFilter()
    flt.add_names(ctx, g)
    assert {"አቶ", "የአቶ", "ከተማ", "አበበ", "ከበደ", "አበባ"} <= set(flt.exclude)
    assert flt.exclude["አቶ"] == "title or place word"
    assert flt.exclude["አበበ"].startswith("title rule")


def test_review_csv_has_excluded_by_column(tmp_path):
    g = a.StopwordGenerator(min_count=1, min_df=1, top_k=3).fit(_title_corpus())
    a.export_review(g, tmp_path / "r.csv", excluded={"አቶ": "title or place word"})
    rows = list(csv.DictReader(open(tmp_path / "r.csv", encoding="utf-8")))
    by = {r["word"]: r for r in rows}
    assert by["አቶ"]["excluded_by"] == "title or place word"
    assert "ridf" in rows[0] and "burst" not in rows[0]


def test_bundled_filter_files_load_and_are_not_languages():
    norm = a.amharic_normalizer
    inc = a.load_lexicon([a.data_file("am.include.txt")], norm)
    exc = a.load_lexicon([a.data_file("am.exclude.txt")], norm)
    assert {"ጊዜ", "ነገር", "እኔ"} <= set(inc) and inc["ጊዜ"] == "am.include.txt"
    assert "መንግስት" in exc and not set(inc) & set(exc)
    assert not set(a.available_languages()) & {"am.include", "am.exclude", "am.person_titles", "am.place_words"}


def test_cli_blank_line_documents_without_blocking(tmp_path):
    from autostopgen.cli import main
    docs = _title_corpus()
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n\n".join(docs), encoding="utf-8")
    state = tmp_path / "s.gz"
    assert main(["generate", str(corpus), "--units", "blank", "--block-tokens", "0", "--min-count", "1",
                 "--min-df", "1", "--top", "4", "--save-state", str(state), "--out", str(tmp_path / "o.txt")]) == 0
    assert a.StopwordGenerator.load_state(state).n_docs_ == len(docs)
