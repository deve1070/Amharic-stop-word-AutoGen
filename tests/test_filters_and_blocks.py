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


def test_cli_blank_line_documents_without_blocking(tmp_path):
    from autostopgen.cli import main
    docs = _title_corpus()
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n\n".join(docs), encoding="utf-8")
    state = tmp_path / "s.gz"
    assert main(["generate", str(corpus), "--units", "blank", "--block-tokens", "0", "--min-count", "1",
                 "--min-df", "1", "--top", "4", "--save-state", str(state), "--out", str(tmp_path / "o.txt")]) == 0
    assert a.StopwordGenerator.load_state(state).n_docs_ == len(docs)
