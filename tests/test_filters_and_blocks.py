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
