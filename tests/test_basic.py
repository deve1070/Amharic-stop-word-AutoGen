import math
import random
import warnings

import pytest

import autostopgen as a
from autostopgen.cli import main as cli_main


# ------------------------------------------------------------ tokenization
def test_ethiopic_punctuation_and_numbers():
    toks = a.WordTokenizer()("ነው፡ ግን። 123 የኢትዮጵያ፣ ፩፪ ሰላም፤")
    assert toks == ["ነው", "ግን", "የኢትዮጵያ", "ሰላም"]


def test_combining_marks_stay_inside_words():
    assert a.WordTokenizer()("हिन्दी भाषा, नमस्ते!") == ["हिन्दी", "भाषा", "नमस्ते"]


def test_apostrophes_inside_but_not_edges():
    assert a.WordTokenizer()("don't 'quoted' l’homme") == ["don't", "quoted", "l’homme"]


def test_char_ngrams():
    assert a.CharNgramTokenizer(n=2)("今天天气") == ["今天", "天天", "天气"]


# ----------------------------------------------------------- normalization
def test_amharic_variants_merge():
    n = a.amharic_normalizer
    assert n("ነዉ") == n("ነው")
    assert n("ሐ") == n("ኀ") == n("ሀ")
    assert n("ሠላም") == n("ሰላም")
    assert n("ዐባይ") == n("አባይ")
    assert n("ፀሐይ") == n("ጸሀይ")
    assert n("ኅ") == "ህ"


# ----------------------------------------------------- statistics are right
def test_entropy_matches_definition():
    rng = random.Random(1)
    docs = [" ".join(rng.choice(["a", "b", "c"]) for _ in range(rng.randint(5, 40))) for _ in range(30)]
    g = a.StopwordGenerator(min_count=1, min_df=1, length_normalize=False, top_k=None).fit(docs)
    for s in g.scores_:
        f = [d.split().count(s.word) for d in docs]
        F = sum(f)
        H = -sum(x / F * math.log(x / F) for x in f if x)
        assert s.entropy == pytest.approx(H / math.log(len(docs)))
        assert s.count == F and s.df == sum(1 for x in f if x)
        assert s.idf == pytest.approx(math.log(len(docs) / s.df))


# ----------------------------------------------- planted-stopword recovery
def _planted_corpus(seed=0, n_blocks=600, block=200, n_func=40, n_topics=20, per_topic=60):
    rng = random.Random(seed)
    func = [f"fw{i}" for i in range(n_func)]
    fw = [1 / (i + 1) for i in range(n_func)]
    topics = [[f"t{t}x{i}" for i in range(per_topic)] for t in range(n_topics)]
    tw = [1 / (i + 1) for i in range(per_topic)]
    # Background vocabulary: large and flat, so each word is individually rare.
    bg = [f"bg{i}" for i in range(5000)]
    bw = [1] * 5000
    blocks = []
    for _ in range(n_blocks):
        topic = rng.choice(topics)
        toks = []
        for _ in range(block):
            u = rng.random()
            if u < 0.45:
                toks.append(rng.choices(func, fw)[0])
            elif u < 0.85:
                toks.append(rng.choices(topic, tw)[0])
            else:
                toks.append(rng.choices(bg, bw)[0])
        blocks.append(" ".join(toks))
    return blocks, set(func)


def test_recovers_planted_stopwords():
    docs, truth = _planted_corpus()
    g = a.StopwordGenerator(top_k=len(truth), min_count=10, min_df=5).fit(docs)
    prec = len(set(g.stopwords_) & truth) / len(truth)
    assert prec >= 0.9
    # frequency alone is fooled by frequent-but-bursty topic words
    tf_only = a.StopwordGenerator(metrics=["tf"], top_k=len(truth), min_count=10, min_df=5).fit(docs)
    assert len(set(tf_only.stopwords_) & truth) / len(truth) < prec


def test_stream_mode_matches_manual_blocks():
    docs, _ = _planted_corpus(n_blocks=50)
    flat = " ".join(docs).split()
    block = 200
    blocks = [" ".join(flat[i:i + block]) for i in range(0, len(flat), block)]
    g1 = a.StopwordGenerator(min_count=1, min_df=1, top_k=None).fit(blocks)
    g2 = a.StopwordGenerator(min_count=1, min_df=1, top_k=None).fit_stream(docs, block_tokens=block)
    assert [(s.word, s.count, s.df) for s in g1.scores_] == [(s.word, s.count, s.df) for s in g2.scores_]


def test_state_roundtrip_and_merge(tmp_path):
    docs, _ = _planted_corpus(n_blocks=100)
    g = a.StopwordGenerator(top_k=40).fit(docs)
    g.save_state(tmp_path / "s.gz")
    h = a.StopwordGenerator.load_state(tmp_path / "s.gz", top_k=40)
    assert h.stopwords_ == g.stopwords_
    p1 = a.StopwordGenerator().fit(docs[:50])
    p2 = a.StopwordGenerator().fit(docs[50:])
    merged = p1.merge(p2)
    whole = a.StopwordGenerator().fit(docs)
    assert merged.n_docs_ == whole.n_docs_ and merged.n_tokens_ == whole.n_tokens_
    assert merged.stopwords_ == whole.stopwords_


def test_rescoring_without_recount():
    docs, _ = _planted_corpus(n_blocks=100)
    g = a.StopwordGenerator(top_k=10).fit(docs)
    assert len(g.stopwords_) == 10
    g.top_k = 25
    assert len(g.stopwords_) == 25
    g.weights = {"entropy": 5}
    assert len(g.stopwords_) == 25


def test_custom_aggregation_and_rank_min():
    docs, truth = _planted_corpus(n_blocks=200)
    g = a.StopwordGenerator(aggregation="rank_min", top_k=len(truth)).fit(docs)
    assert len(set(g.stopwords_) & truth) / len(truth) >= 0.9
    h = a.StopwordGenerator(aggregation=lambda m: m["tf"], top_k=5).fit(docs)
    assert len(h.stopwords_) == 5


def test_warns_for_unsegmented_text():
    sent = "".join(chr(0x4E00 + (i * 7) % 300) for i in range(80))
    g = a.StopwordGenerator(min_count=1, min_df=1)
    g.fit([sent + "。" + sent[::-1] for _ in range(30)])
    with pytest.warns(UserWarning, match="without spaces"):
        g.scores_


def test_unknown_language_falls_back_with_warning():
    with pytest.warns(UserWarning, match="No preset"):
        a.StopwordGenerator.from_preset("xx")


# ------------------------------------------------------ list helpers & CLI
def test_remove_stopwords_uses_normalization():
    out = a.remove_stopwords("ነዉ ግን ሰላም", ["ነው", "ግን"], language="am")
    assert out == ["ሰላም"]


def test_load_stopwords_missing_is_helpful():
    with pytest.raises(LookupError, match="Available"):
        a.load_stopwords("zz")


def test_compare():
    res = a.compare(["a", "b", "c", "d"], ["a", "c", "z"], ks=(2, 4))
    assert res["precision@2"] == 0.5 and res["recall@4"] == pytest.approx(2 / 3)


def test_cli_generate(tmp_path):
    docs, truth = _planted_corpus(n_blocks=200)
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(docs), encoding="utf-8")
    out, scores = tmp_path / "sw.txt", tmp_path / "scores.csv"
    rc = cli_main(["generate", str(corpus), "--top", "40", "--block-tokens", "200",
                   "--out", str(out), "--scores", str(scores)])
    assert rc == 0
    words = a.read_stopword_file(out)
    assert len(set(words) & truth) / len(truth) >= 0.9
    assert scores.read_text(encoding="utf-8").startswith("rank,word,selected")
