import math
import random

import pytest

import autostopgen as a
from autostopgen.core import expected_entropy


def decoy_corpus(seed=3, n_blocks=800, block=200, n_func=30, n_decoy=10, on_prob=0.5, share=0.0045):
    """Function words are placed at random (Poisson). Decoys are as frequent and appear in as
    many blocks, but cluster: they are only 'on' in a random half of the blocks."""
    rng = random.Random(seed)
    func = [f"fw{i}" for i in range(n_func)]
    fw = [1 / (i + 1) for i in range(n_func)]
    fshare = 0.40
    decoys = [f"nw{i}" for i in range(n_decoy)]
    bg = [f"bg{i}" for i in range(6000)]
    blocks = []
    for _ in range(n_blocks):
        on = rng.random() < on_prob
        toks = []
        for _ in range(block):
            u = rng.random()
            if u < fshare:
                toks.append(rng.choices(func, fw)[0])
            elif u < fshare + n_decoy * share / (on_prob if on else 1e9):
                toks.append(rng.choice(decoys))
            else:
                toks.append(rng.choice(bg))
        blocks.append(" ".join(toks))
    return blocks, set(func), set(decoys)


def test_expected_entropy_matches_simulation():
    rng = random.Random(0)
    for N, F in [(200, 100), (1000, 300), (100, 5000)]:
        hs = []
        for _ in range(200):
            cnt = {}
            for _ in range(F):
                k = rng.randrange(N)
                cnt[k] = cnt.get(k, 0) + 1
            hs.append(-sum(v / F * math.log(v / F) for v in cnt.values()))
        assert expected_entropy(F, N) == pytest.approx(sum(hs) / len(hs), abs=0.01)


def test_randomly_placed_word_has_near_zero_burst_and_deficit_and_unit_dispersion():
    docs, truth, decoys = decoy_corpus()
    g = a.StopwordGenerator(min_count=100, min_df=20, top_k=None).fit(docs)
    by = {s.word: s for s in g.scores_}
    fw = [by[w] for w in truth if w in by and by[w].count > 1500]
    assert fw
    for s in fw:
        assert abs(s.burst) < 0.05 and abs(s.deficit) < 0.01 and abs(s.dispersion - 1) < 0.25
    for w in decoys:
        assert by[w].burst > 0.15 and by[w].dispersion > 1.5


def test_state_v2_roundtrip_and_merge_keep_new_metrics(tmp_path):
    docs, _, _ = decoy_corpus(n_blocks=300)
    kw = dict(metrics=a.METRICS, min_count=50, min_df=10, top_k=40)
    g = a.StopwordGenerator(**kw).fit(docs)
    g.save_state(tmp_path / "s.gz")
    h = a.StopwordGenerator.load_state(tmp_path / "s.gz", **kw)
    assert [(s.word, s.burst, s.deficit, s.dispersion) for s in h.scores_] == \
           [(s.word, s.burst, s.deficit, s.dispersion) for s in g.scores_]
    whole = a.StopwordGenerator(**kw).fit(docs)
    merged = a.StopwordGenerator(**kw).fit(docs[:150]).merge(a.StopwordGenerator(**kw).fit(docs[150:]))
    assert merged.stopwords_ == whole.stopwords_


def test_old_state_file_gives_clear_error(tmp_path):
    import gzip, json
    with gzip.open(tmp_path / "old.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"format": "autostopgen-state-1"}) + "\n")
    with pytest.raises(ValueError, match="recount"):
        a.StopwordGenerator.load_state(tmp_path / "old.gz")


def test_consensus_drops_words_that_are_stopwords_in_only_one_corpus():
    docs, truth, decoys = decoy_corpus(seed=5, n_blocks=500)
    # corpus B: same function words, but the decoys are absent (a different genre)
    rng = random.Random(9)
    func = sorted(truth)
    fw = [1 / (i + 1) for i in range(len(func))]
    bg = [f"bq{i}" for i in range(6000)]
    docs_b = [" ".join(rng.choices(func, fw)[0] if rng.random() < 0.4 else rng.choice(bg) for _ in range(200))
              for _ in range(500)]
    kw = dict(metrics=a.METRICS_ROBUST, min_count=100, min_df=20)
    ga = a.StopwordGenerator(**kw).fit(docs)
    gb = a.StopwordGenerator(**kw).fit(docs_b)
    cons = a.consensus_stopwords([ga, gb], top_k=len(truth))
    assert not (set(cons) & decoys)          # decoys are not candidates in corpus B at all
    assert len(set(cons) & truth) / len(truth) >= 0.9
    rows = a.consensus_scores([ga, gb], mode="min")
    assert all(r.score == min(r.scores) for r in rows)
    with pytest.raises(ValueError):
        a.consensus_scores([ga])


def test_review_roundtrip(tmp_path):
    docs, truth, _ = decoy_corpus(n_blocks=300)
    g = a.StopwordGenerator(top_k=30, min_count=50, min_df=10).fit(docs)
    path = tmp_path / "review.csv"
    n = a.export_review(g, path, top=60)
    assert 30 <= n <= 60 and n == min(60, len(g.scores_))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("stopword,word")
    # reviewer flips the first word to 'n'
    first_word = lines[1].split(",")[1]
    lines[1] = "n" + lines[1][1:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final = a.load_review(path)
    assert first_word not in final and len(final) == 29


def test_amharic_ayin_and_glottal_aa_merge_and_single_letters_dropped():
    n = a.amharic_normalizer
    assert n("ዓመት") == n("ኣመት") == n("አመት")
    assert n("ሰዓት") == n("ሰኣት") == n("ሰአት")
    tok = a.get_preset("am").tokenizer
    assert tok("ዶ/ር አበበ ወደ") == ["አበበ", "ወደ"]


def test_cli_rescore_and_apply_review(tmp_path):
    from autostopgen.cli import main
    docs, truth, _ = decoy_corpus(n_blocks=300)
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(docs), encoding="utf-8")
    state = tmp_path / "s.gz"
    assert main(["generate", str(corpus), "--top", "30", "--block-tokens", "200", "--min-count", "50",
                 "--min-df", "10", "--save-state", str(state), "--out", str(tmp_path / "a.txt")]) == 0
    assert main(["rescore", str(state), "--metrics", "robust", "--top", "30", "--min-count", "50",
                 "--min-df", "10", "--out", str(tmp_path / "b.txt"), "--review", str(tmp_path / "r.csv")]) == 0
    assert main(["apply-review", str(tmp_path / "r.csv"), "--out", str(tmp_path / "final.txt")]) == 0
    assert len(a.read_stopword_file(tmp_path / "final.txt")) == 30


def test_default_metrics_are_the_papers_three():
    assert a.StopwordGenerator().metrics == a.METRICS_PAPER


def test_rare_random_words_do_not_outrank_function_words_by_default():
    """Regression: ranking by the corrected metrics filled a real list with rare random-like words."""
    import itertools, bisect, math
    rng = random.Random(0)
    func = [f"fw{i}" for i in range(40)]
    fcum = list(itertools.accumulate(1 / (i + 1) for i in range(40)))
    bg = [f"bg{i}" for i in range(3000)]
    docs = []
    for _ in range(1500):
        mult = math.exp(rng.gauss(0, 0.35))
        toks = [func[bisect.bisect(fcum, rng.random() * fcum[-1])] if rng.random() < min(0.9, 0.38 * mult)
                else rng.choice(bg) for _ in range(300)]
        docs.append(" ".join(toks))
    g = a.StopwordGenerator(min_count=50, min_df=20, top_k=40).fit(docs)
    assert len(set(g.stopwords_) & set(func)) / 40 >= 0.95


def test_build_script_runs_and_reuses_state(tmp_path):
    import subprocess, sys, pathlib
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_amharic_list.py"
    rng = random.Random(1)
    func = "ነው ግን እና ወደ ላይ ነበር ጋር ውስጥ".split()
    topic = [["ትምህርት", "መምህር", "ተማሪ"], ["ኳስ", "ጨዋታ", "ቡድን"]]
    for name in ("a", "b"):
        lines = [" ".join(rng.choice(func) if rng.random() < .5 else rng.choice(rng.choice(topic)) for _ in range(40)) + " ።"
                 for _ in range(600)]
        (tmp_path / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "data"; out.mkdir()
    cmd = [sys.executable, str(script), "--cc100", str(tmp_path / "a.txt"), "--gpac", str(tmp_path / "b.txt"),
           "--top", "8", "--min-count", "20", "--min-df", "10", "--block-tokens", "100",
           "--out-dir", str(out), "--work-dir", str(tmp_path / "w")]
    r1 = subprocess.run(cmd, capture_output=True, text=True); assert r1.returncode == 0, r1.stderr
    first = (out / "am.txt").read_text(encoding="utf-8")
    r2 = subprocess.run(cmd + ["--reuse-state"], capture_output=True, text=True); assert r2.returncode == 0, r2.stderr
    assert (out / "am.txt").read_text(encoding="utf-8") == first
    assert len(a.read_stopword_file(out / "am.txt")) >= 6
