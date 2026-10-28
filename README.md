# autostopgen

Generate stopword lists **from your own corpus**, for any language, and ship a ready-made Amharic list.

Method adapted from Miretie & Khedkar, *Automatic Generation of Stopwords in the Amharic Text*
(IJCA 180(10), 2018), extended with ideas from the later literature (see References). The 2018 paper
does not say how its metrics are combined or evaluated; this package defines that (rank averaging) and
ships tests, an evaluation helper and a review workflow.

## Install

```bash
pip install autostopgen          # no dependencies, Python 3.9+
```

## Metrics

Each candidate word gets a percentile rank on each metric; the ranks are averaged (or the minimum is taken,
`aggregation="rank_min"`).

| metric | stopword-like when | use |
|---|---|---|
| `tf` | high | ranking |
| `idf` | low | ranking |
| `entropy` | high (spread evenly over documents) | ranking (the 2018 paper's three) |
| `ridf` | residual IDF: observed IDF minus the IDF expected at random (Church & Gale) | filter threshold, diagnostic |
| `deficit` | entropy shortfall versus random placement (Gerlach et al.) | diagnostic |
| `dispersion` | variance/mean of per-document counts | diagnostic |

The last three show whether a frequent word is clustered by topic and are written to `scores.csv` and the review
CSV to help a human reviewer. **Do not rank by them.** Rare words are too sparse to show clustering, so they look
random and outrank real stopwords: this was found on a 140-million-word Amharic corpus (the list filled with rare
inflected forms) and reproduced on a synthetic corpus, where the three paper metrics found 100% of the true stopwords
and adding the diagnostics lowered that to 13-37%. Ranking by `ridf` within frequency bands removed known nouns but
also the function words ስለ, እንደ, ወይም and ይህ. `ridf` works instead as a threshold on the ranked candidates (see
*Removing nouns and names*). `deficit` assumes equal-size blocks.

Gries's DP was evaluated and left out: on equal-size blocks it ranks the 5,000 most frequent Amharic words exactly
like document frequency (Spearman -1.000).

## Use

```python
from autostopgen import StopwordGenerator, consensus_stopwords, export_review, iter_documents, load_review

gen = StopwordGenerator.from_preset("am", top_k=300)
gen.fit_stream(iter_documents("cc100.txt"), block_tokens=1000)   # blocks end at document boundaries
# or gen.fit(documents): one string per document, no blocking
gen.stopwords_; gen.scores_[0]; gen.coverage()
gen.metrics = ("tf", "idf", "entropy")     # re-score without recounting
gen.save_state("counts.tsv.gz")            # StopwordGenerator.load_state(...) later

# keep only words that look like stopwords in every corpus (drops genre-bound words)
words = consensus_stopwords([gen_news, gen_books], top_k=300, mode="min")

# human review: edit the y/n "stopword" column in a spreadsheet, then load it
export_review(gen, "review.csv", selected=words, top=600)
final = load_review("review.csv")
```

Command line (reads `.txt`, `.gz`, `.xz`, `.bz2`):

```bash
autostopgen generate am.txt --units blank --lang am --top 300 \
    --save-state counts.tsv.gz --scores scores.csv --review review.csv --out am.txt
autostopgen rescore counts.tsv.gz --lang am --top 150
autostopgen apply-review review.csv --out am_final.txt
autostopgen compare am_final.txt reference.txt --lang am
```

## Blocks and documents

`fit_stream` packs consecutive input texts into blocks of `block_tokens` to `2 * block_tokens` tokens. A block ends
at the first text boundary after it reaches `block_tokens`; only a text that would push it past twice that size is
cut. Feed it whole documents (`iter_documents`, or `--units blank`) so that blocks rarely split an article, or lines
when document boundaries are unknown. Natural CC-100 documents are too short to use one per unit (median 67 tokens),
and IDF on very short units is noisy. `scripts/block_size_sensitivity.py` compares rankings across block sizes
(overlap and Kendall's tau, `autostopgen.list_agreement`).

## Removing nouns and names

Statistics cannot tell an evenly spread noun (ሰው "person", ጉዳይ "matter") from a function word. `StopwordFilter`
walks a ranked list and skips words that are

* in an exclude list (`load_lexicon` of word-list files; each entry records its file),
* above `ridf_max` residual IDF in any corpus (clustered by topic),
* names found by `TitleContext`: words that often follow a person title (አቶ X Y) or precede a place word
  (X ከተማ). Pass it as an observer while counting: `StopwordGenerator(observers=[ctx])`.

An include list (whitelist) overrides every exclusion. `select` returns the kept words and each rejected word with its
reason; pass the reasons to `export_review(..., excluded=...)` to fill the `excluded_by` column.

## Any language

The statistics are language-independent; only text handling differs.

* Default tokenizer: Unicode word splitting (any space-delimited script, combining marks kept, apostrophes kept inside words).
* **Unsegmented scripts** (Chinese, Japanese, Thai, Khmer, Burmese): pass a word segmenter as `tokenizer=`
  (any `Callable[[str], Iterable[str]]`), or use `CharNgramTokenizer` as a crude fallback. A warning is raised
  when the output looks unsegmented.
* Custom normalization: `register_preset("xx", Preset(name, normalizer, tokenizer))`.
* Heavily inflected languages produce many surface forms per word; plug in a morphology-aware tokenizer to merge them.

## Amharic

The `am` preset merges interchangeable letters (ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ, ዉ→ው, ኣ/ዓ→አ), expands slash abbreviations
(ዶ/ር→ዶክተር, ወ/ሮ→ወይዘሮ, ጠ/ሚ→ጠቅላይ ሚኒስትር, ...), handles Ethiopic punctuation (። ፡ ፣ ፤) and drops single-letter
fragments. Surface forms are otherwise kept; pass a morphology-aware tokenizer via `tokenizer=`.

The noun filter uses `data/am.exclude.txt` (words reviewed by hand as nouns, names, adjectives or content verbs),
`data/am.include.txt` (words reviewed as stopwords, including the light nouns ጊዜ and ነገር, plus pronouns that rank
lower), `data/am.person_titles.txt` and `data/am.place_words.txt`. Both lists are grouped by the reason recorded in
`reviews/am_review.csv`, the review of the top 600 candidates.

`scripts/build_amharic_list.py` counts each corpus separately (CC-100 in blocks that end at document boundaries),
writes the combined-corpus list and a cross-corpus consensus list (a word must score well in every corpus) passed
through the noun filter, plus a review CSV. It prints cc100-vs-GPAC agreement before and after filtering and a table
of known nouns and function words. `--reuse-state` re-scores saved counts without recounting. The bundled `am.txt` is the consensus list. Until you
build it, `load_stopwords("am")` raises a clear error.

## Large corpora

Counting is streaming: memory is proportional to vocabulary size, not corpus size. `iter_lines` cuts lines longer
than `max_chars` at whitespace, so a file without line breaks is also streamed. Save counts with
`save_state` and re-score later; count shards in parallel and combine them with `merge`.

## Notes and limits

* Statistics alone cannot separate every generic noun or news verb from a function word; review the list.
* Corpus genre matters. Count several corpora and use `consensus_stopwords` (a word must score well in every one), and state the sources when you publish a list.
* The deficit metric uses a Poisson approximation of random placement; accuracy drops below about 100 blocks.

## References

* Miretie & Khedkar (2018), Automatic Generation of Stopwords in the Amharic Text, IJCA 180(10).
* Church & Gale (1995), Inverse Document Frequency (IDF): A Measure of Deviations from Poisson.
* Gerlach, Shi & Amaral (2019), A universal information theoretic approach to the identification of stopwords, Nature Machine Intelligence 1.
* Zou et al. (2006), Automatic Identification of Chinese Stop Words, Research in Computing Science 18.
* Yeshambel, Mothe & Assabie (2021), Construction of Morpheme-Based Amharic Stopword List for Information Retrieval System, ICAST 2020.
* Sarica & Luo (2021), Stopwords in Technical Language Processing, PLoS ONE 16(8).
* Turki et al. (2024), Text Categorization Can Enhance Domain-Agnostic Stopword Extraction, arXiv:2401.13398.
* Lo, He & Ounis (2005), Automatically Building a Stopword List for an Information Retrieval System, DIR'05.
