Bundled stopword lists live here as <language-code>.txt (one word per line,
'#' lines are comments) with an optional <language-code>.meta.json describing
how the list was built. Build the Amharic list with scripts/build_amharic_list.py.

Noun-filter files for a language (used by autostopgen.StopwordFilter in scripts/build_amharic_list.py):
  <code>.include.txt        whitelist: never excluded by the filter
  <code>.exclude.txt        known nouns and names to exclude
  <code>.person_titles.txt  titles whose next one or two tokens are names (title rule)
  <code>.place_words.txt    place words whose previous token is a place name (title rule)
