"""autostopgen: generate stopword lists from a corpus, for any language."""
from .consensus import ConsensusScore, consensus_scores, consensus_stopwords
from .core import METRICS, METRICS_PAPER, METRICS_ROBUST, StopwordGenerator, WordScore, percentile_ranks
from .evaluate import compare, kendall_tau, list_agreement
from .filters import StopwordFilter, TitleContext, load_lexicon
from .io import iter_documents, iter_lines
from .lists import available_languages, data_file, load_stopwords, read_stopword_file, remove_stopwords
from .normalizers import amharic_normalizer, default_normalizer, make_translation_normalizer
from .presets import Preset, get_preset, register_preset
from .review import export_review, load_review
from .tokenizers import CharNgramTokenizer, WordTokenizer

__version__ = "0.1.0"

__all__ = [
    "ConsensusScore", "consensus_scores", "consensus_stopwords",
    "METRICS", "METRICS_PAPER", "METRICS_ROBUST", "StopwordGenerator", "WordScore", "percentile_ranks",
    "compare", "kendall_tau", "list_agreement", "StopwordFilter", "TitleContext", "load_lexicon",
    "iter_documents", "iter_lines", "available_languages", "data_file", "load_stopwords",
    "read_stopword_file", "remove_stopwords", "amharic_normalizer", "default_normalizer",
    "make_translation_normalizer", "Preset", "get_preset", "register_preset", "export_review",
    "load_review", "CharNgramTokenizer", "WordTokenizer", "__version__",
]
