"""Feature extraction: TF-IDF n-grams and VADER lexicon sentiment scores."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from airline_reviews.text import ensure_nltk_data

VADER_COLUMNS = ["neg", "neu", "pos", "compound"]


def make_tfidf(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    lowercase: bool = True,
) -> TfidfVectorizer:
    """TF-IDF vectoriser with sub-linear term frequency (fit it on training text only)."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=lowercase,
        strip_accents="unicode",
    )


_analyzer = None


def _vader():
    global _analyzer
    if _analyzer is None:
        ensure_nltk_data(["vader_lexicon"])
        from nltk.sentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def vader_scores(texts: Iterable[str]) -> pd.DataFrame:
    """VADER polarity scores (neg/neu/pos/compound) for each text.

    VADER relies on capitalisation and punctuation, so feed it the *raw* review rather
    than the lemmatised/stop-word-filtered version.
    """
    analyzer = _vader()
    rows = [analyzer.polarity_scores(t) for t in texts]
    return pd.DataFrame(rows, columns=VADER_COLUMNS)


def vader_matrix(texts: Iterable[str]) -> np.ndarray:
    """Same as :func:`vader_scores` but as a float array (for sklearn pipelines)."""
    return vader_scores(texts).to_numpy(dtype=float)
