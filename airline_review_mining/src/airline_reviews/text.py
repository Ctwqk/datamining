"""Text preprocessing (the original coursework recipe, made fast and testable).

The recipe is unchanged from the original ``get_tokenized_data`` in ``model2.py``:

1. strip HTML tags,
2. word-tokenise and lower-case,
3. keep only tokens found in the NLTK ``words`` dictionary,
4. drop English stop words,
5. POS-tag and lemmatise with WordNet.

Reviews that end up empty become the single token ``flight`` (an LSTM cannot take an
empty sequence). Compared with the original, stop words and dictionary words are held
in sets (the original rebuilt the stop-word list for every token) and NLTK data is
fetched into a project-local folder instead of the user's home directory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import nltk

from airline_reviews.config import NLTK_DATA_DIR

# resource name -> candidate paths for nltk.data.find
NLTK_RESOURCES: dict[str, tuple[str, ...]] = {
    "words": ("corpora/words", "corpora/words.zip"),
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "punkt_tab": ("tokenizers/punkt_tab",),
    "averaged_perceptron_tagger_eng": ("taggers/averaged_perceptron_tagger_eng",),
    "vader_lexicon": ("sentiment/vader_lexicon.zip", "sentiment/vader_lexicon"),
}

_HTML_TAG = re.compile(r"<.*?>")
EMPTY_TOKEN = "flight"


def ensure_nltk_data(resources: Iterable[str] | None = None) -> None:
    """Make sure the NLTK resources are available, downloading missing ones once."""
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    for name in resources if resources is not None else NLTK_RESOURCES:
        if _is_available(name):
            continue
        nltk.download(name, download_dir=str(NLTK_DATA_DIR), quiet=True)
        if not _is_available(name):
            raise RuntimeError(f"NLTK resource {name!r} could not be downloaded")


def _is_available(name: str) -> bool:
    for candidate in NLTK_RESOURCES[name]:
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def strip_html(text: str) -> str:
    """Remove ``<...>`` tags."""
    return _HTML_TAG.sub("", text)


def wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn-Treebank tag to the WordNet POS used by the lemmatiser (default: noun)."""
    from nltk.corpus import wordnet

    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(
        treebank_tag[:1], wordnet.NOUN
    )


class TextPreprocessor:
    """Callable that turns a raw review into a lemmatised, stop-word-free string."""

    def __init__(self, empty_token: str = EMPTY_TOKEN) -> None:
        ensure_nltk_data(
            ["words", "stopwords", "wordnet", "punkt_tab", "averaged_perceptron_tagger_eng"]
        )
        from nltk.corpus import stopwords, words
        from nltk.stem import WordNetLemmatizer

        self.empty_token = empty_token
        self._english = frozenset(words.words())
        self._stop = frozenset(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def __call__(self, text: str) -> str:
        tokens = nltk.word_tokenize(strip_html(text))
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t in self._english and t not in self._stop]
        if not tokens:
            return self.empty_token
        lemmas = [
            self._lemmatizer.lemmatize(tok, wordnet_pos(tag)) for tok, tag in nltk.pos_tag(tokens)
        ]
        return " ".join(lemmas)

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]
