from airline_reviews.text import EMPTY_TOKEN, strip_html, wordnet_pos


def test_strip_html_removes_tags_only():
    assert strip_html("great <b>seat</b><br/> and <i>crew</i>") == "great seat and crew"


def test_wordnet_pos_mapping():
    from nltk.corpus import wordnet

    assert wordnet_pos("JJR") == wordnet.ADJ
    assert wordnet_pos("VBD") == wordnet.VERB
    assert wordnet_pos("NNS") == wordnet.NOUN
    assert wordnet_pos("RB") == wordnet.ADV
    assert wordnet_pos("DT") == wordnet.NOUN  # default


def test_preprocessor_lowercases_filters_and_lemmatises(preprocessor):
    out = preprocessor("<p>The Flight was RUNNING late and the Staff rude!</p>")
    assert (
        out == "flight run late staff rude"
    )  # html, stop words, case, punctuation; running -> run


def test_preprocessor_removes_negations_like_the_original_recipe(preprocessor):
    """Documented weakness: 'not'/'no' are NLTK stop words, so negation is lost."""
    assert preprocessor("not good, no comfort") == "good comfort"


def test_preprocessor_drops_non_dictionary_tokens(preprocessor):
    out = preprocessor("Seat 12A zzqxv was comfortable")
    assert "zzqxv" not in out.split() and "12a" not in out.split()
    assert "seat" in out.split()


def test_preprocessor_empty_text_becomes_placeholder(preprocessor):
    assert preprocessor("") == EMPTY_TOKEN
    assert preprocessor("the and of 123 !!!") == EMPTY_TOKEN  # nothing survives filtering


def test_transform_matches_call(preprocessor):
    texts = ["good food", "bad seat"]
    assert preprocessor.transform(texts) == [preprocessor(t) for t in texts]
