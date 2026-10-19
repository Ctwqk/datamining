import numpy as np

from airline_reviews.features import VADER_COLUMNS, make_tfidf, vader_matrix, vader_scores


def test_tfidf_is_fit_on_training_vocabulary_only():
    vec = make_tfidf(min_df=1, ngram_range=(1, 1))
    X = vec.fit_transform(["good seat", "good crew"])
    assert set(vec.get_feature_names_out()) == {"good", "seat", "crew"}
    unseen = vec.transform(["terrible meal"])
    assert unseen.nnz == 0  # unseen words are ignored, nothing leaks from the test text
    assert X.shape == (2, 3)


def test_tfidf_rows_are_l2_normalised():
    vec = make_tfidf(min_df=1)
    X = vec.fit_transform(["great crew great seat", "bad food"])
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(norms, 1.0)


def test_vader_scores_shape_and_polarity(nltk_ready):
    df = vader_scores(
        ["I love this airline, wonderful crew!", "Horrible, rude staff and lost bags."]
    )
    assert list(df.columns) == VADER_COLUMNS
    assert df.loc[0, "compound"] > 0.5 > -0.5 > df.loc[1, "compound"]
    assert np.allclose(df[["neg", "neu", "pos"]].sum(axis=1), 1.0, atol=0.01)


def test_vader_matrix_is_float_array(nltk_ready):
    arr = vader_matrix(["fine"])
    assert arr.shape == (1, 4) and arr.dtype == float
