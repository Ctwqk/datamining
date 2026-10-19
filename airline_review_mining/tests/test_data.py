import pandas as pd
import pytest

from airline_reviews.data import (
    encode_recommended,
    join_header_content,
    load_reviews,
    subsample,
    train_test_split_reviews,
)


def _write(tmp_path, df):
    path = tmp_path / "reviews.csv"
    df.to_csv(path)  # index column mimics the committed CSV's 'Unnamed: 0'
    return path


def test_encode_recommended_is_case_insensitive():
    out = encode_recommended(pd.Series(["yes", " No ", "YES"]))
    assert out.tolist() == [1, 0, 1]


def test_encode_recommended_rejects_unknown_values():
    with pytest.raises(ValueError, match="maybe"):
        encode_recommended(pd.Series(["yes", "maybe"]))


def test_join_header_content():
    out = join_header_content(pd.Series(['"Nice trip."']), pd.Series(["Crew were kind."]))
    assert out.iloc[0].endswith("Crew were kind")
    assert "  " not in out.iloc[0]


def test_load_reviews_adds_label_text_and_drops_duplicates(tmp_path, tiny_reviews):
    dup = pd.concat([tiny_reviews, tiny_reviews.iloc[[0]]], ignore_index=True)
    df = load_reviews(_write(tmp_path, dup))
    assert len(df) == len(tiny_reviews)
    assert set(df["label"]) == {0, 1}
    assert df.loc[df["recommended"] == "yes", "label"].eq(1).all()
    assert {"text", "label"} <= set(df.columns)
    assert "Unnamed: 0" not in df.columns


def test_load_reviews_requires_columns(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        load_reviews(_write(tmp_path, pd.DataFrame({"rating": [1]})))


def test_split_is_stratified_disjoint_and_reproducible(tmp_path, tiny_reviews):
    df = load_reviews(_write(tmp_path, tiny_reviews))
    train, test = train_test_split_reviews(df, test_size=0.25, seed=1)
    again_train, again_test = train_test_split_reviews(df, test_size=0.25, seed=1)
    assert len(train) + len(test) == len(df)
    assert not set(train["content"]) & set(test["content"])
    assert abs(train["label"].mean() - test["label"].mean()) < 0.1
    assert train["content"].tolist() == again_train["content"].tolist()
    assert test["content"].tolist() == again_test["content"].tolist()


def test_subsample_noop_and_reduction(tiny_reviews):
    assert subsample(tiny_reviews, None) is tiny_reviews
    assert subsample(tiny_reviews, 1000) is tiny_reviews
    assert len(subsample(tiny_reviews, 10)) == 10


def test_committed_dataset_loads():
    """The shipped snapshot keeps its documented shape and binary labels."""
    df = load_reviews()
    assert len(df) > 15_000
    assert set(df["label"]) == {0, 1}
    assert df["rating"].between(1, 10).all()
