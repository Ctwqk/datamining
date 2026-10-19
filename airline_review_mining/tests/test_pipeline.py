import json

import pandas as pd
import pytest

from airline_reviews import pipeline
from airline_reviews.absa import aspect_table, load_absa, top_aspects


def _csv(tmp_path, df):
    path = tmp_path / "reviews.csv"
    df.to_csv(path)
    return path


def test_preprocess_frame_adds_columns_and_uses_cache(tmp_path, tiny_reviews, nltk_ready):
    from airline_reviews.data import load_reviews

    df = load_reviews(_csv(tmp_path, tiny_reviews))
    cache = tmp_path / "cache"
    first = pipeline.preprocess_frame(df, cache)
    assert {"clean_header", "clean_content", "clean_text", "compound"} <= set(first.columns)
    assert len(list(cache.glob("features_*.csv"))) == 1
    second = pipeline.preprocess_frame(df, cache)  # served from the cache
    pd.testing.assert_frame_equal(first, second)
    assert first["compound"].iloc[0] > 0 > first["compound"].iloc[1]  # good vs bad review


def test_pipeline_smoke_end_to_end(tmp_path, tiny_reviews, nltk_ready):
    out = tmp_path / "out"
    rc = pipeline.main(
        ["run", "--data", str(_csv(tmp_path, tiny_reviews)), "--out", str(out), "--absa", "",
         "--no-cache", "--no-lstm", "--bootstrap", "20", "--jobs", "1"]
    )  # fmt: skip
    assert rc == 0
    metrics = json.loads((out / "metrics.json").read_text())
    names = {r["name"] for r in metrics["classification"]}
    assert {"majority", "vader + LR", "tfidf + LR (raw)"} <= names
    best = max(r["f1"] for r in metrics["classification"])
    majority = next(r for r in metrics["classification"] if r["name"] == "majority")
    assert best >= majority["f1"]
    assert metrics["meta"]["n_reviews"] == 40
    for fig in ("rating_distribution", "sentiment_vs_rating", "model_comparison", "top_terms"):
        assert (out / "figures" / f"{fig}.png").stat().st_size > 1000
    assert (out / "results.md").read_text().startswith("### Classification")


def test_pipeline_smoke_with_lstm(tmp_path, tiny_reviews, nltk_ready):
    pytest.importorskip("torch")
    out = tmp_path / "out"
    rc = pipeline.main(
        ["run", "--data", str(_csv(tmp_path, tiny_reviews)), "--out", str(out), "--absa", "",
         "--no-cache", "--epochs", "2", "--bootstrap", "20", "--jobs", "1"]
    )  # fmt: skip
    assert rc == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert any("LSTM" in r["name"] for r in metrics["classification"])
    assert (out / "figures" / "lstm_training_random_init.png").exists()


def test_absa_summary_on_committed_table():
    table = aspect_table(load_absa())
    top = top_aspects(table, 5)
    assert list(top.columns) == ["Negative", "Neutral", "Positive", "total"]
    assert (top["total"] == top[["Negative", "Neutral", "Positive"]].sum(axis=1)).all()
    assert top["total"].is_monotonic_decreasing
    assert {"food", "service"} <= set(top.index)
