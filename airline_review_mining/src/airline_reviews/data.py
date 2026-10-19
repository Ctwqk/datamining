"""Loading, labelling and splitting the airline-review table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from airline_reviews.config import DEFAULT_REVIEWS_CSV, SEED

REQUIRED_COLUMNS = ["rating", "header", "content", "airline", "recommended"]


def load_reviews(path: str | Path = DEFAULT_REVIEWS_CSV, dedupe: bool = True) -> pd.DataFrame:
    """Read the review CSV and return a clean frame.

    Adds ``label`` (1 = recommended, 0 = not) and ``text`` (header + content, raw).
    Exact duplicate reviews (same airline, header and content) are dropped so a review
    cannot end up in both the train and the test split.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["header"] = df["header"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df = df[df["content"] != ""]
    if dedupe:
        df = df.drop_duplicates(subset=["airline", "header", "content"])
    df["rating"] = df["rating"].astype(int)
    df["label"] = encode_recommended(df["recommended"])
    df["text"] = join_header_content(df["header"], df["content"])
    return df.reset_index(drop=True)


def encode_recommended(values: pd.Series) -> pd.Series:
    """'yes' -> 1, 'no' -> 0 (case/whitespace-insensitive); anything else is an error."""
    norm = values.astype(str).str.strip().str.lower()
    mapped = norm.map({"yes": 1, "no": 0})
    if mapped.isna().any():
        bad = sorted(norm[mapped.isna()].unique())
        raise ValueError(f"unexpected 'recommended' values: {bad}")
    return mapped.astype(int)


def join_header_content(header: pd.Series, content: pd.Series) -> pd.Series:
    """Single document per review: ``header. content``."""
    return (header.str.rstrip(" .") + ". " + content).str.strip(" .")


def train_test_split_reviews(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (label x rating-band) split; the test set is touched only for final scores."""
    strata = df["label"].astype(str) + "_" + (df["rating"] // 4).astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, df["label"].astype(str))
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def subsample(df: pd.DataFrame, n: int | None, seed: int = SEED) -> pd.DataFrame:
    """Random subsample used for fast smoke runs (no-op when ``n`` is None or too large)."""
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
