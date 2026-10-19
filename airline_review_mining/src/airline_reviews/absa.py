"""Summarise the aspect-based sentiment table for negative reviews.

``data/absa_negative_reviews.csv`` holds one row per not-recommended review with the
output of an aspect-term-extraction + polarity model (``aspect``, ``sentiment``,
``confidence`` lists are aligned per extracted aspect). The code that produced the file
is not part of this repository, so it is treated as a given input.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from airline_reviews.config import DEFAULT_ABSA_CSV

LIST_COLUMNS = ["IOB", "tokens", "aspect", "position", "sentiment", "probs", "confidence"]


def load_absa(path: str | Path = DEFAULT_ABSA_CSV) -> pd.DataFrame:
    """Read the CSV and parse the stringified list columns."""
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df


def aspect_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (review, aspect) with its polarity and confidence."""
    rows = [
        {"review": i, "aspect": a, "sentiment": s, "confidence": c}
        for i, (aspects, sents, confs) in enumerate(
            zip(df["aspect"], df["sentiment"], df["confidence"], strict=True)
        )
        for a, s, c in zip(aspects, sents, confs, strict=True)
    ]
    return pd.DataFrame(rows, columns=["review", "aspect", "sentiment", "confidence"])


def top_aspects(table: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Most frequently mentioned aspects with their sentiment breakdown (counts)."""
    counts = table.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    for col in ("Negative", "Neutral", "Positive"):
        if col not in counts:
            counts[col] = 0
    counts["total"] = counts[["Negative", "Neutral", "Positive"]].sum(axis=1)
    return counts.sort_values("total", ascending=False).head(n)[
        ["Negative", "Neutral", "Positive", "total"]
    ]
