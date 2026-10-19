import pandas as pd
import pytest

from airline_reviews.text import TextPreprocessor, ensure_nltk_data


@pytest.fixture(scope="session")
def nltk_ready():
    ensure_nltk_data()


@pytest.fixture(scope="session")
def preprocessor(nltk_ready):
    return TextPreprocessor()


@pytest.fixture()
def tiny_reviews() -> pd.DataFrame:
    """Hand-made table in the same shape as full_airline_review.csv."""
    rows = []
    for i in range(40):
        good = i % 2 == 0
        rows.append(
            {
                "rating": 9 if good else 2,
                "header": "great flight" if good else "terrible delay",
                "content": (
                    "The crew were friendly and the seat was comfortable. "
                    if good
                    else "Awful service, the flight was late and food was cold. "
                )
                + f"Trip {i}.",
                "airline": "emirates" if i % 4 < 2 else "qatar-airways",
                "recommended": "yes" if good else "no",
            }
        )
    return pd.DataFrame(rows)
