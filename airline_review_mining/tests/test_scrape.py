from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from airline_reviews import scrape

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "skytrax_sample_page.html"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    soup = BeautifulSoup(FIXTURE.read_text(encoding="utf8"), "html.parser")
    frame = pd.DataFrame(scrape.parse_page(soup))
    frame["airline"] = "test-air"
    return frame


def test_page_url():
    assert scrape.page_url("emirates", 1).endswith("/airline-reviews/emirates")
    assert scrape.page_url("emirates", 7).endswith("/airline-reviews/emirates/page/7/")


def test_parse_page_extracts_every_field(raw):
    assert len(raw) == 3
    first = raw.iloc[0]
    assert first["rating"] == "9"
    assert first["header"] == "Excellent service from start to finish"  # quotes stripped
    assert first["author"] == "A. Traveller"
    assert first["date"] == "2024-03-01"
    assert first["place"] == "Germany"
    assert first["content"].startswith("✅ Trip Verified |")
    assert first["collect"]["Aircraft"] == "A350"
    assert first["collect"]["Recommended"] == "yes"
    # star-style rows report the highest filled star
    assert first["collect"]["Seat Comfort"] == "4"
    assert first["collect"]["Food & Beverages"] == "3"


def test_parse_page_handles_missing_place(raw):
    assert pd.isna(raw.iloc[2]["place"])


def test_split_verification():
    assert scrape.split_verification("✅ Trip Verified | Nice flight") == (
        "✅ Trip Verified",
        "Nice flight",
    )
    assert scrape.split_verification("❎ Not Verified | ok") == ("❎ Not Verified", "ok")
    # a pipe inside the body is not a verification label: keep the whole text
    text = "Our flight | the one to Lisbon | was late"
    assert scrape.split_verification(text) == (None, text)
    assert scrape.split_verification("no pipes here") == (None, "no pipes here")


def test_split_verification_keeps_text_after_second_pipe():
    _, body = scrape.split_verification("Trip Verified | part one | part two")
    assert body == "part one | part two"


def test_clean_reviews_drops_incomplete_rows_and_aligns_details(raw):
    cleaned = scrape.clean_reviews(raw)
    # review 3 has no place -> dropped (whole-row dropna, as in the notebook)
    assert len(cleaned) == 2
    assert cleaned["rating"].tolist() == [9, 2]
    assert cleaned["recommended"].tolist() == ["yes", "no"]
    assert cleaned["aircraft"].tolist() == ["A350", "A320"]
    assert cleaned["content"].iloc[0].startswith("Boarding was quick")
    assert cleaned["content"].iloc[1].startswith("Our flight | the one to Lisbon")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_clean_reviews_alignment_survives_dropped_rows(raw):
    """Dropping a row must not shift the per-review details onto the wrong review."""
    shuffled = raw.iloc[[2, 1, 0]].copy()  # review 3 (will be dropped) first
    cleaned = scrape.clean_reviews(shuffled)
    assert dict(zip(cleaned["rating"], cleaned["recommended"], strict=True)) == {2: "no", 9: "yes"}


def test_finalize_fills_missing_subratings_with_minus_one(raw):
    final = scrape.finalize([scrape.clean_reviews(raw)])
    assert final["seat_comfort"].tolist() == [4, -1]
    assert final["ground_service"].tolist() == [-1, 1]
    assert "wifi_connectivity" not in final
