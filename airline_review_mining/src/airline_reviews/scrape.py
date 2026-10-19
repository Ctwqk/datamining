"""Scraper and cleaner for airline reviews on https://www.airlinequality.com (Skytrax).

This is the notebook ``01_scrape_and_clean`` turned into importable, testable functions.
The committed dataset ``data/full_airline_review.csv`` was produced with the original
notebook code in April 2024; the site layout may have changed since, so treat live
scraping as best effort. Parsing and cleaning are covered by offline tests against a
synthetic page that mimics the page structure the original selectors expected.

Behaviour kept from the notebook: the same CSS selectors, whole-row ``dropna`` before
cleaning, ``-1`` for missing sub-ratings, and the ``Trip Verified |`` prefix split.
Changes: HTTP timeouts and retries, a polite delay between requests, a ``max_pages``
argument instead of the hard-coded page 367, stopping at the first empty page, a safe
index alignment of the per-review rating table, and a general rule (instead of two
hand-patched row ids) for reviews whose text has no verification prefix.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.airlinequality.com"
USER_AGENT = (
    "Mozilla/5.0 (compatible; datamining-coursework/0.2; +https://github.com/Ctwqk/datamining)"
)
ARTICLE_CLASS = re.compile(
    r"comp comp_media-review-rated list-item media position-content review-[0-9]*"
)
PLACE_RE = re.compile(r".*(\(.*\)).*")

DEFAULT_AIRLINES = [
    "singapore-airlines",
    "qatar-airways",
    "emirates",
    "japan-airlines",
    "southwest-airlines",
    "air-france",
    "korean-air",
    "british-airways",
    "asiana-airlines",
    "airasia",
]

# Column names of the per-review rating table, as in the notebook.
RENAME = {
    "Aircraft": "aircraft",
    "Type Of Traveller": "traveller_type",
    "Seat Type": "seat_type",
    "Route": "route",
    "Date Flown": "date_flown",
    "Seat Comfort": "seat_comfort",
    "Cabin Staff Service": "cabin_staff_service",
    "Food & Beverages": "food_beverages",
    "Ground Service": "ground_service",
    "Value For Money": "value_for_money",
    "Recommended": "recommended",
    "Wifi & Connectivity": "wifi_connectivity",
    "Inflight Entertainment": "entertainment",
}
SUB_RATINGS = [
    "seat_comfort",
    "cabin_staff_service",
    "food_beverages",
    "ground_service",
    "value_for_money",
    "entertainment",
]

_VERIFICATION_RE = re.compile(
    r"^\W*(trip\s+verified|verified\s+review|not\s+verified|unverified)\W*$", re.I
)


def page_url(airline: str, page: int) -> str:
    """URL of review page ``page`` (1-based) for an airline slug."""
    if page <= 1:
        return f"{BASE_URL}/airline-reviews/{airline}"
    return f"{BASE_URL}/airline-reviews/{airline}/page/{page}/"


def _text(node: Tag | None) -> str | None:
    return None if node is None else node.text


def parse_review(article: Tag) -> dict:
    """Extract one review ``<article>`` into a flat dict (missing parts become None)."""
    body = article.find("div", class_="body")
    rating_div = article.find("div", class_="rating-10")
    rating = _text(rating_div.find("span")) if rating_div else None

    header = body.find("h2") if body else None
    author_outer = body.find("span", itemprop="author") if body else None
    author = _text(author_outer.find("span")) if author_outer else None
    time_tag = body.find("time", itemprop="datePublished") if body else None
    h3 = body.find("h3") if body else None
    place_match = PLACE_RE.search(h3.text.strip()) if h3 else None
    content = article.find("div", class_="text_content")

    details: dict[str, str | None] = {}
    table = body.find("table", class_="review-ratings") if body else None
    for row in table.find_all("tr") if table else []:
        key_cell = row.find("td", class_=re.compile("review-rating-header.*"))
        if key_cell is None:
            continue
        value_cell = row.find("td", class_="review-value")
        if value_cell is not None:
            value = value_cell.text
        else:
            stars_cell = row.find("td", class_=re.compile("review-rating-stars.*"))
            filled = (
                [s.text for s in stars_cell.find_all("span", class_="star fill")]
                if stars_cell
                else []
            )
            value = max(filled) if filled else None
        details[key_cell.text] = value

    return {
        "rating": rating,
        "header": header.text.strip('"') if header else None,  # the site wraps titles in quotes
        "author": author,
        "date": time_tag["datetime"]
        if time_tag is not None and time_tag.has_attr("datetime")
        else None,
        "place": place_match.group(1).strip("()").strip() if place_match else None,
        "content": content.text.strip() if content else None,
        "collect": details,
    }


def parse_page(soup: BeautifulSoup) -> list[dict]:
    """All reviews on one result page."""
    return [parse_review(a) for a in soup.find_all("article", attrs={"class": ARTICLE_CLASS})]


def fetch(
    url: str, session: requests.Session, retries: int = 3, timeout: float = 30.0
) -> BeautifulSoup | None:
    """GET a page and parse it; ``None`` for 404 / repeated failures."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.ok:
            return BeautifulSoup(resp.content, "html.parser")
        time.sleep(2**attempt)
    return None


def scrape_airline(
    airline: str,
    max_pages: int = 367,
    delay: float = 1.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Collect up to ``max_pages`` result pages (10 reviews each) for one airline slug."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    records: list[dict] = []
    for page in range(1, max_pages + 1):
        soup = fetch(page_url(airline, page), session)
        reviews = parse_page(soup) if soup is not None else []
        if not reviews:
            break
        records.extend(reviews)
        time.sleep(delay)
    frame = pd.DataFrame(
        records, columns=["rating", "header", "author", "date", "place", "content", "collect"]
    )
    frame["airline"] = airline
    return frame


def split_verification(content: str) -> tuple[str | None, str]:
    """Split ``'✅ Trip Verified | text'`` into ``('✅ Trip Verified', 'text')``.

    If the part before the first ``|`` is not a verification label (some reviews contain a
    pipe in their body, or no label at all), the whole string is the review text.
    """
    if "|" in content:
        head, tail = content.split("|", 1)
        if _VERIFICATION_RE.match(head):
            return head.strip(), tail.strip()
    return None, content


def clean_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn one airline's raw scrape into the analysis table (notebook ``clean_data``)."""
    df = raw.dropna().copy()
    df["rating"] = df["rating"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    # index=df.index keeps the rating table aligned with its review after dropna
    details = pd.DataFrame(list(df["collect"]), index=df.index).rename(columns=RENAME)
    df = df.drop(columns=["collect"]).join(details)
    verified, text = (
        zip(*df["content"].map(split_verification), strict=True) if len(df) else ((), ())
    )
    df["trip_verified"] = list(verified)
    df["content"] = list(text)
    return df


def finalize(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate cleaned airlines and apply the notebook's column fixes."""
    df = pd.concat(list(frames), axis=0)
    df = df.drop(columns=["wifi_connectivity"], errors="ignore")
    if "date_flown" in df:
        df["date_flown"] = pd.to_datetime(df["date_flown"], errors="coerce")
    for col in SUB_RATINGS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["trip_verified"] = df["trip_verified"].fillna("Trip Verified")
    return df.reset_index(drop=True)


def scrape_airlines(
    airlines: Iterable[str], max_pages: int = 3, delay: float = 1.0
) -> pd.DataFrame:
    """Scrape + clean several airlines and return the columns used downstream."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    frames = []
    for airline in airlines:
        raw = scrape_airline(airline, max_pages=max_pages, delay=delay, session=session)
        print(f"{airline}: {len(raw)} reviews scraped")
        if len(raw):
            frames.append(clean_reviews(raw))
    if not frames:
        raise RuntimeError("no reviews could be scraped (is the site reachable / unchanged?)")
    final = finalize(frames)
    cols = ["rating", "header", "content", "airline", "recommended"]
    return final[[c for c in cols if c in final]]
