"""Default paths and small helpers shared by the package.

Everything is relative to the ``airline_review_mining`` project folder (or to
``$AIRLINE_REVIEWS_HOME`` when set); nothing is hard-coded to a machine.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(
    os.environ.get("AIRLINE_REVIEWS_HOME", Path(__file__).resolve().parents[2])
).resolve()
REPO_DIR = PROJECT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
DEFAULT_REVIEWS_CSV = DATA_DIR / "full_airline_review.csv"
DEFAULT_ABSA_CSV = DATA_DIR / "absa_negative_reviews.csv"
PROCESSED_DIR = DATA_DIR / "processed"  # git-ignored cache
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"

# NLTK corpora are downloaded here unless $NLTK_DATA is set (git-ignored).
NLTK_DATA_DIR = Path(
    os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or REPO_DIR / ".nltk_data"
).expanduser()

SEED = 42
