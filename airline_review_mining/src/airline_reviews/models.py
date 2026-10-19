"""Model zoo: scikit-learn baselines for 'recommended?' (classification) and 1-10 rating."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from airline_reviews.config import SEED
from airline_reviews.features import make_tfidf

# "raw" / "clean": a text column; "vader": the 4 precomputed VADER scores per review.
InputKind = Literal["raw", "clean", "vader"]

_CLF_SCORING = {"f1", "accuracy", "roc_auc", "balanced_accuracy"}


@dataclass
class ModelSpec:
    """A named estimator, which input it reads, and its (small) tuning grid."""

    name: str
    description: str
    source: InputKind
    estimator: BaseEstimator
    grid: dict[str, list[Any]] = field(default_factory=dict)


def classification_specs(seed: int = SEED) -> list[ModelSpec]:
    """Models that predict the binary 'recommended' flag from the review text."""
    return [
        ModelSpec(
            "majority",
            "Always predicts the most frequent class",
            "raw",
            DummyClassifier(strategy="most_frequent"),
        ),
        ModelSpec(
            "vader + LR",
            "VADER polarity scores (neg/neu/pos/compound) -> logistic regression",
            "vader",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            ),
        ),
        ModelSpec(
            "tfidf + NB (raw)",
            "TF-IDF 1-2 grams on the raw text -> Complement Naive Bayes",
            "raw",
            Pipeline([("tfidf", make_tfidf()), ("clf", ComplementNB())]),
            {"clf__alpha": [0.1, 0.3, 1.0]},
        ),
        ModelSpec(
            "tfidf + LR (clean)",
            "TF-IDF on the original lemmatised / stop-word-free text -> logistic regression",
            "clean",
            Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
            {"clf__C": [0.5, 2.0, 8.0, 32.0]},
        ),
        ModelSpec(
            "tfidf + LR (raw)",
            "TF-IDF 1-2 grams on the raw text -> logistic regression",
            "raw",
            Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
            {"clf__C": [0.5, 2.0, 8.0, 32.0]},
        ),
        ModelSpec(
            "tfidf + SVM (raw)",
            "TF-IDF 1-2 grams on the raw text -> linear SVM",
            "raw",
            Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("clf", LinearSVC(random_state=seed, dual="auto")),
                ]
            ),
            {"clf__C": [0.1, 0.3, 1.0]},
        ),
    ]


def regression_specs(seed: int = SEED) -> list[ModelSpec]:
    """Models that predict the 1-10 overall rating from the review text."""
    return [
        ModelSpec("mean rating", "Predicts the training-set mean", "raw", DummyRegressor()),
        ModelSpec(
            "vader + Ridge",
            "VADER polarity scores -> ridge regression",
            "vader",
            Pipeline([("scale", StandardScaler()), ("reg", Ridge())]),
        ),
        ModelSpec(
            "tfidf + Ridge (raw)",
            "TF-IDF 1-2 grams on the raw text -> ridge regression",
            "raw",
            Pipeline([("tfidf", make_tfidf()), ("reg", Ridge(random_state=seed))]),
            {"reg__alpha": [0.3, 1.0, 3.0]},
        ),
    ]


def fit_with_cv(
    spec: ModelSpec, X_train, y_train, scoring: str, cv: int = 5, seed: int = SEED, n_jobs: int = 4
):
    """Fit ``spec`` on the training data, tuning its grid with CV *inside* the training set.

    Returns ``(fitted_estimator, best_params, cv_score)``; ``cv_score`` is None when the
    spec has no grid.
    """
    if not spec.grid:
        est = clone(spec.estimator).fit(X_train, y_train)
        return est, {}, None
    folds = StratifiedKFold(cv, shuffle=True, random_state=seed) if scoring in _CLF_SCORING else cv
    search = GridSearchCV(
        spec.estimator, spec.grid, scoring=scoring, cv=folds, n_jobs=n_jobs, refit=True
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def positive_scores(model, X) -> np.ndarray:
    """Continuous score for the positive class (probability if available, else margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
