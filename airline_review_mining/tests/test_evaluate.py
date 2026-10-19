import numpy as np
import pytest
from sklearn.metrics import f1_score

from airline_reviews.evaluate import bootstrap_ci, classification_metrics, regression_metrics


def test_classification_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    pred = [1, 1, 0, 0, 0, 1, 0, 1]  # TP=3 FN=1 FP=1 TN=3
    m = classification_metrics(y, pred, y_score=[0.9, 0.8, 0.4, 0.2, 0.1, 0.6, 0.3, 0.7])
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert 0.5 < m["roc_auc"] <= 1.0


def test_classification_metrics_without_scores_has_no_auc():
    assert "roc_auc" not in classification_metrics([0, 1], [0, 1])


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    pred = np.where(rng.random(400) < 0.85, y, 1 - y)
    point = f1_score(y, pred)
    lo, hi = bootstrap_ci(f1_score, y, pred, n_boot=300, seed=3)
    assert lo < point < hi and hi - lo < 0.15
    assert (lo, hi) == bootstrap_ci(f1_score, y, pred, n_boot=300, seed=3)


def test_regression_metrics_perfect_and_constant():
    y = [1, 3, 5, 7, 10]
    perfect = regression_metrics(y, y)
    assert perfect["mae"] == 0 and perfect["rmse"] == 0 and perfect["r2"] == 1
    assert perfect["spearman"] == pytest.approx(1.0) and perfect["within_1"] == 1.0
    const = regression_metrics(y, [5] * 5)
    assert const["spearman"] is None  # undefined for a constant predictor
    assert const["mae"] == pytest.approx(np.mean(np.abs(np.array(y) - 5)))
