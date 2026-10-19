"""Figures for the airline-review analysis (all saved as PNG; headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PALETTE = {"yes": "#2a9d8f", "no": "#e76f51"}
DPI = 130


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rating_distribution(df: pd.DataFrame, path: str | Path) -> Path:
    """Rating histogram split by 'recommended', and mean rating per airline."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    counts = df.groupby(["rating", "recommended"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(1, 11), fill_value=0)
    bottom = np.zeros(len(counts))
    for label in ("no", "yes"):
        if label in counts:
            axes[0].bar(
                counts.index,
                counts[label],
                bottom=bottom,
                color=PALETTE[label],
                label=f"recommended: {label}",
            )
            bottom += counts[label].to_numpy()
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlabel("Overall rating (1-10)")
    axes[0].set_ylabel("Reviews")
    axes[0].set_title(f"Rating distribution (n = {len(df):,})")
    axes[0].legend()

    per_airline = (
        df.groupby("airline")["rating"].agg(["mean", "count"]).sort_values("mean", ascending=True)
    )
    labels = [f"{a} (n={n})" for a, n in zip(per_airline.index, per_airline["count"], strict=True)]
    axes[1].barh(labels, per_airline["mean"], color="#457b9d")
    axes[1].axvline(df["rating"].mean(), color="k", ls="--", lw=1, label="all airlines")
    axes[1].set_xlabel("Mean rating")
    axes[1].set_title("Mean rating by airline")
    axes[1].set_xlim(0, 10)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_sentiment_vs_rating(
    ratings: pd.Series, compound: pd.Series, spearman: float, path: str | Path
) -> Path:
    """Distribution of the VADER compound score for each star rating."""
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    frame = pd.DataFrame({"rating": ratings.to_numpy(), "compound": compound.to_numpy()})
    sns.boxplot(
        data=frame, x="rating", y="compound", color="#a8dadc", fliersize=1.5, linewidth=1, ax=ax
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Overall rating (1-10)")
    ax.set_ylabel("VADER compound score")
    ax.set_title(f"Lexicon sentiment vs. rating (Spearman rho = {spearman:.2f})")
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(rows: list[dict], path: str | Path) -> Path:
    """Test-set F1 and accuracy (with 95% bootstrap CIs) for every classifier."""
    _style()
    rows = sorted(rows, key=lambda r: r["f1"])
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.8))
    for offset, key, color, label in (
        (-0.2, "f1", "#264653", "F1"),
        (0.2, "accuracy", "#e9c46a", "Accuracy"),
    ):
        vals = np.array([r[key] for r in rows])
        err = None
        if all(f"{key}_ci" in r for r in rows):
            ci = np.array([r[f"{key}_ci"] for r in rows])
            err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        ax.barh(y + offset, vals, height=0.38, xerr=err, color=color, label=label, capsize=2)
        right = vals if err is None else ci[:, 1]
        for yi, v, x in zip(y + offset, vals, right, strict=True):
            ax.text(x + 0.006, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Test-set score (bars: 95% bootstrap CI)")
    ax.set_title("Predicting 'recommended' from review text")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm, path: str | Path, title: str) -> Path:
    _style()
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["not rec.", "recommended"],
        yticklabels=["not rec.", "recommended"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_top_terms(
    positive: list[tuple[str, float]], negative: list[tuple[str, float]], path: str | Path
) -> Path:
    """Most influential logistic-regression coefficients in each direction."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, terms, color, title in (
        (axes[0], negative, PALETTE["no"], "Pushes towards 'not recommended'"),
        (axes[1], positive, PALETTE["yes"], "Pushes towards 'recommended'"),
    ):
        names = [t for t, _ in terms][::-1]
        weights = [abs(w) for _, w in terms][::-1]
        ax.barh(names, weights, color=color)
        ax.set_title(title)
        ax.set_xlabel("|coefficient|")
    fig.tight_layout()
    return _save(fig, path)


def plot_rating_regression(y_true, y_pred, metrics: dict, path: str | Path) -> Path:
    """Predicted rating distribution for each true rating (test set)."""
    _style()
    frame = pd.DataFrame({"true": np.asarray(y_true), "pred": np.asarray(y_pred)})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=frame, x="true", y="pred", color="#bde0fe", fliersize=1.5, ax=ax)
    ax.plot(range(10), range(1, 11), color="k", ls="--", lw=1, label="perfect")
    ax.set_xlabel("True rating")
    ax.set_ylabel("Predicted rating")
    ax.set_title(f"Rating regression: MAE {metrics['mae']:.2f}, Spearman {metrics['spearman']:.2f}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_aspects(table: pd.DataFrame, path: str | Path) -> Path:
    """Most frequent aspects extracted from not-recommended reviews, by polarity."""
    _style()
    table = table.sort_values("total")
    fig, ax = plt.subplots(figsize=(8, 5.4))
    left = np.zeros(len(table))
    for col, color in (("Negative", "#e76f51"), ("Neutral", "#adb5bd"), ("Positive", "#2a9d8f")):
        ax.barh(table.index, table[col], left=left, color=color, label=col)
        left += table[col].to_numpy()
    ax.set_xlabel("Mentions in not-recommended reviews")
    ax.set_title("Most mentioned aspects and their polarity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curves(
    history: dict[str, list[float]], best_epoch: int, path: str | Path
) -> Path:
    _style()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, history["train_loss"], label="train", color="#264653")
    axes[0].plot(epochs, history["val_loss"], label="validation", color="#e76f51")
    axes[0].axvline(best_epoch, color="k", ls=":", lw=1, label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], color="#2a9d8f")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    fig.suptitle("Two-channel LSTM training")
    fig.tight_layout()
    return _save(fig, path)
