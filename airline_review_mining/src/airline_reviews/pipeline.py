"""End-to-end analysis: preprocess -> features -> models -> evaluation -> figures.

Usage (from the repository root)::

    uv run airline-reviews run                  # full run, ~minutes on CPU
    uv run airline-reviews run --sample 1500 --no-lstm --out /tmp/smoke   # smoke test
    uv run airline-reviews scrape --airlines emirates --max-pages 2 --out /tmp/emirates.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from airline_reviews import __version__, plots
from airline_reviews.absa import aspect_table, load_absa, top_aspects
from airline_reviews.config import (
    DEFAULT_ABSA_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_REVIEWS_CSV,
    PROCESSED_DIR,
    SEED,
)
from airline_reviews.data import load_reviews, subsample, train_test_split_reviews
from airline_reviews.evaluate import bootstrap_ci, classification_metrics, regression_metrics
from airline_reviews.features import VADER_COLUMNS, vader_scores
from airline_reviews.models import (
    classification_specs,
    fit_with_cv,
    positive_scores,
    regression_specs,
)
from airline_reviews.text import TextPreprocessor


# --------------------------------------------------------------------------- preprocessing
def preprocess_frame(df: pd.DataFrame, cache_dir: Path | None = PROCESSED_DIR) -> pd.DataFrame:
    """Add ``clean_header``, ``clean_content``, ``clean_text`` and the four VADER columns.

    Both steps are row-wise and need no fitting, so doing them before the train/test split
    cannot leak information. Results are cached under ``data/processed`` keyed by a hash
    of the input text.
    """
    digest = hashlib.md5("\x1f".join(df["header"] + "\x1e" + df["content"]).encode()).hexdigest()
    cache = cache_dir / f"features_{digest[:16]}.csv" if cache_dir else None
    cols = ["clean_header", "clean_content", *VADER_COLUMNS]
    if cache is not None and cache.exists():
        cached = pd.read_csv(cache, keep_default_na=False)
        if len(cached) == len(df):
            out = df.copy()
            for col in cols:
                out[col] = cached[col].to_numpy()
            out["clean_text"] = out["clean_header"] + " " + out["clean_content"]
            return out

    prep = TextPreprocessor()
    out = df.copy()
    out["clean_header"] = prep.transform(df["header"])
    out["clean_content"] = prep.transform(df["content"])
    scores = vader_scores(df["text"])
    for col in VADER_COLUMNS:
        out[col] = scores[col].to_numpy()
    out["clean_text"] = out["clean_header"] + " " + out["clean_content"]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out[cols].to_csv(cache, index=False)
    return out


def _inputs(df: pd.DataFrame, kind: str):
    if kind == "raw":
        return df["text"]
    if kind == "clean":
        return df["clean_text"]
    if kind == "vader":
        return df[VADER_COLUMNS].to_numpy()
    raise ValueError(kind)


# ------------------------------------------------------------------------------- tasks
def _f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def run_classification(
    train: pd.DataFrame, test: pd.DataFrame, jobs: int, seed: int, n_boot: int
) -> tuple[list[dict], dict]:
    rows, fitted = [], {}
    for spec in classification_specs(seed):
        t0 = time.perf_counter()
        model, params, cv_f1 = fit_with_cv(
            spec, _inputs(train, spec.source), train["label"], "f1", seed=seed, n_jobs=jobs
        )
        fit_s = time.perf_counter() - t0
        X_test = _inputs(test, spec.source)
        pred = model.predict(X_test)
        res = classification_metrics(test["label"], pred, positive_scores(model, X_test))
        res["f1_ci"] = bootstrap_ci(_f1, test["label"], pred, n_boot, seed=seed)
        res["accuracy_ci"] = bootstrap_ci(accuracy_score, test["label"], pred, n_boot, seed=seed)
        res.update(
            name=spec.name,
            description=spec.description,
            input=spec.source,
            best_params=params,
            cv_f1=cv_f1,
            fit_seconds=round(fit_s, 1),
        )
        rows.append(res)
        fitted[spec.name] = model
        print(f"[clf] {spec.name:22s} acc={res['accuracy']:.4f} f1={res['f1']:.4f} {fit_s:5.1f}s")
    return rows, fitted


def run_regression(train: pd.DataFrame, test: pd.DataFrame, jobs: int, seed: int):
    rows, preds = [], {}
    for spec in regression_specs(seed):
        model, params, cv_score = fit_with_cv(
            spec,
            _inputs(train, spec.source),
            train["rating"],
            "neg_mean_absolute_error",
            seed=seed,
            n_jobs=jobs,
        )
        pred = np.clip(model.predict(_inputs(test, spec.source)), 1, 10)
        res = regression_metrics(test["rating"], pred)
        res.update(
            name=spec.name,
            description=spec.description,
            best_params=params,
            cv_mae=None if cv_score is None else -cv_score,
        )
        rows.append(res)
        preds[spec.name] = pred
        print(f"[reg] {spec.name:22s} mae={res['mae']:.3f} rho={_fmt(res['spearman'])}")
    return rows, preds


def top_terms(model, n: int = 15):
    """Most positive / negative TF-IDF coefficients of a fitted tfidf+LR pipeline."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    neg = [(names[i], float(coef[i])) for i in order[:n]]
    pos = [(names[i], float(coef[i])) for i in order[::-1][:n]]
    return pos, neg


def run_lstm(train, test, args, figures: Path, glove: str | None):
    import torch

    from airline_reviews.lstm import train_and_evaluate

    torch.set_num_threads(max(1, args.jobs))
    res = train_and_evaluate(
        train["clean_content"].tolist(),
        train["clean_header"].tolist(),
        train["label"].tolist(),
        test["clean_content"].tolist(),
        test["clean_header"].tolist(),
        test["label"].tolist(),
        glove_path=glove,
        freeze_embeddings=args.freeze_embeddings,
        epochs=args.epochs,
        seed=args.seed,
    )
    suffix = "glove" if glove else "random_init"
    plots.plot_training_curves(res.history, res.best_epoch, figures / f"lstm_training_{suffix}.png")
    name = "two-channel LSTM + GloVe" if glove else "two-channel LSTM (random init)"
    row = dict(res.metrics)
    row.update(
        name=name,
        description="Original coursework architecture (review + header LSTMs), repaired training loop",
        input="clean",
        best_params={"best_epoch": res.best_epoch, "epochs_run": len(res.history["train_loss"])},
        cv_f1=None,
        glove_coverage=res.glove_coverage,
        n_parameters=res.n_parameters,
        device=res.device,
        f1_ci=bootstrap_ci(_f1, test["label"], res.y_pred, args.bootstrap, seed=args.seed),
        accuracy_ci=bootstrap_ci(
            accuracy_score, test["label"], res.y_pred, args.bootstrap, seed=args.seed
        ),
    )
    print(
        f"[lstm] {name}: acc={row['accuracy']:.4f} f1={row['f1']:.4f} (best epoch {res.best_epoch})"
    )
    return row


def run_transformer(test, args):
    from airline_reviews.transformer import predict_positive_probability

    t0 = time.perf_counter()
    prob = predict_positive_probability(test["text"].tolist(), model_name=args.transformer_model)
    pred = (prob > 0.5).astype(int)
    row = classification_metrics(test["label"], pred, prob)
    row.update(
        name="DistilBERT SST-2 (zero-shot)",
        description=f"{args.transformer_model}, no fine-tuning; P(positive) > 0.5 -> recommended",
        input="raw",
        best_params={},
        cv_f1=None,
        fit_seconds=round(time.perf_counter() - t0, 1),
        f1_ci=bootstrap_ci(_f1, test["label"], pred, 1000, seed=args.seed),
        accuracy_ci=bootstrap_ci(accuracy_score, test["label"], pred, 1000, seed=args.seed),
        spearman_with_rating=float(spearmanr(prob, test["rating"]).statistic),
    )
    print(f"[transformer] acc={row['accuracy']:.4f} f1={row['f1']:.4f}")
    return row


# --------------------------------------------------------------------------------- main
def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = subsample(load_reviews(args.data), args.sample, args.seed)
    print(f"loaded {len(df):,} reviews from {args.data}")
    df = preprocess_frame(df, None if args.no_cache else PROCESSED_DIR)
    train, test = train_test_split_reviews(df, test_size=0.2, seed=args.seed)
    print(
        f"train={len(train):,} test={len(test):,}  (test positive rate {test['label'].mean():.3f})"
    )

    # --- exploratory figures (no model fitting involved)
    rho = float(spearmanr(df["compound"], df["rating"]).statistic)
    plots.plot_rating_distribution(df, figures / "rating_distribution.png")
    plots.plot_sentiment_vs_rating(
        df["rating"], df["compound"], rho, figures / "sentiment_vs_rating.png"
    )
    eda = {
        "mean_rating": float(df["rating"].mean()),
        "recommended_rate": float(df["label"].mean()),
        "vader_compound_vs_rating_spearman": rho,
        "median_words_per_review": float(df["content"].str.split().str.len().median()),
        "reviews_per_airline": df["airline"].value_counts().to_dict(),
        "vader_mean_compound_recommended": float(df.loc[df.label == 1, "compound"].mean()),
        "vader_mean_compound_not_recommended": float(df.loc[df.label == 0, "compound"].mean()),
    }
    if args.absa and Path(args.absa).exists() and args.sample is None:
        tbl = top_aspects(aspect_table(load_absa(args.absa)), 15)
        plots.plot_aspects(tbl, figures / "negative_review_aspects.png")
        eda["top_negative_review_aspects"] = tbl["total"].to_dict()

    # --- classification
    clf_rows, fitted = run_classification(train, test, args.jobs, args.seed, args.bootstrap)
    if args.lstm:
        clf_rows.append(run_lstm(train, test, args, figures, glove=None))
        if args.glove:
            clf_rows.append(run_lstm(train, test, args, figures, glove=args.glove))
    if args.transformer:
        clf_rows.append(run_transformer(test, args))

    # Keep diagnostic models fixed; the test ranking is descriptive, not model selection.
    diagnostic_clf = next(r for r in clf_rows if r["name"] == "tfidf + LR (raw)")
    plots.plot_model_comparison(clf_rows, figures / "model_comparison.png")
    plots.plot_confusion(
        diagnostic_clf["confusion_matrix"],
        figures / "confusion_matrix.png",
        f"{diagnostic_clf['name']}\nacc {diagnostic_clf['accuracy']:.3f}, F1 {diagnostic_clf['f1']:.3f}",
    )
    lr_model = fitted["tfidf + LR (raw)"]
    pos, neg = top_terms(lr_model)
    plots.plot_top_terms(pos, neg, figures / "top_terms.png")

    # --- rating regression
    reg_rows, reg_preds = run_regression(train, test, args.jobs, args.seed)
    diagnostic_reg = next(r for r in reg_rows if r["name"] == "tfidf + Ridge (raw)")
    plots.plot_rating_regression(
        test["rating"],
        reg_preds[diagnostic_reg["name"]],
        diagnostic_reg,
        figures / "rating_regression.png",
    )

    metrics = {
        "meta": {
            "version": __version__,
            "seed": args.seed,
            "sample": args.sample,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "n_reviews": len(df),
            "n_train": len(train),
            "n_test": len(test),
            "test_positive_rate": float(test["label"].mean()),
        },
        "eda": eda,
        "classification": clf_rows,
        "top_terms": {"recommended": pos, "not_recommended": neg},
        "regression": reg_rows,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n")
    _write_markdown_tables(out / "results.md", clf_rows, reg_rows)
    print(
        f"wrote {out / 'metrics.json'} and {len(list(figures.glob('*.png')))} figures to {figures}"
    )
    return 0


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _fmt(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _write_markdown_tables(path: Path, clf_rows: list[dict], reg_rows: list[dict]) -> None:
    lines = [
        "### Classification: is the airline recommended? (held-out test set)",
        "",
        "| Model | Accuracy | F1 | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for r in sorted(clf_rows, key=lambda r: -r["f1"]):
        lines.append(
            f"| {r['name']} | {_fmt(r['accuracy'])} | {_fmt(r['f1'])} | {_fmt(r.get('roc_auc'))} |"
        )
    lines += [
        "",
        "### Regression: overall rating 1-10 (held-out test set)",
        "",
        "| Model | MAE | RMSE | R2 | Spearman |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(reg_rows, key=lambda r: r["mae"]):
        lines.append(
            f"| {r['name']} | {_fmt(r['mae'])} | {_fmt(r['rmse'])} | {_fmt(r['r2'])} | {_fmt(r['spearman'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def scrape_cmd(args: argparse.Namespace) -> int:
    from airline_reviews.scrape import scrape_airlines

    df = scrape_airlines(args.airlines, max_pages=args.max_pages, delay=args.delay)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} reviews to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="airline-reviews", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full analysis pipeline")
    r.add_argument("--data", default=str(DEFAULT_REVIEWS_CSV), help="reviews CSV")
    r.add_argument("--absa", default=str(DEFAULT_ABSA_CSV), help="aspect-sentiment CSV (optional)")
    r.add_argument("--out", default=str(DEFAULT_RESULTS_DIR), help="output directory")
    r.add_argument("--sample", type=int, default=None, help="use a random subsample (smoke test)")
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--jobs", type=int, default=4, help="parallel jobs for cross-validation")
    r.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs")
    r.add_argument("--no-cache", action="store_true", help="do not read/write data/processed")
    r.add_argument(
        "--lstm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train the two-channel LSTM (needs the 'deep' extra)",
    )
    r.add_argument("--epochs", type=int, default=15, help="max LSTM epochs")
    r.add_argument("--glove", default=None, help="GloVe .txt/.zip for LSTM embeddings (optional)")
    r.add_argument("--freeze-embeddings", action="store_true", help="freeze GloVe vectors")
    r.add_argument(
        "--transformer",
        action="store_true",
        help="also score a zero-shot transformer sentiment baseline (needs the 'transformer' extra)",
    )
    r.add_argument(
        "--transformer-model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    r.set_defaults(func=run)

    s = sub.add_parser("scrape", help="scrape fresh reviews from airlinequality.com")
    s.add_argument(
        "--airlines", nargs="+", required=True, help="slugs, e.g. emirates qatar-airways"
    )
    s.add_argument(
        "--max-pages", type=int, default=3, help="review pages per airline (10 reviews each)"
    )
    s.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    s.add_argument("--out", required=True, help="output CSV")
    s.set_defaults(func=scrape_cmd)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
