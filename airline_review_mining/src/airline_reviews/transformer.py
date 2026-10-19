"""Optional zero-shot transformer sentiment baseline (Hugging Face).

A DistilBERT model fine-tuned on SST-2 movie-review sentiment scores each review; the
probability of the POSITIVE class is used as "recommended" score. Nothing is trained
on the airline data, so it is a baseline for how far general-purpose sentiment gets.

Requires the ``transformer`` extra (``uv sync --extra transformer``) and downloads the
model (~270 MB) on first use. Runs on GPU when PyTorch sees one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def predict_positive_probability(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """P(positive) for every text, in the input order (long texts are truncated)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    positive_id = next(
        (i for i, label in model.config.id2label.items() if label.upper().startswith("POS")), 1
    )

    order = np.argsort([len(t) for t in texts])  # similar lengths per batch -> less padding
    probs = np.empty(len(texts), dtype=float)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            enc = tokenizer(
                [texts[i] for i in idx],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs[idx] = torch.softmax(logits, dim=-1)[:, positive_id].cpu().numpy()
    return probs
