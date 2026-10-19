"""Two-channel LSTM 'recommended?' classifier (the original ``model2.py``, repaired).

Architecture, unchanged from the coursework ``TextModel1``: the review body goes through
``LSTM(emb->50) -> LSTM(50->10)``, the header through ``LSTM(emb->10)``; the two final
hidden states are concatenated and fed to ``Linear(20->5) -> ReLU -> Linear(5->1)``.

What changed relative to the original script (see README, "Design notes"):

* no ``torchtext`` (unmaintained, no wheels for current PyTorch): a tiny vocabulary class
  replaces it, with a real padding index (originally ``<unk>`` and padding were both 0);
* the training loop trains on the *training* loader (the original iterated the validation
  loader), validates once per epoch, and keeps the best-validation checkpoint;
* the test set is only used once, at the end; vocabulary comes from the training split only;
* optional GloVe vectors are loaded straight from ``glove*.txt`` / ``.zip`` keeping only
  vocabulary words (the original materialised all 2.2M vectors in a dict). Setting
  ``module.requires_grad = False`` on the embedding module, as the original did, does not
  freeze its weights, so freezing is now done on the parameter;
* probabilities use a sigmoid (the original thresholded raw logits at 0.5).

Requires PyTorch (``uv sync --extra deep``).
"""

from __future__ import annotations

import copy
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from airline_reviews.config import SEED
from airline_reviews.evaluate import classification_metrics

PAD, UNK = 0, 1


class Vocab:
    """Word -> index mapping with ``<pad>=0`` and ``<unk>=1``."""

    def __init__(self, itos: Sequence[str]) -> None:
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> Vocab:
        counts = Counter(tok for text in texts for tok in text.split())
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls(["<pad>", "<unk>", *words])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(tok, UNK) for tok in text.split()]
        return ids or [UNK]  # an LSTM needs at least one step


def load_glove(path: str | Path, vocab: Vocab, dim: int = 300, seed: int = SEED):
    """Embedding matrix ``(len(vocab), dim)`` from a GloVe text file or a zip containing one.

    Only words present in ``vocab`` are kept in memory. Words missing from GloVe get small
    random vectors; the pad row is zero. Returns ``(matrix, coverage)``, where coverage is
    the fraction of vocabulary words that were found.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype("float32")
    matrix[PAD] = 0.0
    found: set[str] = set()

    def scan(lines: Iterable[str]) -> None:
        for line in lines:
            word, _, rest = line.partition(" ")  # vocabulary words never contain spaces
            idx = vocab.stoi.get(word)
            if idx is None or word in found:
                continue
            values = rest.split()
            if len(values) != dim:  # malformed line, or a multi-word GloVe token
                continue
            try:
                matrix[idx] = np.asarray(values, dtype="float32")
            except ValueError:
                continue
            found.add(word)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".txt"))
            with zf.open(member) as raw:
                scan(line.decode("utf8", errors="ignore") for line in raw)
    else:
        with open(path, encoding="utf8", errors="ignore") as fh:
            scan(fh)
    coverage = len(found) / max(1, len(vocab) - 2)  # exclude <pad>, <unk>
    return matrix, coverage


class ReviewDataset(Dataset):
    """Pairs of token-id sequences (review, header) and a binary label."""

    def __init__(self, reviews: list[list[int]], headers: list[list[int]], labels) -> None:
        if not (len(reviews) == len(headers) == len(labels)):
            raise ValueError("reviews, headers and labels must have the same length")
        self.reviews, self.headers = reviews, headers
        self.labels = np.asarray(labels, dtype="float32")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.reviews[i], dtype=torch.long),
            torch.tensor(self.headers[i], dtype=torch.long),
            torch.tensor(self.labels[i]),
        )


def collate(batch):
    """Pad each batch to its longest sequence; returns ids, lengths and labels."""
    reviews, headers, labels = zip(*batch, strict=True)
    len_r = torch.tensor([len(x) for x in reviews])
    len_h = torch.tensor([len(x) for x in headers])
    return (
        pad_sequence(reviews, batch_first=True, padding_value=PAD),
        pad_sequence(headers, batch_first=True, padding_value=PAD),
        len_r,
        len_h,
        torch.stack(labels),
    )


class DualLSTM(nn.Module):
    """Review + header LSTM classifier producing one logit per example."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
        self.embedding.weight.requires_grad_(not freeze_embeddings)
        self.lstm_review_1 = nn.LSTM(embedding_dim, 50, batch_first=True)
        self.lstm_review_2 = nn.LSTM(50, 10, batch_first=True)
        self.lstm_header = nn.LSTM(embedding_dim, 10, batch_first=True)
        self.dropout_in = nn.Dropout(dropout)
        self.fc1 = nn.Linear(20, 5)
        self.dropout_mid = nn.Dropout(dropout)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, reviews, headers, len_reviews, len_headers):
        r = pack_padded_sequence(
            self.embedding(reviews), len_reviews.cpu(), batch_first=True, enforce_sorted=False
        )
        h = pack_padded_sequence(
            self.embedding(headers), len_headers.cpu(), batch_first=True, enforce_sorted=False
        )
        out1, _ = self.lstm_review_1(r)
        _, (hid_review, _) = self.lstm_review_2(out1)
        _, (hid_header, _) = self.lstm_header(h)
        x = torch.cat((hid_review[-1], hid_header[-1]), dim=1)
        x = self.fc1(self.dropout_in(x))
        x = torch.relu(self.dropout_mid(x))
        return self.fc2(x).squeeze(-1)


@dataclass
class LSTMResult:
    metrics: dict
    history: dict[str, list[float]]
    best_epoch: int
    vocab_size: int
    glove_coverage: float | None
    n_parameters: int
    device: str
    y_pred: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    y_score: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    scores, targets = [], []
    with torch.set_grad_enabled(training):
        for reviews, headers, len_r, len_h, y in loader:
            reviews, headers, y = reviews.to(device), headers.to(device), y.to(device)
            logits = model(reviews, headers, len_r, len_h)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(y.cpu().numpy())
    scores, targets = np.concatenate(scores), np.concatenate(targets)
    acc = float(((scores > 0.5) == (targets > 0.5)).mean())
    return total_loss / n, acc, scores


def train_and_evaluate(
    train_reviews: Sequence[str],
    train_headers: Sequence[str],
    train_labels: Sequence[int],
    test_reviews: Sequence[str],
    test_headers: Sequence[str],
    test_labels: Sequence[int],
    *,
    glove_path: str | Path | None = None,
    freeze_embeddings: bool = False,
    embedding_dim: int = 300,
    epochs: int = 15,
    patience: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = SEED,
    device: str | None = None,
    verbose: bool = True,
) -> LSTMResult:
    """Train on the training texts (with an internal validation split), score on the test texts.

    All texts are expected to be pre-processed (space-separated lemmas, see
    :class:`airline_reviews.text.TextPreprocessor`).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    y_train = np.asarray(train_labels)
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=val_fraction, random_state=seed, stratify=y_train
    )

    def pick(seq, ids):
        return [seq[i] for i in ids]

    vocab = Vocab.build(pick(train_reviews, fit_idx) + pick(train_headers, fit_idx))

    def make_loader(reviews, headers, labels, shuffle):
        ds = ReviewDataset(
            [vocab.encode(t) for t in reviews], [vocab.encode(t) for t in headers], labels
        )
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=gen
        )

    train_loader = make_loader(
        pick(train_reviews, fit_idx), pick(train_headers, fit_idx), y_train[fit_idx], True
    )
    val_loader = make_loader(
        pick(train_reviews, val_idx), pick(train_headers, val_idx), y_train[val_idx], False
    )
    test_loader = make_loader(test_reviews, test_headers, test_labels, False)

    matrix, coverage = None, None
    if glove_path is not None:
        matrix, coverage = load_glove(glove_path, vocab, dim=embedding_dim, seed=seed)
        if verbose:
            print(f"GloVe coverage: {coverage:.1%} of {len(vocab)} vocabulary words")
    model = DualLSTM(
        len(vocab),
        embedding_dim,
        matrix,
        freeze_embeddings=freeze_embeddings and matrix is not None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state, best_loss, best_epoch, bad = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, _ = _run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if verbose:
            print(
                f"epoch {epoch:2d}  train_loss={tr_loss:.4f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"
            )
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    _, _, scores = _run_epoch(model, test_loader, criterion, device)
    y_pred = (scores > 0.5).astype(int)
    metrics = classification_metrics(test_labels, y_pred, scores)
    return LSTMResult(
        metrics=metrics,
        history=history,
        best_epoch=best_epoch,
        vocab_size=len(vocab),
        glove_coverage=coverage,
        n_parameters=n_params,
        device=device,
        y_pred=y_pred,
        y_score=scores,
    )
