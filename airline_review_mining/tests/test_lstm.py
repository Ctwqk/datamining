import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airline_reviews import lstm  # noqa: E402


def test_vocab_reserves_pad_and_unk_and_encodes():
    vocab = lstm.Vocab.build(["good seat", "bad seat"])
    assert vocab.itos[:2] == ["<pad>", "<unk>"]
    assert len(vocab) == 5
    assert vocab.encode("good seat") == [vocab.stoi["good"], vocab.stoi["seat"]]
    assert vocab.encode("unseen word") == [lstm.UNK, lstm.UNK]
    assert vocab.encode("") == [lstm.UNK]  # never an empty sequence


def test_vocab_min_freq():
    vocab = lstm.Vocab.build(["a a b", "a c"], min_freq=2)
    assert vocab.itos == ["<pad>", "<unk>", "a"]


def test_collate_pads_to_longest_with_lengths():
    ds = lstm.ReviewDataset([[2, 3, 4], [5]], [[6], [7, 8]], [1, 0])
    reviews, headers, len_r, len_h, y = lstm.collate([ds[0], ds[1]])
    assert reviews.tolist() == [[2, 3, 4], [5, lstm.PAD, lstm.PAD]]
    assert headers.tolist() == [[6, lstm.PAD], [7, 8]]
    assert len_r.tolist() == [3, 1] and len_h.tolist() == [1, 2]
    assert y.tolist() == [1.0, 0.0]


def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        lstm.ReviewDataset([[1]], [[1], [2]], [1])


def test_model_forward_shape_and_padding_invariance():
    torch.manual_seed(0)
    model = lstm.DualLSTM(vocab_size=10, embedding_dim=8).eval()
    reviews = torch.tensor([[2, 3, 4], [5, 0, 0]])
    headers = torch.tensor([[6, 7], [8, 0]])
    len_r, len_h = torch.tensor([3, 1]), torch.tensor([2, 1])
    out = model(reviews, headers, len_r, len_h)
    assert out.shape == (2,)
    # the second example must not depend on how much padding follows it
    alone = model(torch.tensor([[5]]), torch.tensor([[8]]), torch.tensor([1]), torch.tensor([1]))
    assert torch.allclose(out[1], alone[0], atol=1e-5)


def test_freezing_embeddings_is_effective():
    frozen = lstm.DualLSTM(10, 8, freeze_embeddings=True)
    trainable = lstm.DualLSTM(10, 8, freeze_embeddings=False)
    assert not frozen.embedding.weight.requires_grad
    assert trainable.embedding.weight.requires_grad


def test_load_glove_keeps_only_vocab_words(tmp_path):
    dim = 4
    path = tmp_path / "glove.txt"
    path.write_text(
        "seat 0.1 0.2 0.3 0.4\nnot_in_vocab 1 1 1 1\ncrew 1.0 2.0 3.0 4.0\nbroken 1 2\n",
        encoding="utf8",
    )
    vocab = lstm.Vocab.build(["seat crew meal"])
    matrix, coverage = lstm.load_glove(path, vocab, dim=dim)
    assert matrix.shape == (len(vocab), dim)
    assert np.allclose(matrix[vocab.stoi["crew"]], [1, 2, 3, 4])
    assert np.allclose(matrix[vocab.stoi["seat"]], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(matrix[lstm.PAD], 0)
    assert coverage == pytest.approx(2 / 3)  # 'meal' missing


def test_load_glove_from_zip(tmp_path):
    import zipfile

    path = tmp_path / "glove.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("glove.txt", "seat 1 2 3\n")
    vocab = lstm.Vocab.build(["seat"])
    matrix, coverage = lstm.load_glove(path, vocab, dim=3)
    assert np.allclose(matrix[vocab.stoi["seat"]], [1, 2, 3]) and coverage == 1.0


def test_training_learns_a_separable_toy_problem():
    pos = [
        "great crew wonderful seat",
        "excellent food lovely crew",
        "wonderful lovely flight",
    ] * 30
    neg = ["awful delay terrible food", "rude crew horrible seat", "terrible awful flight"] * 30
    texts = pos + neg
    headers = ["great"] * len(pos) + ["awful"] * len(neg)
    labels = [1] * len(pos) + [0] * len(neg)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    headers = [headers[i] for i in order]
    labels = [labels[i] for i in order]
    res = lstm.train_and_evaluate(
        texts[:120], headers[:120], labels[:120], texts[120:], headers[120:], labels[120:],
        embedding_dim=16, epochs=25, patience=25, batch_size=32, lr=5e-3, device="cpu", verbose=False,
    )  # fmt: skip
    assert res.metrics["accuracy"] > 0.9
    assert len(res.history["train_loss"]) == len(res.history["val_loss"]) >= 1
    assert res.history["train_loss"][-1] < res.history["train_loss"][0]


def test_training_uses_fit_rows_and_scores_test_only_at_the_end(monkeypatch):
    run_epoch = lstm._run_epoch
    calls = []

    def record_epoch(model, loader, criterion, device, optimizer=None):
        calls.append((len(loader.dataset), optimizer is not None))
        return run_epoch(model, loader, criterion, device, optimizer)

    monkeypatch.setattr(lstm, "_run_epoch", record_epoch)
    result = lstm.train_and_evaluate(
        [f"{'good' if i % 2 == 0 else 'bad'} flight token{i}" for i in range(20)],
        ["good", "bad"] * 10,
        [1, 0] * 10,
        ["unseen"] * 3,
        ["unseen"] * 3,
        [1, 0, 1],
        embedding_dim=4,
        epochs=2,
        batch_size=4,
        val_fraction=0.1,
        device="cpu",
        verbose=False,
    )
    assert calls == [(18, True), (2, False), (18, True), (2, False), (3, False)]
    # 18 fit-only tokens + good/bad/flight + pad/unk; no validation- or test-only words.
    assert result.vocab_size == 23
    assert len(result.y_pred) == 3


def test_epoch_thresholds_probabilities_and_keeps_partial_batch():
    model = lstm.DualLSTM(vocab_size=3, embedding_dim=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc2.bias.fill_(0.2)  # positive probability, despite logit being below 0.5
    dataset = lstm.ReviewDataset([[2]] * 5, [[2]] * 5, [1] * 5)
    loader = lstm.DataLoader(dataset, batch_size=2, collate_fn=lstm.collate)
    _, accuracy, scores = lstm._run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), "cpu")
    assert accuracy == 1.0
    assert scores.shape == (5,)
    assert np.allclose(scores, 1 / (1 + np.exp(-0.2)))
