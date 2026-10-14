import torch

from vrc.model import NUM_BUCKETS, ReplyClassifier, encode_batch, feature_ids
from vrc.train import TrainConfig, load_torch, predict_torch, save_torch, train_torch

TEXTS = [
    "We will amend the invoice in next month's return.",
    "We shall correct it in the coming month filing.",
    "Please find attached the invoice copy.",
    "Attached is the signed invoice PDF.",
    "Can we schedule a call to discuss?",
    "Please share a time for a meeting.",
] * 5
LABELS = ["WILL_AMEND", "WILL_AMEND", "DOCUMENTS_SUBMITTED", "DOCUMENTS_SUBMITTED",
          "MEETING_REQUEST", "MEETING_REQUEST"] * 5
CLASSES = sorted(set(LABELS))


def test_feature_ids_are_stable_and_in_range():
    a = feature_ids("Please find attached the invoice")
    assert a == feature_ids("Please find attached the invoice")
    assert all(0 <= i < NUM_BUCKETS for i in a)


def test_forward_shape():
    model = ReplyClassifier(num_classes=4)
    ids, offsets = encode_batch(["hello there", "invoice attached", ""])
    assert model(ids, offsets).shape == (3, 4)


def test_training_learns_a_simple_task():
    res = train_torch(TEXTS, LABELS, TEXTS, LABELS, CLASSES, TrainConfig(epochs=30, patience=30))
    pred = [CLASSES[i] for i in predict_torch(res.model, TEXTS).argmax(axis=1)]
    assert sum(p == y for p, y in zip(pred, LABELS)) / len(LABELS) >= 0.9


def test_probabilities_sum_to_one():
    model = ReplyClassifier(num_classes=3)
    probs = predict_torch(model, ["any text", "another"])
    assert torch.allclose(torch.tensor(probs.sum(axis=1)), torch.ones(2), atol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    cfg = TrainConfig(epochs=2)
    res = train_torch(TEXTS, LABELS, TEXTS, LABELS, CLASSES, cfg)
    path = tmp_path / "m.pt"
    save_torch(res, cfg, path)
    model, labels = load_torch(path)
    assert labels == CLASSES
    assert (predict_torch(model, TEXTS[:3]) - predict_torch(res.model, TEXTS[:3])).max() < 1e-6
