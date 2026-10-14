"""Training for the PyTorch model and the scikit-learn baseline."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from torch import nn

from vrc.model import ReplyClassifier, encode_batch
from vrc.preprocess import normalize


@dataclass
class TrainConfig:
    """Defaults chosen by comparing settings over 5 seeds (see README, Experiments)."""
    epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-3
    weight_decay: float = 1e-4
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.5
    # Early stopping is off by default: with one validation phrasing per class the
    # validation score is too noisy to pick an epoch reliably. Set a number to enable.
    patience: int | None = None
    seed: int = 42


@dataclass
class TrainResult:
    model: ReplyClassifier
    labels: list[str]
    best_epoch: int
    best_val_f1: float
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_torch(model: ReplyClassifier, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Class probabilities, shape (len(texts), num_classes)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            ids, offsets = encode_batch(texts[i:i + batch_size], model.num_buckets)
            out.append(torch.softmax(model(ids, offsets), dim=1).numpy())
    return np.concatenate(out) if out else np.empty((0, model.head[-1].out_features))


def train_torch(train_texts: list[str], train_labels: list[str], val_texts: list[str], val_labels: list[str],
                labels: list[str], cfg: TrainConfig | None = None) -> TrainResult:
    """Train with AdamW + cross-entropy. Validation is tracked every epoch; with
    cfg.patience set, the best-validation epoch is kept, otherwise the last one."""
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    idx = {label: i for i, label in enumerate(labels)}
    y_train = torch.tensor([idx[label] for label in train_labels], dtype=torch.long)
    y_val = np.array([idx[label] for label in val_labels])

    model = ReplyClassifier(len(labels), embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state, since_best = -1.0, 0, None, 0
    history = []
    order = list(range(len(train_texts)))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            ids, offsets = encode_batch([train_texts[i] for i in batch], model.num_buckets)
            loss = loss_fn(model(ids, offsets), y_train[batch])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(batch)

        val_f1 = float("nan")
        if val_texts:
            val_pred = predict_torch(model, val_texts).argmax(axis=1)
            val_f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 4), "val_macro_f1": round(val_f1, 4)})

        if cfg.patience is None:
            best_f1, best_epoch = val_f1, epoch
            continue
        if val_f1 > best_f1:
            best_f1, best_epoch, since_best = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, labels, best_epoch, round(best_f1, 4), history)


def save_torch(result: TrainResult, cfg: TrainConfig, path: Path | str) -> None:
    torch.save({"state_dict": result.model.state_dict(), "labels": result.labels, "config": asdict(cfg)}, path)


def load_torch(path: Path | str) -> tuple[ReplyClassifier, list[str]]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = ReplyClassifier(len(ckpt["labels"]), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                            dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["labels"]


def train_baseline(train_texts: list[str], train_labels: list[str], seed: int = 42) -> Pipeline:
    """TF-IDF (word 1-2 grams + character 3-5 grams) + logistic regression."""
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    pipe = Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=seed))])
    return pipe.fit(train_texts, train_labels)
