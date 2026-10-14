"""PyTorch text classifier: hashed n-gram features -> EmbeddingBag -> MLP.

This is the fastText idea in a few lines of PyTorch. Each reply becomes a bag of
word 1-2 grams and character 3-5 grams; every feature is hashed into a fixed
number of buckets, so there is no vocabulary to build or keep in sync, and words
never seen in training still get embeddings from their character pieces.
"""
from __future__ import annotations

import zlib

import torch
from torch import nn

from vrc.preprocess import features

NUM_BUCKETS = 2 ** 16


def feature_ids(text: str, num_buckets: int = NUM_BUCKETS) -> list[int]:
    # crc32 rather than hash(): Python's hash() is randomized per process, which
    # would make a saved model useless after a restart.
    feats = features(text) or ["<empty>"]
    return [zlib.crc32(f.encode("utf-8")) % num_buckets for f in feats]


def encode_batch(texts: list[str], num_buckets: int = NUM_BUCKETS) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch into (ids, offsets), the input format EmbeddingBag expects."""
    ids: list[int] = []
    offsets: list[int] = []
    for t in texts:
        offsets.append(len(ids))
        ids.extend(feature_ids(t, num_buckets))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class ReplyClassifier(nn.Module):
    def __init__(self, num_classes: int, num_buckets: int = NUM_BUCKETS, embed_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, embed_dim, mode="mean")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(ids, offsets))
