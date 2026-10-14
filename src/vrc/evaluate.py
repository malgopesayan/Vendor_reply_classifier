"""Metrics shared by both models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def confusion_frame(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def top_confusions(y_true: list[str], y_pred: list[str], labels: list[str], k: int = 5) -> list[tuple[str, str, int]]:
    """The k most frequent (true, predicted) mistakes, for error analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]]
    return sorted(pairs, key=lambda p: -p[2])[:k]
