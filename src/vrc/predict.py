"""Classify a vendor reply from the command line.

    python -m vrc.predict "We will correct the invoice in next month's return."
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vrc.train import load_torch, predict_torch

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "reply_classifier.pt"


def classify(texts: list[str], model_path: Path = DEFAULT_MODEL, top_k: int = 3) -> list[list[tuple[str, float]]]:
    model, labels = load_torch(model_path)
    probs = predict_torch(model, texts)
    return [[(labels[i], round(float(row[i]), 3)) for i in row.argsort()[::-1][:top_k]] for row in probs]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    a = p.parse_args()
    for text, preds in zip(a.text, classify(a.text, a.model)):
        print(text)
        for label, prob in preds:
            print(f"  {label:<26} {prob:.3f}")


if __name__ == "__main__":
    main()
