"""Train both models, evaluate on the held-out test templates, write results.

Run from the repo root:
    PYTHONPATH=src python -m vrc.run_experiment                       # full run
    PYTHONPATH=src python -m vrc.run_experiment --epochs 3 --seeds 1  # quick smoke run (CI)

Protocol
  1. Both models are trained on the train templates only.
  2. The PyTorch model is trained with 5 different seeds; the test score is
     reported as mean and standard deviation, because one seed on a small
     dataset can be lucky or unlucky by several points.
  3. Error analysis (confusion matrix, per-class scores) uses the seed-42 model.
  4. The model saved for prediction is refit on train + validation templates
     with the same settings, the usual last step once settings are chosen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vrc import data
from vrc.evaluate import confusion_frame, metrics, top_confusions
from vrc.train import TrainConfig, predict_torch, save_torch, train_baseline, train_torch

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 1, 7, 3, 11]


def run(epochs: int, n_seeds: int, out_dir: Path, model_dir: Path) -> dict:
    df = data.load()
    s = data.make_splits(df)
    x_tr, y_tr = s.train["text"].tolist(), s.train["label"].tolist()
    x_va, y_va = s.val["text"].tolist(), s.val["label"].tolist()
    x_te, y_te = s.test["text"].tolist(), s.test["label"].tolist()

    t0 = time.perf_counter()
    baseline = train_baseline(x_tr, y_tr)
    base_secs = time.perf_counter() - t0
    m_base = metrics(y_te, baseline.predict(x_te).tolist(), s.labels)

    runs, first = [], None
    for seed in SEEDS[:n_seeds]:
        cfg = TrainConfig(epochs=epochs, seed=seed)
        t0 = time.perf_counter()
        res = train_torch(x_tr, y_tr, x_va, y_va, s.labels, cfg)
        secs = time.perf_counter() - t0
        pred = [s.labels[i] for i in predict_torch(res.model, x_te).argmax(axis=1)]
        m = metrics(y_te, pred, s.labels)
        runs.append({"seed": seed, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "train_seconds": round(secs, 2)})
        if first is None:
            first = (res, pred, m)

    res0, pred0, m0 = first
    acc = np.array([r["accuracy"] for r in runs])
    f1 = np.array([r["macro_f1"] for r in runs])
    summary = {
        "dataset": {"total": len(df), "train": len(s.train), "val": len(s.val), "test": len(s.test),
                    "classes": len(s.labels), "train_templates_per_class": int(s.train.groupby("label")["template_id"]
                                                                               .nunique().iloc[0])},
        "baseline_tfidf_logreg": {"accuracy": m_base["accuracy"], "macro_f1": m_base["macro_f1"],
                                  "train_seconds": round(base_secs, 2)},
        "pytorch_embeddingbag": {"accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(float(acc.std()), 4),
                                 "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std()), 4),
                                 "seeds": runs},
        "pytorch_top_confusions_seed42": top_confusions(y_te, pred0, s.labels),
    }

    final_cfg = TrainConfig(epochs=epochs, seed=SEEDS[0])
    final = train_torch(x_tr + x_va, y_tr + y_va, [], [], s.labels, final_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    per_class = pd.DataFrame(m0["per_class"]).T.loc[s.labels, ["precision", "recall", "f1-score", "support"]].round(3)
    per_class.to_csv(out_dir / "per_class_pytorch_seed42.csv")
    (out_dir / "training_history_seed42.json").write_text(json.dumps(res0.history, indent=2))
    confusion_frame(y_te, pred0, s.labels).to_csv(out_dir / "confusion_matrix_pytorch_seed42.csv")
    data.describe(df).to_csv(out_dir / "dataset_stats.csv")
    save_torch(final, final_cfg, model_dir / "reply_classifier.pt")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--seeds", type=int, default=len(SEEDS), help="how many seeds to evaluate (max 5)")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    a = p.parse_args()
    print(json.dumps(run(a.epochs, max(1, min(a.seeds, len(SEEDS))), a.out, a.models), indent=2))


if __name__ == "__main__":
    main()
