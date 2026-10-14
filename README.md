# Vendor Reply Intent Classifier

![CI](https://github.com/malgopesayan/vendor-reply-classifier/actions/workflows/ci.yml/badge.svg)

Classifies a vendor's email reply to a GST mismatch reminder into one of 10 intents, such as
"will correct next month", "already corrected", or "invoice attached". It compares a
**PyTorch** neural model against a **scikit-learn** TF-IDF + logistic regression baseline.

The idea comes from a production GST reconciliation system I built, where an LLM agent reads
vendor replies. This repo asks a narrower question: **how far can a small, fast, trained
classifier get on the same task?**

## Results

Both models are tested only on **phrasings they never saw during training** (see Data).

| Model | Test accuracy | Test macro-F1 | Train time (CPU) |
|---|---|---|---|
| TF-IDF (word + char n-grams) + logistic regression | **0.744** | **0.741** | 0.4 s |
| PyTorch EmbeddingBag + MLP (mean ± std over 5 seeds) | 0.698 ± 0.024 | 0.681 ± 0.030 | ~33 s per seed |

**Takeaway:** with only 6 training phrasings per intent, the linear baseline beats the neural
model and is about 80× faster to train. The neural model needs more varied data to pull ahead.

## Data

`data/generate_dataset.py` creates 1,200 synthetic replies (10 intents × 10 templates × 12 samples)
with a fixed seed, so the dataset is fully reproducible.

- **Why synthetic:** the real replies belong to a client and contain names, GSTINs and invoice
  numbers. The intents and phrasing style are modelled on that workflow, but every sentence,
  name and number here is made up.
- **Why split by template:** if one template appeared in both train and test, a model could
  score close to 100% by memorizing the wording. Instead, 3 templates per intent are test-only,
  1 is validation-only, and 6 are for training. The test score therefore measures how well a
  model handles new wording.
- **Noise:** about a third of replies are lowercased, have a dropped word, or a typo, like real email.

| Split | Replies | Templates per intent |
|---|---|---|
| Train | 720 | 6 |
| Validation | 120 | 1 |
| Test | 360 | 3 |

## Method

**Preprocessing** (`src/vrc/preprocess.py`): lowercase, then replace intent-free values with
placeholder tokens (`<doc>`, `<amount>`, `<gstin>`, `<month>`, `<num>`). This stops a model
from memorizing specific invoice numbers. Features are word 1-2 grams plus character 3-5 grams,
so "amend", "amended" and "amendment" share most of their features.

**Baseline** (`train.py`): TF-IDF on word and character n-grams, then logistic regression (C=1).

**PyTorch model** (`model.py`): the fastText idea. Every feature is hashed into 65,536
buckets, averaged by an `nn.EmbeddingBag` (128 dims), then a 2-layer MLP with dropout 0.5.
Hashing means there is no vocabulary to maintain, and unseen words still get embeddings from
their character pieces. Trained with AdamW and cross-entropy for 20 epochs.

## Experiments

Each change was kept only if it helped on validation and was then confirmed on test.

| Change | Baseline macro-F1 | PyTorch macro-F1 |
|---|---|---|
| Word 1-2 grams only | 0.690 | 0.668 (1 seed, early stopping) |
| + character 3-5 grams | **0.741** | 0.62 – 0.73 depending on seed |
| PyTorch: 20 fixed epochs, 128 dims, dropout 0.5 | — | **0.681 ± 0.030** (5 seeds) |

Notes:
- **Early stopping was removed.** With one validation phrasing per intent, the validation score
  was too noisy to pick the right epoch, and results swung by ±5 points between seeds.
- **Reporting the 5-seed mean** instead of the best single seed, because one lucky seed
  (0.734) would overstate the model.

## Error analysis (PyTorch, seed 42)

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| CONTACT_CHANGE | 0.92 | 0.94 | 0.93 |
| CREDIT_NOTE_EXPLANATION | 0.84 | 1.00 | 0.91 |
| ACKNOWLEDGEMENT | 0.78 | 1.00 | 0.88 |
| MEETING_REQUEST | 1.00 | 0.67 | 0.80 |
| WRONG_BUYER_GSTIN | 1.00 | 0.67 | 0.80 |
| DOCUMENTS_SUBMITTED | 0.60 | 0.75 | 0.67 |
| ALREADY_CORRECTED | 0.57 | 0.67 | 0.62 |
| STATUS_REQUEST | 0.41 | 0.67 | 0.51 |
| DISPUTES_DATA | 0.35 | 0.33 | 0.34 |
| WILL_AMEND | 0.67 | 0.11 | 0.19 |

The model is weakest on intents that share vocabulary and differ only in tense or stance:

- **"will amend" vs "already corrected"**: both use *amend, correct, return*. A bag-of-n-grams
  model barely sees tense. In the real system this is the most important distinction (one
  means re-check next month, the other means verify now), which is why that system checks it
  with explicit rules rather than trusting a model.
- **"disputes our data" vs "status request"**: both ask the reader to check something.

Full numbers are in `results/` (confusion matrix, per-class scores, training history).

## Limitations

- Synthetic data: real replies are messier, mix several intents, and include Hindi words.
- Overconfident probabilities: the PyTorch model often outputs ~1.00 even when wrong, so its
  scores should not be used as confidence thresholds without calibration.
- Next steps: pretrained sentence embeddings, more templates per intent, and probability calibration.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python data/generate_dataset.py                 # rebuild the dataset (deterministic)
PYTHONPATH=src python -m vrc.run_experiment      # train, evaluate, write results/ and models/
PYTHONPATH=src python -m vrc.predict "We will rectify this in the next return."
pytest -q                                        # 13 tests
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request, on Python 3.10 and 3.12:

1. Lint with ruff
2. Run the unit tests
3. Regenerate the dataset and fail if it changed (reproducibility check)
4. Smoke-train both models for 3 epochs
5. Run the full experiment and fail if either model drops below its quality gate
   (baseline ≥ 0.68, PyTorch ≥ 0.60 macro-F1), a few points under the results above

## Project layout

```
data/generate_dataset.py   synthetic dataset generator
src/vrc/preprocess.py      normalization, word and character n-grams
src/vrc/data.py            loading, template-level splits, dataset stats
src/vrc/model.py           PyTorch EmbeddingBag classifier
src/vrc/train.py           training loops for both models, save / load
src/vrc/evaluate.py        metrics, confusion matrix, top confusions
src/vrc/run_experiment.py  full experiment, writes results/
src/vrc/predict.py         command-line prediction
tests/                     unit tests
```
