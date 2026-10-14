"""Text preprocessing shared by the PyTorch model and the scikit-learn baseline.

Vendor replies are full of values that carry no intent: invoice numbers, GSTINs,
amounts and dates. Replacing them with placeholder tokens stops the model from
memorizing specific numbers and lets it learn the wording around them instead.
"""
from __future__ import annotations

import re

GSTIN_RE = re.compile(r"\b\d{2}[a-z]{5}\d{4}[a-z]\d?z[a-z0-9]\b", re.IGNORECASE)
DOC_RE = re.compile(r"\b[a-z]{2,4}[-/]?\d{3,}\b", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\b(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d+)?\b", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
MONTH_RE = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                      re.IGNORECASE)
TOKEN_RE = re.compile(r"<[a-z]+>|[a-z]+(?:'[a-z]+)?|\d")


def normalize(text: str) -> str:
    """Lowercase and replace intent-free values with placeholder tokens."""
    text = text.lower()
    text = GSTIN_RE.sub(" <gstin> ", text)
    text = AMOUNT_RE.sub(" <amount> ", text)
    text = DOC_RE.sub(" <doc> ", text)
    text = MONTH_RE.sub(" <month> ", text)
    text = NUM_RE.sub(" <num> ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalize, then split into word tokens (placeholders kept whole)."""
    return TOKEN_RE.findall(normalize(text))


def ngrams(tokens: list[str], n_max: int = 2) -> list[str]:
    """Unigrams plus word n-grams up to n_max, e.g. 'next month' as one feature."""
    feats = list(tokens)
    for n in range(2, n_max + 1):
        feats.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return feats


def char_ngrams(tokens: list[str], sizes: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    """Character n-grams inside each word, with boundary markers (fastText style).

    'amend', 'amended' and 'amendment' share most of their pieces, and a typo
    like 'corected' still overlaps 'corrected', so these help on wording the
    model has not seen. Placeholder tokens are skipped.
    """
    feats = []
    for t in tokens:
        if t.startswith("<"):
            continue
        w = f"<{t}>"
        feats.extend(w[i:i + n] for n in sizes for i in range(len(w) - n + 1))
    return feats


def features(text: str) -> list[str]:
    """All features for one reply: word 1-2 grams plus character 3-5 grams."""
    tokens = tokenize(text)
    return ngrams(tokens) + char_ngrams(tokens)
