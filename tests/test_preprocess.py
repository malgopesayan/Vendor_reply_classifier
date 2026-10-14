from vrc.preprocess import ngrams, normalize, tokenize


def test_replaces_invoice_numbers_amounts_months_and_gstins():
    text = "Invoice INV-48213 of Rs 1,25,000 for GSTIN 27ABCDE1234F1Z5 will be fixed in March."
    out = normalize(text)
    assert "<doc>" in out
    assert "<amount>" in out
    assert "<gstin>" in out
    assert "<month>" in out
    assert "48213" not in out and "27abcde" not in out


def test_tokenize_keeps_placeholders_whole():
    assert tokenize("Please check TI/9921") == ["please", "check", "<doc>"]


def test_ngrams_include_bigrams():
    feats = ngrams(["next", "month", "return"])
    assert "next month" in feats and "month return" in feats and "next" in feats


def test_empty_text_is_safe():
    assert tokenize("") == []
