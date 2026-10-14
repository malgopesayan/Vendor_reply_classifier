import sys
from pathlib import Path

import pytest

from vrc import data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import generate_dataset


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    import csv

    path = tmp_path_factory.mktemp("d") / "replies.csv"
    rows = generate_dataset.generate()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return data.make_splits(data.load(path))


def test_generator_is_reproducible():
    assert generate_dataset.generate(seed=1) == generate_dataset.generate(seed=1)


def test_no_template_leaks_between_splits(splits):
    train, val, test = (set(df["template_id"]) for df in (splits.train, splits.val, splits.test))
    assert not train & val
    assert not train & test
    assert not val & test


def test_every_class_is_in_every_split(splits):
    for df in (splits.train, splits.val, splits.test):
        assert set(df["label"]) == set(splits.labels)


def test_load_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("text,label\nhello,ACKNOWLEDGEMENT\n")
    with pytest.raises(ValueError):
        data.load(bad)
