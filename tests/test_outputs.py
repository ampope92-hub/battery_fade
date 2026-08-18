"""Verifier for the battery capacity model.

test.sh deletes any predictions.csv the agent left behind and rebuilds and runs the
Cargo project at /app/predictor, so the file graded here is always the output of a
clean release build. We check that the file exists, its schema, that it covers every
field pack exactly once, that the values are numeric and physically plausible, that
the predictions are a real spread rather than a constant, that the shipped export was
not edited, and that the R2 against the bench results in /opt/golden/test_labels.csv
(never present in the agent environment) reaches the 0.88 stated in the task.

Only pytest is installed, so the CSV reading and the R2 come from the standard library
rather than pandas/scikit-learn.
"""

import csv
import hashlib
import math
import pathlib

import pytest

PRED_PATH = pathlib.Path("/app/predictions.csv")
# Read from the root-only copy baked into the verifier image, not the /tests mount: the
# agent's crate is rebuilt and run in this container, and only this path is closed to it.
LABELS_PATH = pathlib.Path("/opt/golden/test_labels.csv")
TEST_FEATURES = pathlib.Path("/app/data/test_features.csv")
R2_BAR = 0.88

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "fa38f3e37481e9f432d486a9f35dbcf704f7245bab018c9d049b4e989dc802bd"
TRAIN_SHA = "2d4347bf3aae7117c6a342c3a8bfa0b4348f038c1f44c0574e159776c4d46fc8"


def _read_csv(path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, [row for row in reader if row]


@pytest.fixture(scope="session")
def preds():
    assert PRED_PATH.exists(), (
        "/app/predictions.csv was not written — `cargo run --release` in "
        "/app/predictor must produce it"
    )
    return _read_csv(PRED_PATH)


@pytest.fixture(scope="session")
def labels():
    _, rows = _read_csv(LABELS_PATH)
    return {row[0]: float(row[1]) for row in rows}


def _values(preds):
    """capacity_pct column as floats, in file order."""
    _, rows = preds
    return [float(row[1]) for row in rows]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(pred, truth):
    mean = sum(truth) / len(truth)
    ss_res = sum((t - p) ** 2 for p, t in zip(pred, truth))
    ss_tot = sum((t - mean) ** 2 for t in truth)
    assert ss_tot > 0, "bench results have no variance"
    return 1.0 - ss_res / ss_tot


def test_shipped_data_unmodified():
    """The shipped train.csv and test_features.csv are byte-for-byte as issued."""
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    """predictions.csv carries exactly the two columns pack_id and capacity_pct."""
    header, _ = preds
    assert header == ["pack_id", "capacity_pct"], (
        f"predictions.csv must have exactly columns pack_id,capacity_pct, got {header}"
    )


def test_covers_every_pack_exactly_once(preds, labels):
    """Every field pack has exactly one row — no duplicates, omissions or unknown ids."""
    _, rows = preds
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate pack_id entries"
    missing = set(labels) - set(ids)
    extra = set(ids) - set(labels)
    assert not missing, f"{len(missing)} field packs missing (e.g. {sorted(missing)[:3]})"
    assert not extra, f"{len(extra)} unknown pack_ids (e.g. {sorted(extra)[:3]})"


def test_capacities_valid(preds):
    """Each capacity_pct is a finite number inside the physical 0-100 range."""
    _, rows = preds
    for row in rows:
        try:
            value = float(row[1])
        except ValueError:
            pytest.fail(f"capacity_pct is not numeric for pack_id {row[0]!r}: {row[1]!r}")
        assert math.isfinite(value), f"capacity_pct is not finite for pack_id {row[0]!r}"
        assert 0.0 <= value <= 100.0, (
            f"capacity_pct outside 0-100 for pack_id {row[0]!r}: {value}"
        )


def test_predictions_not_degenerate(preds):
    """Predictions differ across packs rather than being one repeated constant."""
    v = _values(preds)
    assert len(set(v)) >= 50, "predictions are (near-)constant — not a real model"
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
    assert sd > 0.5, "predictions carry almost no variation"


def test_r2_bar(preds, labels):
    """R2 against the withheld bench results reaches the 0.88 the task requires."""
    _, rows = preds
    pred = [float(row[1]) for row in rows]
    truth = [labels[row[0]] for row in rows]
    r2 = _r2(pred, truth)
    assert r2 >= R2_BAR, (
        f"R2 against the bench results is {r2:.4f}, below the required {R2_BAR:.2f} — "
        f"the model does not carry to the field packs"
    )
