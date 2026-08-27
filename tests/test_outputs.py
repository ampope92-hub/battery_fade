"""Verifier for the battery capacity and replacement-scheduling model.

test.sh deletes any predictions.csv the agent left behind and rebuilds and runs the
Cargo project at /app/predictor, so the file graded here is always the output of a
clean release build.

Two things are graded, because the fleet is scheduled off both. The capacity column is
scored by R2 against the withheld bench results. The replace_cycle column is scored by
how often the recommendation lands too late — after the pack has already dropped through
80% of nameplate — which is checked fleet-wide and again inside each third of the fleet
by fast-charge share, since a schedule can hit the fleet number by running early on one
kind of duty and late on another.

Only pytest is installed, so the CSV reading and the arithmetic come from the standard
library rather than pandas/scikit-learn.
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

R2_BAR = 0.55
# Share of packs whose recommended replacement cycle falls after they have already crossed
# 80% of nameplate. The task asks for at most about 1 in 10; both bands are two-sided, so a
# schedule that replaces everything far too early fails as well.
FLEET_LATE_PCT = (7.0, 13.5)
BAND_LATE_PCT = (6.0, 14.5)
N_BANDS = 3
THRESHOLD_PCT = 80.0

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "3057b258630f005818f0f38ee3dee952f9179f9e22b74b30df95331538a492d1"
TRAIN_SHA = "389df1daa5b626fc4c3020ec985db64d137d703e1d40306ba73b63158c4e9813"


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
    """pack_id -> (bench capacity_pct, the cycle count at which the pack really crossed 80%)."""
    header, rows = _read_csv(LABELS_PATH)
    i_cap = header.index("capacity_pct")
    i_cross = header.index("true_cross_cycle")
    return {row[0]: (float(row[i_cap]), float(row[i_cross])) for row in rows}


def _column(preds, index):
    _, rows = preds
    return [float(row[index]) for row in rows]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(pred, truth):
    mean = sum(truth) / len(truth)
    ss_res = sum((t - p) ** 2 for p, t in zip(pred, truth))
    ss_tot = sum((t - mean) ** 2 for t in truth)
    assert ss_tot > 0, "bench results have no variance"
    return 1.0 - ss_res / ss_tot


def _late_pct(rows, labels, ids=None):
    """Share of the given packs whose recommendation lands after the true crossing."""
    sel = [r for r in rows if ids is None or r[0] in ids]
    assert sel, "no packs in this group"
    late = sum(1 for r in sel if float(r[2]) > labels[r[0]][1])
    return 100.0 * late / len(sel)


def test_shipped_data_unmodified():
    """The shipped train.csv and test_features.csv are byte-for-byte as issued."""
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    """predictions.csv carries exactly pack_id, capacity_pct and replace_cycle."""
    header, _ = preds
    assert header == ["pack_id", "capacity_pct", "replace_cycle"], (
        "predictions.csv must have exactly columns pack_id,capacity_pct,replace_cycle, "
        f"got {header}"
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


def test_replace_cycles_valid(preds):
    """Each replace_cycle is a finite, positive cycle count."""
    _, rows = preds
    for row in rows:
        try:
            value = float(row[2])
        except ValueError:
            pytest.fail(f"replace_cycle is not numeric for pack_id {row[0]!r}: {row[2]!r}")
        assert math.isfinite(value), f"replace_cycle is not finite for pack_id {row[0]!r}"
        assert value > 0.0, (
            f"replace_cycle must be a positive cycle count for pack_id {row[0]!r}: {value}"
        )


def test_predictions_not_degenerate(preds):
    """Both columns differ across packs rather than being one repeated constant."""
    for index, name in ((1, "capacity_pct"), (2, "replace_cycle")):
        v = _column(preds, index)
        assert len(set(v)) >= 50, f"{name} is (near-)constant — not a real model"
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
        assert sd > 0.5, f"{name} carries almost no variation"


@pytest.fixture(scope="session")
def duty_bands():
    """The field packs split into three equal groups by fast-charge share.

    Duty comes from the shipped export, so this re-checks its sha256 rather than relying on
    test_shipped_data_unmodified having run first — the crate is rebuilt and run in this
    container, and which packs land in which group must not be something it can steer.
    Ordering breaks ties on pack_id so the split is identical on every run.
    """
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    header, rows = _read_csv(TEST_FEATURES)
    i_id, i_fast = header.index("pack_id"), header.index("fast_charge_pct")
    ordered = sorted(rows, key=lambda r: (float(r[i_fast]), r[i_id]))
    n = len(ordered)
    bands = [set() for _ in range(N_BANDS)]
    for rank, row in enumerate(ordered):
        bands[min(rank * N_BANDS // n, N_BANDS - 1)].add(row[i_id])
    return bands


def test_r2_bar(preds, labels):
    """R2 against the withheld bench results reaches the level the task requires."""
    _, rows = preds
    pred = [float(row[1]) for row in rows]
    truth = [labels[row[0]][0] for row in rows]
    r2 = _r2(pred, truth)
    assert r2 >= R2_BAR, (
        f"R2 against the bench results is {r2:.4f}, below the required {R2_BAR:.2f} — "
        f"the capacity model does not carry to the field packs"
    )


def test_replacement_schedule_coverage(preds, labels):
    """Fleet-wide, the share of packs replaced after they crossed 80% is near 1 in 10."""
    _, rows = preds
    late = _late_pct(rows, labels)
    lo, hi = FLEET_LATE_PCT
    assert lo <= late <= hi, (
        f"{late:.2f}% of the fleet is scheduled for replacement after it has already "
        f"dropped through {THRESHOLD_PCT:.0f}% of nameplate; the task asks for at most "
        f"about 1 in 10, and the accepted range is {lo:.1f}-{hi:.1f}%"
    )


def test_replacement_schedule_coverage_within_duty_bands(preds, labels, duty_bands):
    """That share also holds inside each third of the fleet by fast-charge share."""
    _, rows = preds
    lo, hi = BAND_LATE_PCT
    names = ("lowest", "middle", "highest")
    for name, ids in zip(names, duty_bands):
        late = _late_pct(rows, labels, ids)
        assert lo <= late <= hi, (
            f"in the {name} third of the fleet by fast-charge share, {late:.2f}% of packs "
            f"are scheduled for replacement after they have already dropped through "
            f"{THRESHOLD_PCT:.0f}% of nameplate; the accepted range is {lo:.1f}-{hi:.1f}% "
            f"in every third"
        )
