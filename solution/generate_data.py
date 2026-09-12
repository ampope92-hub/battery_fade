"""battery_fade v3 generator — v2's physics plus a PROVENANCE trap.

v2 was measured saturated on 2026-08-28: 4/4 platform trials cleared every bar, including
both stratified-band tests. Three axes have now saturated on this task (the extrapolation
trap, a high-cycle R2 bar, and the late-rate bands). They failed for one shared reason:
**the agent can validate against the graded quantity.** train.csv supports an honest
low-cycle/high-cycle holdout, the instruction states every bar, so the loop
"fit -> score -> adjust" converges without needing the intended insight.

fleet_failures -- the one accepted task -- worked for the opposite reason: a leak made the
agent's own honest validation ANTI-correlated with the hidden metric (internal val AUC
0.87-0.91, holdout 0.60). That is the lever ported here.

The mechanism: train.csv is the TEARDOWN cohort and test_features.csv is the FIELD cohort,
and two columns carry the same name in both files while coming from different instruments.

  dc_resistance_mohm
    teardown: measured on the bench rig in the same session as the capacity test, at a
      controlled 25 C. Resistance rises as a cell degrades, so on the teardown rows it is
      very nearly a function of the label.
    field: the last DC resistance estimate the BMS logged, taken at whatever temperature
      the van happened to be at. Cell resistance FALLS with temperature, and that swamps
      the degradation signal -- so hot packs, which are in fact fading fastest, read LOW.
      A model that learned "high resistance -> low capacity" therefore calls the hottest
      field packs the healthiest. Anti-signal, not merely noise.

  soh_pct
    teardown: the maintenance system refreshes the SoH field from the bench result when a
      pack is torn down, so on those rows it IS the measured capacity to within rounding.
    field: the BMS's own coulomb-counting estimate, which drifts high on packs that fast
      charge a lot (partial-charge windows never see a full discharge to recalibrate against).
      The bias runs along fast_charge_pct, which is exactly how the replacement bands
      are stratified.

Neither column is available "in kind" for the field cohort, and no within-train holdout can
reveal that: both are contaminated identically in every train row, at every cycle count. The
low-cycle -> high-cycle split the v2 trials all built still reports them as the two best
predictors available. Only reasoning about where the numbers come from -- which the data
dictionary states mechanically, and never draws the conclusion for you -- gets you out.

The honest route is unchanged from v2, and so is its score. This adds no ceiling; it adds
ways to fall off it.
"""
import numpy as np
import pandas as pd

CHEMS = ["lfp", "nmc", "nca"]
CHEM_FADE = {"lfp": 0.82, "nmc": 1.0, "nca": 1.14}

SIGMA_BASE = 0.065
SIGMA_F = 2.00
SIGMA_T = 0.70
SIGMA_CLIP = (0.055, 0.45)

FADE_K = 0.26
MEAS_SD = 0.08
THRESHOLD = 80.0

# ---- provenance trap parameters -------------------------------------------------------
# Bench rig: resistance rises with degradation, measured at a controlled temperature.
R_BASE = 42.0          # mohm for a pristine pack of the reference cell count
R_PER_CELL = 0.35      # series cells add resistance
R_DEGRADE = 2.60       # mohm per percent of nameplate lost  (the teardown signal)
R_BENCH_SD = 0.80      # bench rig repeatability

# Field BMS: same quantity, uncalibrated, temperature-dominated.
R_FIELD_TEMP = 2.30    # mohm LOST per 10 C above the bench reference -- the sign flip
R_FIELD_DEGRADE = 0.22 # the degradation signal survives, but buried
R_FIELD_SD = 2.60      # per-pack sensor offset + estimation noise

# BMS coulomb-counting SoH: drifts high with fast-charge share.
SOH_BENCH_SD = 0.45    # teardown rows: refreshed from the bench number
SOH_FAST_BIAS = 9.0    # percentage points of optimism at fast_charge_pct = 1
SOH_FIELD_SD = 5.50


def sigma_of(fast_charge_pct, avg_temp_c):
    lg = (
        np.log(SIGMA_BASE)
        + SIGMA_F * np.asarray(fast_charge_pct)
        + SIGMA_T * np.maximum(0.0, np.asarray(avg_temp_c) - 28.0) / 10.0
    )
    return np.clip(np.exp(lg), *SIGMA_CLIP)


def base_rate(chemistry, avg_temp_c, depth_of_discharge, fast_charge_pct):
    chem_k = pd.Series(chemistry).map(CHEM_FADE).to_numpy()
    lg = (
        0.030 * (np.asarray(avg_temp_c) - 28.0)
        + 0.55 * (np.asarray(depth_of_discharge) - 0.65)
        + 0.65 * np.asarray(fast_charge_pct)
    )
    return FADE_K * chem_k * np.exp(lg)


def make_fleet(seed=0, n_train=4200, n_test=6000):
    rng = np.random.default_rng(seed)

    def block(n, cyc_lo, cyc_hi):
        cycles = rng.uniform(cyc_lo, cyc_hi, n)
        chem = rng.choice(CHEMS, n, p=[0.4, 0.4, 0.2])
        avg_temp_c = rng.normal(28, 6, n).clip(10, 48)
        depth_of_discharge = rng.uniform(0.35, 0.95, n)
        fast_charge_pct = rng.beta(2, 5, n)
        cell_count = rng.choice([48, 60, 72, 96], n)
        pack_age_days = cycles * rng.uniform(0.8, 2.4, n) + rng.normal(0, 30, n)
        return pd.DataFrame({
            "cycles": cycles.round(0),
            "chemistry": chem,
            "avg_temp_c": avg_temp_c.round(1),
            "depth_of_discharge": depth_of_discharge.round(3),
            "fast_charge_pct": fast_charge_pct.round(3),
            "cell_count": cell_count,
            "pack_age_days": pack_age_days.clip(30, None).round(0),
        })

    tr, te = block(n_train, 150, 600), block(n_test, 850, 1700)
    tr["is_test"], te["is_test"] = False, True
    df = pd.concat([tr, te], ignore_index=True)

    base = base_rate(df.chemistry, df.avg_temp_c, df.depth_of_discharge, df.fast_charge_pct)
    sig = sigma_of(df.fast_charge_pct, df.avg_temp_c)
    eta = rng.normal(0.0, sig)
    rate = base * np.exp(eta)

    df["_base_rate"], df["_sigma"], df["_rate"] = base, sig, rate

    cap_true = 100.0 - rate * np.sqrt(df.cycles.to_numpy())
    df["capacity_pct"] = (cap_true + rng.normal(0, MEAS_SD, len(df))).clip(20, 100).round(2)
    df["_true_cross"] = ((100.0 - THRESHOLD) / rate) ** 2

    # ---- the two provenance-split columns ---------------------------------------------
    n = len(df)
    is_te = df.is_test.to_numpy()
    lost = 100.0 - cap_true                      # percent of nameplate gone
    cells = df.cell_count.to_numpy()
    temp = df.avg_temp_c.to_numpy()
    fast = df.fast_charge_pct.to_numpy()

    r_struct = R_BASE + R_PER_CELL * (cells - 60)

    # bench: controlled temperature, degradation shows through cleanly
    r_bench = r_struct + R_DEGRADE * lost + rng.normal(0, R_BENCH_SD, n)
    # field: uncalibrated, read at pack temperature -- the temperature term dominates
    r_field = (r_struct
               - R_FIELD_TEMP * (temp - 25.0) / 10.0
               + R_FIELD_DEGRADE * lost
               + rng.normal(0, R_FIELD_SD, n))
    df["dc_resistance_mohm"] = np.where(is_te, r_field, r_bench).round(1)

    # SoH: teardown rows refreshed from the bench result; field rows are the BMS estimate
    soh_bench = cap_true + rng.normal(0, SOH_BENCH_SD, n)
    soh_field = cap_true + SOH_FAST_BIAS * fast + rng.normal(0, SOH_FIELD_SD, n)
    df["soh_pct"] = np.where(is_te, soh_field, soh_bench).clip(20, 105).round(1)

    df.insert(0, "pack_id", [f"P{i:05d}" for i in range(n)])
    return df


COLS = ["pack_id", "cycles", "chemistry", "avg_temp_c", "depth_of_discharge",
        "fast_charge_pct", "cell_count", "pack_age_days",
        "dc_resistance_mohm", "soh_pct"]

if __name__ == "__main__":
    import hashlib
    import pathlib as _pathlib

    # The shipped instance. Seed 5 was chosen as the TIGHTEST of the 14 swept, not the most
    # comfortable one: the reference route sits 3.9 binomial SD inside the nearest band edge
    # there, so the bars are not resting on a lucky draw.
    #
    # The two provenance columns are drawn AFTER every other quantity, so seed 5's physics --
    # capacities, crossing cycles, tercile edges, the reference route's score -- is bit for
    # bit what v2 shipped. golden/test_labels.csv is unchanged.
    SEED = 5
    FEATS = ["cycles", "chemistry", "avg_temp_c", "depth_of_discharge",
             "fast_charge_pct", "cell_count", "pack_age_days",
             "dc_resistance_mohm", "soh_pct"]

    out = _pathlib.Path("regenerated")
    (out / "golden").mkdir(parents=True, exist_ok=True)
    d = make_fleet(SEED)
    tr, te = d[~d.is_test], d[d.is_test]

    tr[["pack_id"] + FEATS + ["capacity_pct"]].to_csv(out / "train.csv", index=False)
    te[["pack_id"] + FEATS].to_csv(out / "test_features.csv", index=False)
    te[["pack_id", "capacity_pct", "_true_cross"]].rename(
        columns={"_true_cross": "true_cross_cycle"}).round(
        {"capacity_pct": 2, "true_cross_cycle": 1}).to_csv(
        out / "golden" / "test_labels.csv", index=False)

    for f in ("train.csv", "test_features.csv", "golden/test_labels.csv"):
        print(f"{f:24s} sha256 {hashlib.sha256((out / f).read_bytes()).hexdigest()}")
    print()
    print(f"train  n={len(tr)} cycles {tr.cycles.min():.0f}-{tr.cycles.max():.0f}")
    print(f"field  n={len(te)} cycles {te.cycles.min():.0f}-{te.cycles.max():.0f}")
    print("field packs already past their 80% crossing: "
          f"{(te._true_cross <= te.cycles).mean() * 100:.2f}%")
