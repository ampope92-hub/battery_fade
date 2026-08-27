"""Regenerates the shipped battery_fade CSVs, byte for byte.

Run it beside the task (`python solution/generate_data.py`) and compare the printed sha256s
against the ones pinned in tests/test_outputs.py — that is the audit trail for the data.
Needs numpy and pandas; it never runs inside the task container, and it lives in solution/,
which is not part of the agent image, so shipping it cannot leak anything.

Each pack has its own fade rate

    rate_i = base(x_i) * exp(eta_i),     eta_i ~ N(0, sigma(x_i)^2)

and capacity at cycle c is 100 - rate_i*sqrt(c). Two things follow, and the task grades both.

The rate itself is driven by chemistry, heat, depth of discharge and fast-charge share, all
of which MULTIPLY it rather than shifting it, so log(rate) is linear in the drivers. And the
SPREAD of that per-pack effect is itself a function of duty: aggressive use does not only
fade a pack faster on average, it makes it fade less predictably, because cell-to-cell
imbalance and local hot spots grow with heat and fast charging. So sigma rises with
fast_charge_pct and temperature.

That second property is what makes a replacement schedule a real modelling problem rather
than arithmetic. With only i.i.d. measurement noise a pack's future capacity would be a
deterministic function of its features — there would be no predictive distribution, hence no
margin to size, and any constant safety factor would do.
"""
import numpy as np
import pandas as pd

CHEMS = ["lfp", "nmc", "nca"]
CHEM_FADE = {"lfp": 0.82, "nmc": 1.0, "nca": 1.14}

# Scale of the per-pack fade-rate effect. Log-linear in duty: hard use multiplies the
# spread rather than adding to it, the same way it multiplies the fade rate itself. A gently
# used pack is nearly deterministic; a hard-charged hot one is not.
SIGMA_BASE = 0.065
SIGMA_F = 2.00      # per unit fast_charge_pct (beta(2,5), mean 0.29)
SIGMA_T = 0.70      # per 10 C above 28
SIGMA_CLIP = (0.055, 0.45)   # loose -- it should almost never bind

FADE_K = 0.26       # overall fade scale. v1 used 1.05, which crossed 80% inside the TRAIN
                    # range, so every field pack was already past it and a recommendation was moot.
MEAS_SD = 0.08      # bench measurement error. Must stay small next to the pack effect:
                    # a teardown rate estimate is (100-cap)/sqrt(cycles), so at low cycles
                    # measurement error is divided by a small number and swamps eta.
THRESHOLD = 80.0    # capacity_pct defining end of service life


def sigma_of(fast_charge_pct, avg_temp_c):
    lg = (
        np.log(SIGMA_BASE)
        + SIGMA_F * np.asarray(fast_charge_pct)
        + SIGMA_T * np.maximum(0.0, np.asarray(avg_temp_c) - 28.0) / 10.0
    )
    return np.clip(np.exp(lg), *SIGMA_CLIP)


def base_rate(chemistry, avg_temp_c, depth_of_discharge, fast_charge_pct):
    """Median fade rate. Stress enters MULTIPLICATIVELY, exponential in temperature the way
    an Arrhenius rate is, so log(rate) is exactly linear in the drivers. That matters for
    more than realism: with stress entering linearly (as in v1) log(rate) is the log of a
    linear function, every log-scale fit is misspecified, and the bias leaks into the
    per-stratum spread estimate that axis 2 grades."""
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
        return pd.DataFrame(
            {
                "cycles": cycles.round(0),
                "chemistry": chem,
                "avg_temp_c": avg_temp_c.round(1),
                "depth_of_discharge": depth_of_discharge.round(3),
                "fast_charge_pct": fast_charge_pct.round(3),
                "cell_count": cell_count,
                "pack_age_days": pack_age_days.clip(30, None).round(0),
            }
        )

    tr, te = block(n_train, 150, 600), block(n_test, 850, 1700)
    tr["is_test"], te["is_test"] = False, True
    df = pd.concat([tr, te], ignore_index=True)

    base = base_rate(df.chemistry, df.avg_temp_c, df.depth_of_discharge, df.fast_charge_pct)
    sig = sigma_of(df.fast_charge_pct, df.avg_temp_c)
    eta = rng.normal(0.0, sig)
    rate = base * np.exp(eta)

    df["_base_rate"] = base
    df["_sigma"] = sig
    df["_rate"] = rate

    cap_true = 100.0 - rate * np.sqrt(df.cycles.to_numpy())
    df["capacity_pct"] = (cap_true + rng.normal(0, MEAS_SD, len(df))).clip(20, 100).round(2)

    # The cycle count at which this pack actually crosses THRESHOLD. Deterministic given the
    # pack's own rate, unknown to anyone who has not seen it fail. This is axis 2's truth.
    df["_true_cross"] = (( 100.0 - THRESHOLD) / rate) ** 2

    df.insert(0, "pack_id", [f"P{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    import hashlib
    import pathlib as _pathlib

    # The shipped instance. Seed 5 was chosen as the TIGHTEST of the 14 swept, not the most
    # comfortable one: the reference route sits 3.9 binomial SD inside the nearest band edge
    # there, so the bars are not resting on a lucky draw.
    SEED = 5
    FEATS = ["cycles", "chemistry", "avg_temp_c", "depth_of_discharge",
             "fast_charge_pct", "cell_count", "pack_age_days"]

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
