"""Data generator for battery_fade — provenance for review, not part of the solve.

Regenerates the exact CSVs this task ships. It is here so the calibration numbers quoted in
task.toml can be checked rather than taken on trust: the shipped train.csv,
test_features.csv and tests/golden/test_labels.csv are seed 0 of this generator, verified
byte-identical (sha256) to what is in the task folder.

    pip install numpy pandas scikit-learn
    python solution/generate_data.py          # summary of the frozen instance
    python solution/calibrate.py 0            # every modelling route, scored, on seed 0

Not executed by solve.sh and never present in the agent's environment — solution/ is absent
from the agent image, so shipping it cannot leak the labels.
"""
import numpy as np
import pandas as pd

CHEMS = ["lfp", "nmc", "nca"]
CHEM_FADE = {"lfp": 0.82, "nmc": 1.0, "nca": 1.14}


def make_fleet(seed=0, n_train=4200, n_test=1800):
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

    tr = block(n_train, 50, 600)
    te = block(n_test, 850, 1700)
    tr["is_test"] = False
    te["is_test"] = True
    df = pd.concat([tr, te], ignore_index=True)

    # capacity fade: sqrt in cycles, scaled by chemistry, worsened by heat, deep
    # discharge and fast charging. Temperature acts on the fade term, not additively,
    # so the interaction is real but bounded.
    chem_k = df.chemistry.map(CHEM_FADE).to_numpy()
    stress = (
        1.0
        + 0.045 * (df.avg_temp_c.to_numpy() - 28.0)
        + 0.70 * (df.depth_of_discharge.to_numpy() - 0.65)
        + 0.80 * df.fast_charge_pct.to_numpy()
    )
    fade = 1.05 * chem_k * stress * np.sqrt(df.cycles.to_numpy())
    capacity = 100.0 - fade + rng.normal(0, 0.9, len(df))
    df["capacity_pct"] = capacity.clip(20, 100).round(2)

    df.insert(0, "pack_id", [f"P{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    d = make_fleet(0)
    for name, sub in [("train", d[~d.is_test]), ("test", d[d.is_test])]:
        print(
            f"{name:6s} n={len(sub):5d} cycles {sub.cycles.min():.0f}-{sub.cycles.max():.0f} "
            f"capacity {sub.capacity_pct.min():.1f}-{sub.capacity_pct.max():.1f} "
            f"mean {sub.capacity_pct.mean():.2f}"
        )
