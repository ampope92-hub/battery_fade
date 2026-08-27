"""Scores every route this task can be attempted by, against the bars the verifier applies.

`python solution/calibrate.py` sweeps the default 14 generator seeds; pass seeds as arguments
to look at one. Needs numpy and pandas, and never runs inside the task container.

What it is for: the bars in tests/test_outputs.py are only defensible if the correct routes
clear them and the plausible wrong ones do not, on more than the one seed that shipped. Each
route is scored on BOTH graded axes, so the interaction is visible — every route here gets
the capacity column equally right, and they separate only on the replacement schedule.

Routes A and B are correct and deliberately unlike each other: A models the spread
parametrically by maximum likelihood, B never assumes a shape and reads an empirical
quantile off duty bins. C is the near-miss the whole second axis exists to catch — the right
mean model with the spread treated as one pooled number. D schedules off the median crossing
with no margin at all. E and F are the gaming attacks: both are tuned on the WITHHELD truth
until the fleet-wide count lands exactly on the target, which is why only the stratified
bands reject them.
"""
import sys

import generate_data as G
import numpy as np
import pandas as pd

THRESHOLD, DROP = 80.0, 20.0
Z90 = 1.2815515655446004

R2_BAR = 0.55
FLEET = (7.0, 13.5)     # fleet-wide late rate, %
BAND = (6.0, 14.5)      # per-tercile late rate, %


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def design(df):
    """Drivers of the log fade rate — the same information the agent is given."""
    X = pd.DataFrame({
        "temp": df.avg_temp_c - 28.0,
        "dod": df.depth_of_discharge - 0.65,
        "fast": df.fast_charge_pct,
        "lfp": (df.chemistry == "lfp").astype(float),
        "nca": (df.chemistry == "nca").astype(float),
    })
    X.insert(0, "one", 1.0)
    return X.to_numpy()


def duty_design(df):
    """Drivers of the pack-to-pack spread around that rate."""
    return np.column_stack([
        np.ones(len(df)),
        df.fast_charge_pct.to_numpy(),
        np.maximum(0.0, df.avg_temp_c.to_numpy() - 28.0) / 10.0,
    ])


def fit_mean(tr):
    rate = np.clip((100.0 - tr.capacity_pct.to_numpy()) / np.sqrt(tr.cycles.to_numpy()), 1e-6, None)
    y = np.log(rate)
    X = design(tr)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def fit_log_sigma(tr, resid):
    """MLE for log(sigma) = Z@g under resid ~ N(0, sigma_i^2). Newton on
    sum(2*Z@g + resid^2 * exp(-2*Z@g)).

    Regressing log|resid| on duty is the obvious first pass and it biases the gentle end:
    a rate estimate is (100-capacity)/sqrt(cycles), so bench error is divided by a small
    number at low cycles and inflates the apparent spread exactly where it is smallest.
    """
    Z = duty_design(tr)
    g = np.zeros(Z.shape[1])
    g[0] = np.log(resid.std())
    for _ in range(200):
        w = resid ** 2 * np.exp(-2.0 * (Z @ g))
        grad = 2.0 * Z.T @ (1.0 - w)
        H = 4.0 * (Z * w[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, grad)
        g -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return g


def bands(te):
    """Exactly the rule tests/test_outputs.py applies: order by (fast_charge_pct, pack_id),
    cut into three equal counts. Tie-safe and not steerable by row order."""
    order = np.lexsort((te.pack_id.to_numpy(), te.fast_charge_pct.to_numpy()))
    lab = np.empty(len(te), dtype=object)
    n = len(order)
    for rank, idx in enumerate(order):
        lab[idx] = ("lowest", "middle", "highest")[min(rank * 3 // n, 2)]
    return lab


def late_rates(te, rec, lab):
    late = np.asarray(rec, dtype=float) > te._true_cross.to_numpy()
    return late.mean() * 100, {b: late[lab == b].mean() * 100 for b in ("lowest", "middle", "highest")}


def routes(tr, te, lab):
    beta, resid = fit_mean(tr)
    base = np.exp(design(te) @ beta)
    cap = 100.0 - base * np.sqrt(te.cycles.to_numpy())
    out = {}

    s_hat = np.exp(duty_design(te) @ fit_log_sigma(tr, resid))
    out["A. duty-dependent spread, MLE (reference)"] = (cap, (DROP / (base * np.exp(Z90 * s_hat))) ** 2)

    edges = np.quantile(tr.fast_charge_pct.to_numpy(), np.linspace(0, 1, 7)[1:-1])
    btr = np.digitize(tr.fast_charge_pct.to_numpy(), edges)
    bte = np.digitize(te.fast_charge_pct.to_numpy(), edges)
    emp = np.array([np.quantile(resid[btr == b], 0.90) for b in range(len(edges) + 1)])
    out["B. empirical duty bins (correct)"] = (cap, (DROP / (base * np.exp(emp[bte]))) ** 2)

    out["C. pooled spread (near miss)"] = (cap, (DROP / (base * np.exp(Z90 * resid.std()))) ** 2)
    out["D. median crossing, no margin"] = (cap, (DROP / base) ** 2)

    def tuned(seed_rec, lo, hi):
        """Scale until the FLEET-wide count lands on the 10% target, using the truth."""
        for _ in range(60):
            mid = (lo + hi) / 2
            if late_rates(te, seed_rec * mid if np.ndim(seed_rec) else np.full(len(te), mid), lab)[0] > 10.0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    pt = (DROP / base) ** 2
    out["E. scaled mean, tuned on truth"] = (cap, pt * tuned(pt, 0.01, 3.0))
    out["F. one number, tuned on truth"] = (cap, np.full(len(te), tuned(1.0, 100.0, 40000.0)))
    return out


def run(seed):
    d = G.make_fleet(seed)
    tr, te = d[~d.is_test].copy(), d[d.is_test].copy()
    lab = bands(te)
    y = te.capacity_pct.to_numpy()
    rows = []
    for name, (cap, rec) in routes(tr, te, lab).items():
        fleet, per = late_rates(te, rec, lab)
        rows.append(dict(
            seed=seed, route=name, r2=r2(y, cap), fleet=fleet, **per,
            passes=(r2(y, cap) >= R2_BAR and FLEET[0] <= fleet <= FLEET[1]
                    and all(BAND[0] <= v <= BAND[1] for v in per.values())),
        ))
    return rows


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or list(range(14))
    df = pd.DataFrame([r for s in seeds for r in run(s)])
    pd.set_option("display.width", 220)
    print(f"bars: R2 >= {R2_BAR}   fleet late {FLEET}%   per-third late {BAND}%\n")
    print(df.groupby("route").agg(
        r2_min=("r2", "min"), r2_max=("r2", "max"),
        fleet_min=("fleet", "min"), fleet_max=("fleet", "max"),
        lowest_min=("lowest", "min"), lowest_max=("lowest", "max"),
        highest_min=("highest", "min"), highest_max=("highest", "max"),
        n_pass=("passes", "sum"), n=("passes", "size"),
    ).to_string(float_format=lambda v: f"{v:.2f}"))
    sd = np.sqrt(0.10 * 0.90 / (len(df) and 2000)) * 100
    print(f"\nbinomial SD per third at p=10%, n=2000: {sd:.2f} pp — a bar closer than about"
          f"\n3 SD to any route it has to separate is a coin flip, not a bar.")
