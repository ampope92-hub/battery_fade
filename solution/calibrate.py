"""battery_fade — route sweep and bar validation. Run it beside the task:

    python solution/calibrate.py            # 14 seeds
    python solution/calibrate.py 5          # just the shipped instance

It never runs inside the task container; it reads only generate_data.py, so a reviewer can
re-derive every number in task.toml rather than take it on trust. Needs numpy, pandas and
(for the tree arms) scikit-learn.

Does the provenance trap actually deceive?

Three things have to hold at once, and the third is the one v2 never had:

  1. the HONEST route still clears both bars (the ceiling must not move);
  2. every LEAK route fails at least one bar decisively;
  3. an honest holdout built on train.csv alone RANKS THE LEAK ROUTES FIRST.

(3) is the whole design. The v2 trials all built a low-cycle -> high-cycle split inside
train and it told them the truth, so they converged. Here that same split is contaminated
in every row, so it recommends the trap.
"""
import sys

import generate_data as G
import numpy as np
import pandas as pd

THRESH, Z90 = 80.0, 1.2815515655446004
DROP = 100.0 - THRESH

HONEST = ["temp", "dod", "fast", "lfp", "nca"]


def design(df, extra=()):
    X = pd.DataFrame({
        "temp": df.avg_temp_c - 28.0,
        "dod": df.depth_of_discharge - 0.65,
        "fast": df.fast_charge_pct,
        "lfp": (df.chemistry == "lfp").astype(float),
        "nca": (df.chemistry == "nca").astype(float),
    })
    for e in extra:
        if e == "res":
            X["res"] = (df.dc_resistance_mohm - 50.0) / 10.0
        elif e == "soh":
            X["soh"] = (df.soh_pct - 90.0) / 10.0
    X.insert(0, "one", 1.0)
    return X.to_numpy()


def fit_mean(tr, extra=()):
    rate = np.clip((100.0 - tr.capacity_pct.to_numpy()) / np.sqrt(tr.cycles.to_numpy()), 1e-6, None)
    y = np.log(rate)
    X = design(tr, extra)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def duty_design(df):
    return np.column_stack([np.ones(len(df)), df.fast_charge_pct.to_numpy(),
                            np.maximum(0.0, df.avg_temp_c.to_numpy() - 28.0) / 10.0])


def fit_sigma(tr, resid):
    Z = duty_design(tr); g = np.zeros(Z.shape[1]); g[0] = np.log(resid.std())
    for _ in range(200):
        w = resid**2 * np.exp(-2.0 * (Z @ g))
        grad = 2.0 * Z.T @ (1.0 - w)
        H = 4.0 * (Z * w[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, grad); g -= step
        if np.max(np.abs(step)) < 1e-11: break
    return g


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def metrics(te, rec):
    late = rec > te._true_cross.to_numpy()
    band = pd.qcut(te.fast_charge_pct, 3, labels=["gentle", "mid", "hard"])
    per = {b: late[(band == b).to_numpy()].mean() * 100 for b in ["gentle", "mid", "hard"]}
    return late.mean() * 100, per


def _rate_route(tr, te, extra):
    """log-rate OLS on the given feature set, then the hetero-MLE schedule."""
    beta, resid = fit_mean(tr, extra)
    base_hat = np.exp(design(te, extra) @ beta)
    cap = 100.0 - base_hat * np.sqrt(te.cycles.to_numpy())
    s = np.exp(duty_design(te) @ fit_sigma(tr, resid))
    return cap, (DROP / (base_hat * np.exp(Z90 * s))) ** 2


def routes(tr, te):
    out = {}
    out["1. honest log-rate + hetero MLE"] = _rate_route(tr, te, ())
    out["2. leak: soh_pct shipped directly"] = (
        te.soh_pct.to_numpy(),
        (DROP / (np.clip((100.0 - te.soh_pct.to_numpy()), 0.05, None)
                 / np.sqrt(te.cycles.to_numpy()) * np.exp(Z90 * 0.15))) ** 2)
    out["3. leak: soh in the rate model"] = _rate_route(tr, te, ("soh",))
    out["4. leak: resistance in the rate model"] = _rate_route(tr, te, ("res",))
    out["5. leak: both in the rate model"] = _rate_route(tr, te, ("res", "soh"))

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        def tab(df, cols):
            d = df[cols].copy()
            d["chem"] = pd.Categorical(df.chemistry, G.CHEMS).codes
            return d.to_numpy(float)
        base = ["cycles", "avg_temp_c", "depth_of_discharge", "fast_charge_pct",
                "cell_count", "pack_age_days"]
        for nm, cols in [("6. GBM honest cols", base),
                         ("7. GBM kitchen sink (leaks in)",
                          base + ["dc_resistance_mohm", "soh_pct"])]:
            m = HGB(random_state=0).fit(tab(tr, cols), tr.capacity_pct.to_numpy())
            cap = m.predict(tab(te, cols))
            rt = np.clip((100.0 - cap) / np.sqrt(te.cycles.to_numpy()), 1e-6, None)
            out[nm] = (cap, (DROP / (rt * np.exp(Z90 * 0.15))) ** 2)
    except ImportError:
        pass
    return out


def holdout_rank(tr):
    """What an HONEST validation built on train.csv alone reports.

    Fit on the low-cycle half, score on the high-cycle half -- exactly the split the v2
    trials built to guard against the extrapolation trap. If the trap works, this ranks the
    leak routes ABOVE the honest one.
    """
    cut = tr.cycles.median()
    a, b = tr[tr.cycles <= cut].copy(), tr[tr.cycles > cut].copy()
    y = b.capacity_pct.to_numpy()
    res = {}
    for nm, extra in [("honest", ()), ("+soh", ("soh",)), ("+res", ("res",)),
                      ("+both", ("res", "soh"))]:
        beta, _ = fit_mean(a, extra)
        cap = 100.0 - np.exp(design(b, extra) @ beta) * np.sqrt(b.cycles.to_numpy())
        res[nm] = r2(y, cap)
    res["soh direct"] = r2(y, b.soh_pct.to_numpy())
    return res




# ---------------------------------------------------------------------------------------
# Per-seed pass/fail with binomial SD margins. A band edge closer than ~3 SD to any route is
# not a bar, it is a coin flip. Coverage is a binomial count over 2000 packs per tercile, so
# SD = sqrt(p(1-p)/n) ~ 0.67pp at p=0.10.
# ---------------------------------------------------------------------------------------
R2_BAR = 0.55
FLEET = (7.0, 13.5)
BAND = (6.0, 14.5)


def extra_adversaries(tr, te):
    """The attacks fleet_failures learned to grant: tuning on withheld truth, and
    blend/smoothing rescues that can launder a leak back into a usable signal."""
    out = {}
    beta, resid = fit_mean(tr, ())
    base_hat = np.exp(design(te, ()) @ beta)
    pt = (DROP / base_hat) ** 2
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if metrics(te, pt * mid)[0] > 10.0: hi = mid
        else: lo = mid
    cap_h = 100.0 - base_hat * np.sqrt(te.cycles.to_numpy())
    out["8. scaled mean TUNED ON TRUTH"] = (cap_h, pt * (lo + hi) / 2)

    # blend the honest prediction with the leak columns -- the "keep a bit of it" rescue
    bres, _ = fit_mean(tr, ("res",))
    cap_r = 100.0 - np.exp(design(te, ("res",)) @ bres) * np.sqrt(te.cycles.to_numpy())
    for w in (0.25, 0.5):
        cap = (1 - w) * cap_h + w * cap_r
        rt = np.clip((100.0 - cap) / np.sqrt(te.cycles.to_numpy()), 1e-6, None)
        s = np.exp(duty_design(te) @ fit_sigma(tr, resid))
        out[f"9. blend honest/res w={w}"] = (cap, (DROP / (rt * np.exp(Z90 * s))) ** 2)
    cap_s = 0.5 * cap_h + 0.5 * te.soh_pct.to_numpy()
    rt = np.clip((100.0 - cap_s) / np.sqrt(te.cycles.to_numpy()), 1e-6, None)
    out["10. blend honest/soh w=0.5"] = (
        cap_s, (DROP / (rt * np.exp(Z90 * np.exp(duty_design(te) @ fit_sigma(tr, resid))))) ** 2)
    return out


def run(seed):
    d = G.make_fleet(seed)
    tr, te = d[~d.is_test].copy(), d[d.is_test].copy()
    y = te.capacity_pct.to_numpy()
    rr = routes(tr, te); rr.update(extra_adversaries(tr, te))
    rows = []
    for nm, (cap, rec) in rr.items():
        fleet, per = metrics(te, np.clip(rec, 1, 1e7))
        ok = (r2(y, cap) >= R2_BAR and FLEET[0] <= fleet <= FLEET[1]
              and all(BAND[0] <= v <= BAND[1] for v in per.values()))
        rows.append(dict(seed=seed, route=nm, R2=r2(y, cap), fleet=fleet, **per, PASS=ok))
    return rows, holdout_rank(tr)


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or list(range(14))
    allrows, holds = [], []
    for s in seeds:
        r, h = run(s); allrows += r; holds.append(h)
    df = pd.DataFrame(allrows); pd.set_option("display.width", 240)
    g = df.groupby("route").agg(
        R2_min=("R2", "min"), R2_max=("R2", "max"),
        fleet_min=("fleet", "min"), fleet_max=("fleet", "max"),
        worst_band=("gentle", "min"), n_pass=("PASS", "sum"), n=("PASS", "size"))
    # worst band across all three, per route
    g["band_lo"] = df.groupby("route")[["gentle", "mid", "hard"]].min().min(axis=1)
    g["band_hi"] = df.groupby("route")[["gentle", "mid", "hard"]].max().max(axis=1)
    g = g.drop(columns=["worst_band"])
    print(f"=== {len(seeds)} seeds — bars R2>={R2_BAR}, fleet {FLEET}, thirds {BAND} ===")
    print(g.to_string(float_format=lambda v: f"{v:7.3f}"))
    sd = np.sqrt(0.10 * 0.90 / 2000) * 100
    print(f"\nbinomial SD per tercile (n=2000, p=0.10): {sd:.2f} pp")
    h = df[df.route.str.startswith("1.")]
    print(f"honest route: R2 {h.R2.min():.3f}-{h.R2.max():.3f}  "
          f"fleet {h.fleet.min():.2f}-{h.fleet.max():.2f}  "
          f"worst band {df[df.route.str.startswith('1.')][['gentle','mid','hard']].min().min():.2f}"
          f"-{df[df.route.str.startswith('1.')][['gentle','mid','hard']].max().max():.2f}")
    print(f"  margin to nearest band edge: "
          f"{(df[df.route.str.startswith('1.')][['gentle','mid','hard']].min().min()-BAND[0])/sd:.1f} SD low, "
          f"{(BAND[1]-df[df.route.str.startswith('1.')][['gentle','mid','hard']].max().max())/sd:.1f} SD high")
    print("\n=== TRAIN-ONLY HONEST HOLDOUT (mean over seeds) ===")
    print(pd.DataFrame(holds).mean().to_string(float_format=lambda v: f"{v:.4f}"))
