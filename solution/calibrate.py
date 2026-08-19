"""Calibration sweep for battery_fade — provenance for review, not part of the solve.

Scores every modelling route against the held-out packs, which is where the numbers in
task.toml's difficulty_explanation come from: the best wrong route reaches R2 0.8217, the
weakest correct one 0.9242, and the 0.88 bar was frozen between them against the tightest of
six seeds. Run it on any seed to reproduce that separation.

    python solution/calibrate.py 0

Not executed by solve.sh and never present in the agent's environment.
"""
import sys

import numpy as np
import pandas as pd
from generate_data import make_fleet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE = ["cycles", "avg_temp_c", "depth_of_discharge", "fast_charge_pct",
        "cell_count", "pack_age_days"]


def dummies(df):
    return pd.get_dummies(df.chemistry, prefix="chem").astype(float)


def raw(df):
    return pd.concat([df[BASE].reset_index(drop=True), dummies(df).reset_index(drop=True)], axis=1)


def sqrt_feats(df, interactions):
    """sqrt(cycles) in place of cycles; optionally scaled by the stress drivers."""
    d = df.reset_index(drop=True)
    s = np.sqrt(d.cycles.to_numpy())
    out = pd.DataFrame({"sqrt_cycles": s})
    for c in ["avg_temp_c", "depth_of_discharge", "fast_charge_pct", "cell_count", "pack_age_days"]:
        out[c] = d[c].to_numpy()
    ch = dummies(d)
    for c in ch.columns:
        out[c] = ch[c].to_numpy()
    if interactions == "chem":
        for c in ch.columns:
            out[f"s_{c}"] = s * ch[c].to_numpy()
        return out
    if interactions:
        for c in ch.columns:
            out[f"s_{c}"] = s * ch[c].to_numpy()
        out["s_temp"] = s * d.avg_temp_c.to_numpy()
        out["s_dod"] = s * d.depth_of_discharge.to_numpy()
        out["s_fast"] = s * d.fast_charge_pct.to_numpy()
    return out


def log_feats(df):
    d = df.reset_index(drop=True)
    out = raw(d).copy()
    out["cycles"] = np.log(d.cycles.to_numpy())
    return out


def run(name, xtr, ytr, xte, yte, model):
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    print(f"  {name:<42s} R2 {r2_score(yte, pred):8.4f}   RMSE {np.sqrt(((yte-pred)**2).mean()):6.2f}")
    return r2_score(yte, pred)


def main(seed):
    df = make_fleet(seed)
    tr, te = df[~df.is_test], df[df.is_test]
    ytr, yte = tr.capacity_pct.to_numpy(), te.capacity_pct.to_numpy()
    print(f"seed {seed}: train {len(tr)}, test {len(te)}")

    print(" WRONG — cannot leave the training range")
    wrong = [
        run("HistGradientBoosting (defaults)", raw(tr), ytr, raw(te), yte,
            HistGradientBoostingRegressor(random_state=0)),
        run("HistGradientBoosting (tuned deep)", raw(tr), ytr, raw(te), yte,
            HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                          max_leaf_nodes=63, random_state=0)),
        run("RandomForest", raw(tr), ytr, raw(te), yte,
            RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
        run("KNN (k=10)", raw(tr), ytr, raw(te), yte,
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))),
        run("GBM on sqrt-transformed cycles", sqrt_feats(tr, False), ytr,
            sqrt_feats(te, False), yte, HistGradientBoostingRegressor(random_state=0)),
    ]

    print(" PARTIAL — extrapolates, wrong functional form")
    partial = [
        run("linear, raw cycles", raw(tr), ytr, raw(te), yte, LinearRegression()),
        run("ridge, raw cycles", raw(tr), ytr, raw(te), yte,
            make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        run("poly2 on raw features", raw(tr), ytr, raw(te), yte,
            make_pipeline(PolynomialFeatures(2), LinearRegression())),
        run("0.5*linear + 0.5*GBM blend", raw(tr), ytr, raw(te), yte, Blend()),
    ]

    print(" CORRECT — sqrt/log form, extrapolates with the right slope")
    correct = [
        run("linear on sqrt(cycles)", sqrt_feats(tr, False), ytr,
            sqrt_feats(te, False), yte, LinearRegression()),
        run("linear on log(cycles)", log_feats(tr), ytr, log_feats(te), yte, LinearRegression()),
        run("linear on sqrt(cycles) + interactions", sqrt_feats(tr, True), ytr,
            sqrt_feats(te, True), yte, LinearRegression()),
    ]
    print(" ALTERNATIVE CORRECT ROUTES")
    alt = [
        run("log(fade) ~ log(cycles) + covariates", raw(tr), ytr, raw(te), yte, LogFade()),
        run("linear on sqrt, chemistry interaction only", sqrt_feats(tr, "chem"), ytr,
            sqrt_feats(te, "chem"), yte, LinearRegression()),
        run("linear on sqrt+inter, high-cycle half only", sqrt_feats(tr[tr.cycles > 325], True),
            tr[tr.cycles > 325].capacity_pct.to_numpy(), sqrt_feats(te, True), yte,
            LinearRegression()),
    ]
    print(f"\n  alt min     {min(alt):.4f}")
    print(f"\n  wrong max   {max(wrong):.4f}")
    print(f"  partial max {max(partial):.4f}")
    print(f"  correct min {min(correct):.4f}  max {max(correct):.4f}")


class LogFade:
    """Fade is multiplicative, so regress log(100 - capacity) on log(cycles)."""

    def fit(self, x, y):
        self.x_cols = list(x.columns)
        xx = x.copy()
        xx["cycles"] = np.log(xx["cycles"].to_numpy())
        self.m = LinearRegression().fit(xx, np.log(np.clip(100.0 - y, 0.01, None)))
        return self

    def predict(self, x):
        xx = x.copy()
        xx["cycles"] = np.log(xx["cycles"].to_numpy())
        return 100.0 - np.exp(self.m.predict(xx))


class Blend:
    """A plausible hedge: average a linear model and a boosted tree."""

    def fit(self, x, y):
        self.a = LinearRegression().fit(x, y)
        self.b = HistGradientBoostingRegressor(random_state=0).fit(x, y)
        return self

    def predict(self, x):
        return 0.5 * self.a.predict(x) + 0.5 * self.b.predict(x)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
