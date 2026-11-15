"""
me_sim.py - when does the error-variance bound (1) and regression calibration work?

Each patient i has a true level mu_i ~ N(0, 1) and an exact covariate Z_i (corr 0.4 with mu_i). n_i measurements
x_it = mu_i + s_i e_it are taken, where e_it is a stationary AR(1) series with lag-1 correlation rho and unit
variance, either Gaussian or skewed (standardised log-normal, shape 0.75), and s_i = 1 (constant spread) or
exp(0.4 mu_i)/c (spread increasing with level). The number of measurements is n0 (fixed), n0 exp(0.5 mu_i)
(sicker patients measured more) or n0 (1 + Y_i) (outcome-dependent measurement frequency).
The true covariate is X*_i = mean of the n_i measurements (as in PhysioNet), the outcome is
logit P(Y=1) = a + 0.5 X*_i/sd(X*) + 0.3 Z_i (prevalence about 15%), and the analyst sees min, max and n_i.
Methods: oracle, mid-point Wald, regression calibration with kappa = 1, HPPSM (kappa in {0, 0.5, 1}), HPPSM
with kappa up to 2, and HPPSM with the AR(1)-adjusted bound, kappa in [0, (1 + rho)/(1 - rho)]. Estimand: the coefficient of X*.
"""
import os, sys, itertools, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hppsm_v3 import logit_fit, regression_calibration, error_variance_bound, Q95

N = 3000; BETA = 0.5


def series(rng, n, nmax, rho, dist):
    e = np.empty((n, nmax)); e[:, 0] = rng.standard_normal(n)
    for t in range(1, nmax):
        e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    if dist == "skewed":
        s = 0.75; e = (np.exp(s * e) - np.exp(s ** 2 / 2)) / np.sqrt((np.exp(s ** 2) - 1) * np.exp(s ** 2))
    return e


def one_rep(rng, rho, n0, dist, spread, freq):
    mu = rng.standard_normal(N); Z = 0.4 * mu + np.sqrt(1 - 0.16) * rng.standard_normal(N)
    s = np.ones(N) if spread == "constant" else np.exp(0.4 * mu) / np.exp(0.08)
    y0 = None
    if freq == "fixed":
        n = np.full(N, n0)
    elif freq == "level":
        n = np.clip(np.round(n0 * np.exp(0.5 * mu)), 3, 4 * n0).astype(int)
    nmax_guess = 2 * n0 if freq == "outcome" else int(n.max()) if freq != "outcome" else 2 * n0
    e = series(rng, N, max(nmax_guess, 3), rho, dist)
    x = mu[:, None] + s[:, None] * e
    if freq == "outcome":
        # outcome generated from the mean of the first n0 measurements, then patients who die get n0 more
        xs0 = x[:, :n0].mean(1); lp0 = BETA * (xs0 - xs0.mean()) / xs0.std() + 0.3 * Z
        a = np.log(0.15 / 0.85) - 0.1; y0 = rng.uniform(size=N) < 1 / (1 + np.exp(-(a + lp0)))
        n = np.where(y0, 2 * n0, n0)
    mask = np.arange(x.shape[1])[None, :] < n[:, None]
    xm = np.where(mask, x, np.nan)
    Xs = np.nanmean(xm, 1); lo = np.nanmin(xm, 1); hi = np.nanmax(xm, 1); W = (lo + hi) / 2
    if y0 is None:
        lp = BETA * (Xs - Xs.mean()) / Xs.std() + 0.3 * Z
        a = np.log(0.15 / 0.85) - 0.1
        y = (rng.uniform(size=N) < 1 / (1 + np.exp(-(a + lp)))).astype(float)
        truth = BETA / Xs.std()
    else:
        y = y0.astype(float); truth = BETA / xs0.std()      # outcome depends on the first n0 measurements
        Xs = xs0                                             # target covariate: mean over the base window
    b2 = error_variance_bound(lo, hi, n); U = W - Xs
    Zc = Z[:, None]
    res = {}
    b, se, _ = logit_fit(y, np.column_stack([Xs, Z])); res["Oracle"] = (b[0], b[0] - Q95 * se[0], b[0] + Q95 * se[0])
    b, se, _ = logit_fit(y, np.column_stack([W, Z])); res["Mid-point Wald"] = (b[0], b[0] - Q95 * se[0], b[0] + Q95 * se[0])
    ints = {}
    for k in (0.0, 0.5, 1.0, 1.5, 2.0):
        Xc = regression_calibration(W[:, None], Zc, np.array([k * b2.mean()]))[:, 0] if k > 0 else W
        b, se, _ = logit_fit(y, np.column_stack([Xc, Z])); ints[k] = (b[0], b[0] - Q95 * se[0], b[0] + Q95 * se[0])
    res["Regression calibration (kappa = 1)"] = ints[1.0]
    res["HPPSM (kappa in [0, 1])"] = (ints[0][0], min(ints[k][1] for k in (0, .5, 1)), max(ints[k][2] for k in (0, .5, 1)))
    res["HPPSM (kappa in [0, 2])"] = (ints[0][0], min(v[1] for v in ints.values()), max(v[2] for v in ints.values()))
    de = (1 + rho) / (1 - rho)                     # AR(1) design effect for a mean: bound (1) x de
    for k in (0.5 * de, de):
        Xc = regression_calibration(W[:, None], Zc, np.array([k * b2.mean()]))[:, 0]
        b, se, _ = logit_fit(y, np.column_stack([Xc, Z])); ints[("de", k)] = (b[0], b[0] - Q95 * se[0], b[0] + Q95 * se[0])
    keys = [0.0, ("de", 0.5 * de), ("de", de)]
    res["HPPSM (AR(1)-adjusted bound)"] = (ints[0][0], min(ints[k][1] for k in keys), max(ints[k][2] for k in keys))
    diag = dict(ratio_EU2_bound=float(np.mean(U ** 2) / np.mean(b2)), meanU_sdX=float(U.mean() / Xs.std()),
                corrUX=float(np.corrcoef(U, Xs)[0, 1]), reliability=float(1 - U.var() / W.var()))
    return res, truth, diag


def main(reps=250, out="results_me"):
    os.makedirs(out, exist_ok=True)
    path = f"{out}/me_sim.csv"
    done = set(map(tuple, pd.read_csv(path)[["rho", "n0", "dist", "spread", "freq"]].drop_duplicates().values)) if os.path.exists(path) else set()
    grid = list(itertools.product((0.0, 0.5, 0.9), (6, 24, 72), ("normal", "skewed"), ("constant", "level"), ("fixed", "level", "outcome")))
    for gi, g in enumerate(grid):
        if (g[0], g[1], g[2], g[3], g[4]) in done:
            continue
        rho, n0, dist, spread, freq = g
        rng = np.random.default_rng(880_000 + gi); rows = []; t0 = time.time()
        for r in range(reps):
            res, truth, diag = one_rep(rng, rho, n0, dist, spread, freq)
            for k, (e, l, h) in res.items():
                rows.append(dict(rho=rho, n0=n0, dist=dist, spread=spread, freq=freq, rep=r, method=k, est=e, lo=l, hi=h,
                                 truth=truth, **diag))
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(g, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
