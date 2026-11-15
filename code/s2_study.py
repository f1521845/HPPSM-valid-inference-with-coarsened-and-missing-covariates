"""
s2_study.py - SUPPORT2 plasmode simulation for Draft 7.

Scenarios (all with 3,000 patients unless stated; replicate r uses seed 10_000 + r for the data):
  main       M1-M4 (as in the paper), 1,000 replicates
  extended   n500, n1000            : smaller samples (M1)
             misslow, misshigh      : missing rates x0.5 and x1.5 (M1)
             self05, self10, self20 : self-masking MNAR - missingness also depends on the variable's own
                                      true value with strength lambda = 0.5, 1, 2 (logit per SD), on top of M1
             quad                   : outcome model misspecified (true log hazard has a quadratic BUN term;
                                      the analysis model is linear); estimand = pseudo-true coefficients
  dense      verification of the finite-configuration union against a dense search (M1-M4)
Methods: Oracle, Fill-in Wald, Complete case, MI (chained equations), MI (SMC-FCS),
         HPPSM v2 (2q+3 configurations; as in Draft 6), HPPSM v2 (adaptive vertices; primary),
         and in extended scenarios HPPSM with dmax = 0.25 and 1.0.
"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hppsm_v3 import (RoystonParmarPH, rp_basis, rubin, smcfcs_rp, Envelope, Q95)
import jax.numpy as jnp
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

LABS = pd.DataFrame([
    ("alb", "id", 1.0, 5.5, 3.5), ("pafi", "id", 50.0, 600.0, 333.3), ("bili", "log", 0.1, 30.0, 1.01),
    ("crea", "log", 0.3, 12.0, 1.01), ("bun", "log", 3.0, 150.0, 6.51), ("wblc", "log", 0.5, 60.0, 9.0),
    ("urine", "log1p", 0.0, 8000., 2502.), ("ph", "id", 6.9, 7.7, 7.40), ("glucose", "log", 40.0, 600.0, 100.0),
], columns=["var", "tf", "lo", "hi", "fill"]).set_index("var")
TF = {"id": lambda v: v, "log": np.log, "log1p": np.log1p}
TERMS_X = ["bun", "crea", "alb", "glucose"]; TERMS_Z = ["age", "scoma"]; TERMS = TERMS_X + TERMS_Z
LO_SUP = np.array([TF[LABS.loc[v, "tf"]](LABS.loc[v, "lo"]) for v in LABS.index])
HI_SUP = np.array([TF[LABS.loc[v, "tf"]](LABS.loc[v, "hi"]) for v in LABS.index])
MECH = {"M1": (0.75, False), "M2": (1.5, False), "M3": (0.0, False), "M4": (1.0, True)}
EXT = {  # name: (gamma, ydep, n, miss_scale, self_lambda, quad)
    "n500": (0.75, False, 500, 1.0, 0.0, 0.0), "n1000": (0.75, False, 1000, 1.0, 0.0, 0.0),
    "misslow": (0.75, False, 3000, 0.5, 0.0, 0.0), "misshigh": (0.75, False, 3000, 1.5, 0.0, 0.0),
    "self05": (0.75, False, 3000, 1.0, 0.5, 0.0), "self10": (0.75, False, 3000, 1.0, 1.0, 0.0),
    "self20": (0.75, False, 3000, 1.0, 2.0, 0.0), "quad": (0.75, False, 3000, 1.0, 0.0, 0.15),
}
DMAX = 0.5


def build_truth(seed=11):
    from SurvSet.data import SurvLoader
    raw, _ = SurvLoader().load_dataset("support2").values()
    d = raw.rename(columns=lambda c: c.replace("num_", "", 1).replace("fac_", "", 1)).rename(columns={"num_co": "num.co"})
    d = d.loc[:, ~d.columns.duplicated()]
    d = d.dropna(subset=["age", "num.co", "scoma", "hday", "meanbp", "hrt", "resp", "temp", "sod", "sex",
                         "dzgroup", "diabetes", "dementia", "ca"]).reset_index(drop=True)
    Z = pd.DataFrame(index=d.index)
    Z["age"] = d["age"]; Z["age_sq"] = ((d["age"] - 65) / 10) ** 2
    Z["num_co"] = d["num.co"]; Z["scoma"] = d["scoma"]; Z["log_hday"] = np.log1p(d["hday"])
    for v, c in [("meanbp", 80), ("hrt", 90), ("resp", 20), ("temp", 37), ("sod", 138)]:
        Z[v] = d[v]; Z[v + "_sq"] = ((d[v] - c) / d[v].std()) ** 2
    Z["male"] = (d["sex"].astype(str) == "male").astype(float)
    Z["diabetes"] = d["diabetes"].astype(float); Z["dementia"] = d["dementia"].astype(float)
    ca = d["ca"].astype(str); Z["ca_yes"] = (ca == "yes").astype(float); Z["ca_metastatic"] = (ca == "metastatic").astype(float)
    dz = pd.get_dummies(d["dzgroup"].astype(str), prefix="dz", dtype=float)
    Z = Z.join(dz.drop(columns=["dz_ARF/MOSF w/Sepsis"], errors="ignore"))
    Xobs = pd.DataFrame({v: TF[r.tf](d[v].clip(r.lo, r.hi)) for v, r in LABS.iterrows()})
    miss_rate = Xobs.isna().mean()
    imp = IterativeImputer(random_state=seed, sample_posterior=True, max_iter=10)
    Xtrue = pd.DataFrame(imp.fit_transform(np.hstack([Z.values, Xobs.values]))[:, Z.shape[1]:], columns=Xobs.columns)
    for v, r in LABS.iterrows():
        Xtrue[v] = Xtrue[v].clip(TF[r.tf](r.lo), TF[r.tf](r.hi))
    T, E = d["time"].astype(float).clip(lower=0.5).values, d["event"].values.astype(float)
    m = RoystonParmarPH(df=3).fit(T, E, Z.values, Xtrue.values)
    theta = m.theta.copy()
    for v, b in {"bun": 0.35, "crea": 0.35, "alb": -0.35, "glucose": 0.2}.items():
        theta[m.ix_x[list(Xtrue.columns).index(v)]] = b
    tc = {**{v: theta[m.ix_x[list(Xtrue.columns).index(v)]] / m.x_sd[list(Xtrue.columns).index(v)] for v in TERMS_X},
          **{v: theta[m.ix_z[list(Z.columns).index(v)]] / m.z_sd[list(Z.columns).index(v)] for v in TERMS_Z}}
    return dict(model=m, theta=theta, Z=Z, Xtrue=Xtrue, miss_rate=miss_rate, true_coef=tc)


def simulate_times(tr, rows, rng, quad=0.0):
    m, th = tr["model"], tr["theta"]
    Zs = (tr["Z"].values[rows] - m.z_mu) / m.z_sd
    Xs = (tr["Xtrue"].values[rows] - m.x_mu) / m.x_sd
    eta = Zs @ th[m.ix_z] + Xs @ th[m.ix_x]
    if quad:
        jb = list(tr["Xtrue"].columns).index("bun"); eta = eta + quad * (Xs[:, jb] ** 2 - 1)
    target = np.log(-np.log(rng.uniform(size=len(rows)))) - eta
    g = th[m.ix_g]
    lo, hi = np.full(len(rows), np.log(0.5)), np.full(len(rows), np.log(20000.0))
    s = lambda u: np.asarray(rp_basis(jnp.asarray(u), m.knots)) @ g
    for _ in range(60):
        mid = (lo + hi) / 2; up = s(mid) < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    t = np.exp((lo + hi) / 2)
    c = rng.uniform(180, 2000, len(rows))
    return np.minimum(t, c), (t <= c).astype(float), eta, t


def coarsen(tr, rows, driver, gamma, rng, miss_scale=1.0, self_lambda=0.0):
    Xt = tr["Xtrue"].iloc[rows].reset_index(drop=True)
    sev = (driver - driver.mean()) / driver.std()
    miss = pd.DataFrame(index=Xt.index)
    for v in Xt.columns:
        pm = min(float(tr["miss_rate"][v]) * miss_scale, 0.85)
        if float(tr["miss_rate"][v]) <= 0.005:
            miss[v] = False; continue
        own = (Xt[v] - Xt[v].mean()) / Xt[v].std()
        lin = -gamma * sev - self_lambda * own.values        # high values (abnormal) are measured more often
        a = np.log(pm / (1 - pm))
        for _ in range(30):
            p = 1 / (1 + np.exp(-(a + lin)))
            a += np.log(pm / (1 - pm)) - np.log(p.mean() / (1 - p.mean()))
        miss[v] = rng.uniform(size=len(Xt)) < 1 / (1 + np.exp(-(a + lin)))
    Xref = Xt.copy()
    for v in Xt.columns:
        Xref.loc[miss[v].values, v] = TF[LABS.loc[v, "tf"]](LABS.loc[v, "fill"])
    return Xt, Xref, miss


def generate(tr, seed, n, gamma, ydep, miss_scale=1.0, self_lambda=0.0, quad=0.0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(tr["Z"]), n, replace=False)
    T, E, eta, t_true = simulate_times(tr, rows, rng, quad)
    Xt, Xref, miss = coarsen(tr, rows, -np.log(t_true) if ydep else eta, gamma, rng, miss_scale, self_lambda)
    Z = tr["Z"].iloc[rows].reset_index(drop=True); Z = Z.loc[:, Z.std() > 0]
    return rng, rows, T, E, Xt, Xref, miss, Z


def rp_terms(T, E, Z, X, theta0=None, return_theta=False):
    m = RoystonParmarPH(df=3).fit(T, E, Z.values, X, theta0=theta0)
    zn, xn = list(Z.columns), list(LABS.index)
    est = [m.coef("x", xn.index(v)) for v in TERMS_X] + [m.coef("z", zn.index(v)) for v in TERMS_Z]
    res = np.array([e for e, _ in est]), np.array([s for _, s in est])
    return (*res, m.theta) if return_theta else res


def wald(e, s):
    return e, e - Q95 * s, e + Q95 * s


def smc_envelope(T, E, Z, Xt, miss, rep, m_smc=3, theta0=None, diag=None):
    """SMC-FCS imputations (m = 3, 8 cycles, at most 1,000 proposals per value) and the envelope object over the
    heavily missing variables. theta0 (optional) is a starting value for the model fits; fits are refined to the
    exact maximum, so it affects speed only. If `diag` is a dict, rejection-sampling diagnostics are stored in it."""
    Xm = Xt.mask(miss); sd = Xm.std()
    imps, dg = smcfcs_rp(T, E, Z.values, Xm.values, LO_SUP, HI_SUP, m=m_smc, seed=7000 + rep, theta0=theta0, return_diag=True)
    if diag is not None:
        diag.update(dg)
    heavy = [v for v in LABS.index if miss[v].mean() > 0.05]
    cols = list(LABS.index)
    warm = {k: theta0 for k in range(len(imps))} if theta0 is not None else {}

    def core(d):
        ests, vars_ = [], []
        for k, A in enumerate(imps):
            A2 = A.copy()
            for j, v in enumerate(heavy):
                if d[j]:
                    A2[miss[v].values, cols.index(v)] += d[j] * sd[v]
            e, s, th = rp_terms(T, E, Z, A2, theta0=warm.get(k), return_theta=True)
            if k not in warm or warm[k] is theta0:
                warm[k] = th                           # warm start from the reference-configuration fit
            ests.append(e); vars_.append(s ** 2)
        return rubin(ests, vars_)
    return Envelope(core, len(heavy)), imps, heavy


def run_rep(tr, scen, rep, truth=None):
    if scen in MECH:
        gamma, ydep = MECH[scen]; n, ms, lam, quad = 3000, 1.0, 0.0, 0.0
    else:
        gamma, ydep, n, ms, lam, quad = EXT[scen]
    truth = truth or tr["true_coef"]
    _, _, T, E, Xt, Xref, miss, Z = generate(tr, 10_000 + rep, n, gamma, ydep, ms, lam, quad)
    out = {}
    e0, s0, th0 = rp_terms(T, E, Z, Xt.values, return_theta=True); out["Oracle"] = wald(e0, s0)
    env, imps, heavy = smc_envelope(T, E, Z, Xt, miss, rep, theta0=th0)
    out["MI, SMC-FCS"] = env.eval(np.zeros(env.q))
    (lo, hi), _ = env.vertex_2q3(DMAX); out["HPPSM v2 (2q+3 configurations)"] = (out["MI, SMC-FCS"][0], lo, hi)
    (lo, hi), ncfg = env.adaptive(DMAX); out["HPPSM v2 (proposed)"] = (out["MI, SMC-FCS"][0], lo, hi)
    if scen not in MECH:
        for dm in (0.25, 1.0):
            (lo, hi), _ = env.adaptive(dm); out[f"HPPSM v2 (dmax = {dm})"] = (out["MI, SMC-FCS"][0], lo, hi)
    out["Fill-in Wald"] = wald(*rp_terms(T, E, Z, Xref.values, theta0=th0))
    cc = ~miss[TERMS_X].any(axis=1).values
    if cc.sum() > 100:
        Zc = Z[cc].loc[:, Z[cc].std() > 0]
        try:
            out["Complete case"] = wald(*rp_terms(T[cc], E[cc], Zc, Xt.values[cc]))
        except Exception:
            pass
    # chained-equation MI (a comparator known to be biased here) is computed for the first 200 replicates of
    # the main mechanisms only, because it accounts for about half of the computing time
    if scen not in MECH or rep >= 200:
        k_mi = 0
    else:
        k_mi = 5
    from lifelines import NelsonAalenFitter
    na = NelsonAalenFitter().fit(T, E).cumulative_hazard_at_times(T).values
    Xm = Xt.mask(miss); ests, vars_ = [], []
    for k in range(k_mi):
        A = IterativeImputer(random_state=1000 * rep + k, sample_posterior=True, max_iter=5).fit_transform(
            np.hstack([Z.values, E[:, None], na[:, None], Xm.values]))[:, Z.shape[1] + 2:]
        e, s = rp_terms(T, E, Z, A, theta0=th0); ests.append(e); vars_.append(s ** 2)
    if k_mi:
        out["MI, chained equations"] = rubin(ests, vars_)
    rows = pd.DataFrame([dict(rep=rep, method=k, term=t, est=e[i], lo=l[i], hi=h[i], truth=truth[t])
                         for k, (e, l, h) in out.items() for i, t in enumerate(TERMS)])
    # empirical MNAR shift: mean (true - imputed) among missing values, in SD units of the observed values
    cols = list(LABS.index)
    dl = {v: float(np.mean([np.mean(Xt[v].values[miss[v].values] - A[miss[v].values, cols.index(v)]) for A in imps])
                   / Xm[v].std()) for v in heavy}
    info = dict(rep=rep, n_configs=ncfg, cc=int(cc.sum()), **{f"delta_{v}": dl[v] for v in heavy})
    return rows, info


def vertex_rep(tr, scen, rep, truth=None):
    """Full vertex enumeration (primary in Draft 8) for one replicate; the data, imputations and reference
    analysis are identical to run_rep, so the rows can be merged with its output."""
    if scen in MECH:
        gamma, ydep = MECH[scen]; n, ms, lam, quad = 3000, 1.0, 0.0, 0.0
    else:
        gamma, ydep, n, ms, lam, quad = EXT[scen]
    truth = truth or tr["true_coef"]
    _, _, T, E, Xt, Xref, miss, Z = generate(tr, 10_000 + rep, n, gamma, ydep, ms, lam, quad)
    th0 = rp_terms(T, E, Z, Xt.values, return_theta=True)[2]
    env, imps, heavy = smc_envelope(T, E, Z, Xt, miss, rep, theta0=th0)
    e0 = env.eval(np.zeros(env.q))[0]
    (lo, hi), nv = env.vertices(DMAX)
    (la, ha), _ = env.adaptive(DMAX)
    out = {"HPPSM v2 (vertices only)": (e0, lo, hi), "HPPSM v2 (adaptive, check)": (e0, la, ha),
           "HPPSM v2 (vertices + axes)": (e0, np.minimum(lo, la), np.maximum(hi, ha))}   # primary in Draft 8
    nv = len(env.cache)
    return pd.DataFrame([dict(rep=rep, method=k, term=t, est=e[i], lo=l[i], hi=h[i], truth=truth[t], n_configs=nv)
                         for k, (e, l, h) in out.items() for i, t in enumerate(TERMS)])


def smc_rep(tr, scen, rep, truth=None):
    """All SMC-FCS-based methods for one replicate (Draft 9: exact truncated proposals, explicit fall-back).
    Data are identical to run_rep; the non-SMC methods (oracle, fill-in, complete case, chained-equation MI)
    do not depend on the sampler and are taken from run_rep."""
    if scen in MECH:
        gamma, ydep = MECH[scen]; n, ms, lam, quad = 3000, 1.0, 0.0, 0.0
    else:
        gamma, ydep, n, ms, lam, quad = EXT[scen]
    truth = truth or tr["true_coef"]
    _, _, T, E, Xt, Xref, miss, Z = generate(tr, 10_000 + rep, n, gamma, ydep, ms, lam, quad)
    th0 = rp_terms(T, E, Z, Xt.values, return_theta=True)[2]
    dg = {}
    env, imps, heavy = smc_envelope(T, E, Z, Xt, miss, rep, theta0=th0, diag=dg)
    e0, l0, h0 = env.eval(np.zeros(env.q))
    out = {"MI, SMC-FCS": (e0, l0, h0)}
    (l, h), _ = env.vertex_2q3(DMAX); out["HPPSM v2 (2q+3 configurations)"] = (e0, l, h)
    (la, ha), _ = env.adaptive(DMAX); out["HPPSM v2 (adaptive)"] = (e0, la, ha)
    (lv, hv), _ = env.vertices(DMAX); out["HPPSM v2 (vertices only)"] = (e0, lv, hv)
    out["HPPSM v2 (proposed)"] = (e0, np.minimum(lv, la), np.maximum(hv, ha))           # vertices + axes
    if scen not in MECH:
        for dm in (0.25, 1.0):
            (l, h), _ = env.adaptive(dm); out[f"HPPSM v2 (dmax = {dm})"] = (e0, l, h)
    rows = pd.DataFrame([dict(rep=rep, method=k, term=t, est=e[i], lo=l_[i], hi=h_[i], truth=truth[t])
                         for k, (e, l_, h_) in out.items() for i, t in enumerate(TERMS)])
    cols = list(LABS.index); Xm = Xt.mask(miss)
    dl = {v: float(np.mean([np.mean(Xt[v].values[miss[v].values] - A[miss[v].values, cols.index(v)]) for A in imps]) / Xm[v].std()) for v in heavy}
    info = dict(rep=rep, n_configs=len(env.cache), **{f"smc_{k}": v for k, v in dg.items()}, **{f"delta_{v}": dl[v] for v in heavy})
    return rows, info


def cindex_rep(tr, rep, mech, n):
    """C-index of each method on an independent test sample of 3,000 patients with true covariates."""
    from sksurv.metrics import concordance_index_censored
    from lifelines import NelsonAalenFitter
    gamma, ydep = MECH[mech]
    rng, rows, T, E, Xt, Xref, miss, Z = generate(tr, 900_000 + 1000 * rep + n + {"M1": 1, "M2": 2, "M3": 3, "M4": 4}[mech], n, gamma, ydep)
    trow = rng.choice(len(tr["Z"]), 3000, replace=False)
    Tt, Et, _, _ = simulate_times(tr, trow, rng)
    Zt = tr["Z"].iloc[trow].reset_index(drop=True)[Z.columns]; Xtt = tr["Xtrue"].iloc[trow].values
    C = lambda lp: concordance_index_censored(Et.astype(bool), Tt, lp)[0]
    lp = lambda TT, EE, ZZ, XX, Zte=Zt: RoystonParmarPH().fit(TT, EE, ZZ.values, XX).lp(Zte.values, Xtt)
    res = {"Oracle": C(lp(T, E, Z, Xt.values)), "Fill-in": C(lp(T, E, Z, Xref.values))}
    cc = ~miss.any(axis=1).values
    if cc.sum() > 50:
        Zc = Z[cc].loc[:, Z[cc].std() > 0]
        try:
            res["Complete case"] = C(lp(T[cc], E[cc], Zc, Xt.values[cc], Zt[Zc.columns]))
        except Exception:
            pass
    na = NelsonAalenFitter().fit(T, E).cumulative_hazard_at_times(T).values; Xm = Xt.mask(miss)
    res["MI, chained equations"] = C(np.mean([lp(T, E, Z, IterativeImputer(random_state=k, sample_posterior=True, max_iter=5).fit_transform(
        np.hstack([Z.values, E[:, None], na[:, None], Xm.values]))[:, Z.shape[1] + 2:]) for k in range(3)], 0))
    res["HPPSM v2 (proposed)"] = C(np.mean([lp(T, E, Z, A) for A in smcfcs_rp(T, E, Z.values, Xm.values, LO_SUP, HI_SUP, m=3, seed=rep, n_cycles=6)], 0))
    return pd.DataFrame([dict(mech=mech, n=n, rep=rep, method=k, C=v) for k, v in res.items()])


def dense_rep(tr, mech, rep, n_random=100):
    gamma, ydep = MECH[mech]
    _, _, T, E, Xt, Xref, miss, Z = generate(tr, 10_000 + rep, 3000, gamma, ydep)
    th0 = rp_terms(T, E, Z, Xt.values, return_theta=True)[2]
    env, imps, heavy = smc_envelope(T, E, Z, Xt, miss, rep, theta0=th0)
    t0 = time.time(); (l1, h1), n1 = env.vertex_2q3(DMAX)
    (l2, h2), n2 = env.adaptive(DMAX)
    (l4, h4), n4 = env.vertices(DMAX)
    l5, h5 = np.minimum(l4, l2), np.maximum(h4, h2)          # vertices + axes (reference, singles, adaptive vertices)
    (l3, h3), n3 = env.dense(DMAX, n_random, np.random.default_rng(rep))
    # union of everything evaluated (dense + adaptive) is the best available approximation of the full union
    keys = list(env.cache); L = np.min([env.cache[k][1] for k in keys], 0); H = np.max([env.cache[k][2] for k in keys], 0)
    out = []
    for i, t in enumerate(TERMS):
        out.append(dict(mech=mech, rep=rep, term=t, truth=tr["true_coef"][t], lo_full=L[i], hi_full=H[i],
                        lo_2q3=l1[i], hi_2q3=h1[i], lo_adapt=l2[i], hi_adapt=h2[i], lo_dense=l3[i], hi_dense=h3[i], lo_vert=l4[i], hi_vert=h4[i], lo_va=l5[i], hi_va=h5[i],
                        n_2q3=n1, n_adapt=n2, n_all=len(keys)))
    return pd.DataFrame(out)


def pseudo_truth(tr, quad, K=20):
    """Pseudo-true coefficients of the (misspecified) linear working model: average of oracle fits to K
    simulated full cohorts with true covariates."""
    est = []
    for k in range(K):
        rng = np.random.default_rng(424242 + k); rows = np.arange(len(tr["Z"]))
        T, E, _, _ = simulate_times(tr, rows, rng, quad)
        Z = tr["Z"].loc[:, tr["Z"].std() > 0]
        est.append(rp_terms(T, E, Z, tr["Xtrue"].values)[0])
    return dict(zip(TERMS, np.mean(est, 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", required=True); ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--out", default="results_s2")
    ap.add_argument("--dense", action="store_true"); ap.add_argument("--vertex", action="store_true")
    ap.add_argument("--smc", action="store_true"); ap.add_argument("--cindex", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tr = build_truth()
    if a.dense:
        path = f"{a.out}/dense_{a.scen}.csv"
        done = set(pd.read_csv(path).rep) if os.path.exists(path) else set()
        for r in range(a.start, a.reps):
            if r in done:
                continue
            t0 = time.time(); dense_rep(tr, a.scen, r).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[dense {a.scen}] rep {r} {time.time()-t0:.0f}s", flush=True)
        sys.exit()
    truth = None
    if a.cindex:
        path = f"{a.out}/cindex_support2.csv"
        done = set(map(tuple, pd.read_csv(path)[["mech", "n", "rep"]].values)) if os.path.exists(path) else set()
        for r in range(a.start, a.reps):
            for m in MECH:
                for n in (500, 1000, 3000):
                    if (m, n, r) not in done:
                        cindex_rep(tr, r, m, n).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[cindex] rep {r}", flush=True)
        sys.exit()
    if a.smc:
        if a.scen in EXT and EXT[a.scen][5]:
            truth = json.load(open(f"{a.out}/pseudo_truth_{a.scen}.json"))
        path = f"{a.out}/{a.scen}_smc.csv"; ipath = f"{a.out}/{a.scen}_smc_info.csv"
        done = set(pd.read_csv(path).rep) if os.path.exists(path) else set()
        for r in range(a.start, a.reps):
            if r in done:
                continue
            t0 = time.time(); rows, info = smc_rep(tr, a.scen, r, truth)
            rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            pd.DataFrame([info]).to_csv(ipath, mode="a", header=not os.path.exists(ipath), index=False)
            print(f"[smc {a.scen}] rep {r} {time.time()-t0:.0f}s", flush=True)
        sys.exit()
    if a.vertex:
        if a.scen in EXT and EXT[a.scen][5]:
            truth = json.load(open(f"{a.out}/pseudo_truth_{a.scen}.json"))
        path = f"{a.out}/{a.scen}_vert.csv"
        done = set(pd.read_csv(path).rep) if os.path.exists(path) else set()
        for r in range(a.start, a.reps):
            if r in done:
                continue
            t0 = time.time(); vertex_rep(tr, a.scen, r, truth).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[vertex {a.scen}] rep {r} {time.time()-t0:.0f}s", flush=True)
        sys.exit()
    if a.scen in EXT and EXT[a.scen][5]:
        pt = f"{a.out}/pseudo_truth_{a.scen}.json"
        if not os.path.exists(pt):
            json.dump(pseudo_truth(tr, EXT[a.scen][5]), open(pt, "w"))
        truth = json.load(open(pt))
    path = f"{a.out}/{a.scen}_{a.start}.csv"; ipath = f"{a.out}/{a.scen}_{a.start}_info.csv"
    done = set(pd.read_csv(path).rep) if os.path.exists(path) else set()
    for r in range(a.start, a.reps):
        if r in done:
            continue
        t0 = time.time()
        rows, info = run_rep(tr, a.scen, r, truth)
        rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        pd.DataFrame([info]).to_csv(ipath, mode="a", header=not os.path.exists(ipath), index=False)
        print(f"[{a.scen}] rep {r} {time.time()-t0:.0f}s", flush=True)
