"""
p12_study.py - PhysioNet/CinC Challenge 2012 analyses for Draft 7.

  coverage : real-data resampling experiment with full-cohort reference values (scenarios A and B), 1,000 replicates
  dense    : verification of the finite-configuration union (scenario B: all 16 delta-vertices + 50 random
             points, kappa on a 0.1 grid)
  me       : empirical check of the measurement-error assumptions (bias, bound, nondifferentiality,
             within-patient autocorrelation)
  app      : full-cohort application, with crisp and graded (triangular) contours
"""
import os, sys, glob, time, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hppsm_v3 import logit_fit, rubin, regression_calibration, error_variance_bound, Envelope, Q95
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# paths: set P12_RAWDIR to the folder with set-a/, set-b/, set-c/ and Outcomes-*.txt; the summary is built by p12_build.py
RAWDIR = os.environ.get("P12_RAWDIR", "physionet-challenge-2012")
SUMMARY = os.environ.get("P12_SUMMARY", "p12_summary.pkl")
VARS = ["HR", "Temp", "GCS", "BUN", "Creatinine", "HCT", "WBC", "Platelets", "Glucose", "Na", "K", "HCO3"]
LOGV = {"BUN", "Creatinine", "WBC", "Platelets", "Glucose"}
PHYS = {"HR": (20, 220), "Temp": (30, 43), "GCS": (3, 15), "BUN": (2, 200), "Creatinine": (0.2, 15), "HCT": (10, 60),
        "WBC": (0.1, 100), "Platelets": (5, 1000), "Glucose": (20, 1000), "Na": (110, 175), "K": (1.5, 8), "HCO3": (5, 50)}
NORMAL = {"BUN": 14, "Creatinine": 0.9, "Glucose": 110, "WBC": 8}; MNAR_VARS = list(NORMAL)
TERMS = VARS + ["Age"]; DMAX = 0.5; KAPPAS = (0.0, 0.5, 1.0)
f = lambda v, x: np.log(x) if v in LOGV else x


def covariates(X):
    Z = pd.DataFrame({"Age": X.Age, "male": (X.Gender == 1).astype(float), "MechVent": X.MechVent_any.astype(float),
                      **{f"ICU{k}": (X.ICUType == k).astype(float) for k in (2, 3, 4)}}).reset_index(drop=True)
    get = lambda st: pd.DataFrame({v: f(v, X[f"{v}_{st}"].clip(*PHYS[v])) for v in VARS}).reset_index(drop=True)
    NN = pd.DataFrame({v: X[f"{v}_n"] for v in VARS}).reset_index(drop=True)
    return Z, get("mean"), get("min"), get("max"), get("first"), NN


RAW = pd.read_pickle(SUMMARY)
keep = RAW[[f"{v}_mean" for v in VARS]].notna().all(1) & RAW.Age.notna() & RAW["In-hospital_death"].notna()
Zall, Mall, LOall, HIall, FIall, NNall = covariates(RAW[keep])
yall = RAW.loc[keep, "In-hospital_death"].values.astype(float)
AGE_J = list(Zall.columns).index("Age")


def logit_terms(y, Xv, Z):
    b, se, _ = logit_fit(y, np.hstack([Xv, Z]))
    idx = list(range(len(VARS))) + [len(VARS) + AGE_J]
    return b[idx], se[idx]


TRUE_COEF = dict(zip(TERMS, logit_terms(yall, Mall.values, Zall.values)[0]))
_b, _, _a = logit_fit(yall, np.hstack([Mall.values, Zall.values]))
LP_TRUE = np.hstack([Mall.values, Zall.values]) @ _b + _a


def mnar_deletion(idx, n, rng):
    sev = (LP_TRUE[idx] - LP_TRUE[idx].mean()) / LP_TRUE[idx].std()
    miss = pd.DataFrame(False, index=range(n), columns=VARS)
    for v in MNAR_VARS:
        a = np.log(0.4 / 0.6)
        for _ in range(30):
            p = 1 / (1 + np.exp(-(a - sev))); a += np.log(0.4 / 0.6) - np.log(p.mean() / (1 - p.mean()))
        miss[v] = rng.uniform(size=n) < 1 / (1 + np.exp(-(a - sev)))
    return miss


def setup(rep, scenario, n=3000):
    rng = np.random.default_rng(50_000 + rep)
    idx = rng.integers(0, len(yall), n)
    y = yall[idx]; Z = Zall.iloc[idx].reset_index(drop=True)
    M, LO, HI, FI, NN = (A.iloc[idx].reset_index(drop=True).copy() for A in (Mall, LOall, HIall, FIall, NNall))
    MID = (LO + HI) / 2
    B2 = pd.DataFrame(error_variance_bound(LO.values, HI.values, NN.values), columns=VARS)
    miss = mnar_deletion(idx, n, rng) if scenario == "B" else pd.DataFrame(False, index=MID.index, columns=VARS)
    return dict(rep=rep, y=y, Z=Z, M=M, MID=MID, FI=FI, B2=B2, miss=miss)


def ar1_design_effect():
    """(1 + rho)/(1 - rho) per variable, rho = median within-patient lag-1 autocorrelation (external information,
    here estimated from the raw PhysioNet series of the whole cohort)."""
    ac = pd.read_csv("results_p12/p12_lag1.csv", index_col=0).iloc[:, 0]
    return np.array([(1 + ac[v]) / (1 - ac[v]) for v in VARS])


def make_core(S, m_env=3, de=None):
    """Returns core(kappa, delta) -> (est, lo, hi) and the list of shifted variables."""
    y, Zv, MID, miss, B2 = S["y"], S["Z"].values, S["MID"], S["miss"], S["B2"].copy()
    if de is not None:
        B2 = B2 * de
    if miss.values.any():
        Xm = MID.mask(miss); sd = Xm.std(); B2 = B2.mask(miss, 0.0)
        imps = [IterativeImputer(random_state=1000 * S["rep"] + k, sample_posterior=True, max_iter=5).fit_transform(
            np.hstack([Zv, y[:, None], Xm.values]))[:, Zv.shape[1] + 1:] for k in range(m_env)]
        shifted = MNAR_VARS
    else:
        imps, sd, shifted = [MID.values], None, []
    su2 = B2.mean().values

    def core(kappa, d):
        ests, vars_ = [], []
        for A in imps:
            A2 = A.copy()
            for j, v in enumerate(shifted):
                if d[j]:
                    A2[miss[v].values, VARS.index(v)] += d[j] * sd[v]
            Xc = regression_calibration(A2, Zv, kappa * su2) if kappa > 0 else A2
            e, s = logit_terms(y, Xc, Zv); ests.append(e); vars_.append(s ** 2)
        return rubin(ests, vars_) if len(imps) > 1 else (np.array(ests[0]), ests[0] - Q95 * np.sqrt(vars_[0]), ests[0] + Q95 * np.sqrt(vars_[0]))
    return core, shifted


def envelope_union(env, dmax, method):
    """method: 'adaptive', '2q3' or 'vertices+axes' (all 2^q vertices, the reference, single-coordinate and
    adaptive configurations; primary in Draft 8). Returns the interval hull of the evaluated core intervals."""
    if method == "adaptive":
        return env.adaptive(dmax)[0]
    if method == "2q3":
        return env.vertex_2q3(dmax)[0]
    (l1, h1), _ = env.vertices(dmax); (l2, h2), _ = env.adaptive(dmax)
    return np.minimum(l1, l2), np.maximum(h1, h2)


def hppsm(S, method="adaptive", kappas=KAPPAS, dmax=DMAX, de=None):
    core, shifted = make_core(S, de=de)
    envs = {k: Envelope(lambda d, k=k: core(k, d), len(shifted)) for k in kappas}
    lo, hi = [], []
    for k, env in envs.items():
        if shifted:
            l, h = envelope_union(env, dmax, method)
        else:
            _, l, h = env.eval(np.zeros(0))
        lo.append(l); hi.append(h)
    est = envs[kappas[0]].eval(np.zeros(len(shifted)))[0]
    return est, np.min(lo, 0), np.max(hi, 0), envs


def wald(e, s):
    return e, e - Q95 * s, e + Q95 * s


def coverage_rep(rep, scenario):
    S = setup(rep, scenario); y, Z, Zv, miss = S["y"], S["Z"], S["Z"].values, S["miss"]
    MIDf, FIf = S["MID"].copy(), S["FI"].copy()
    for v in MNAR_VARS:
        MIDf.loc[miss[v].values, v] = f(v, NORMAL[v]); FIf.loc[miss[v].values, v] = f(v, NORMAL[v])
    out = {"Oracle": wald(*logit_terms(y, S["M"].values, Zv)),
           "Mid-point Wald": wald(*logit_terms(y, MIDf.values, Zv)),
           "First-value Wald": wald(*logit_terms(y, FIf.values, Zv))}
    if scenario == "B":
        out["Missing-indicator Wald"] = wald(*logit_terms(y, MIDf.values, np.hstack([Zv, miss[MNAR_VARS].values.astype(float)])))
        Xm = S["MID"].mask(miss); ests, vars_ = [], []
        for k in range(5):
            A = IterativeImputer(random_state=1000 * rep + k, sample_posterior=True, max_iter=5).fit_transform(
                np.hstack([Zv, y[:, None], Xm.values]))[:, Zv.shape[1] + 1:]
            e, s = logit_terms(y, A, Zv); ests.append(e); vars_.append(s ** 2)
        out["Multiple imputation"] = rubin(ests, vars_)
    e, l, h, envs = hppsm(S, "adaptive")
    out["HPPSM v2 (proposed)"] = (e, l, h)
    if scenario == "B":
        lo = np.min([env.vertex_2q3(DMAX)[0][0] for env in envs.values()], 0)
        hi = np.max([env.vertex_2q3(DMAX)[0][1] for env in envs.values()], 0)
        out["HPPSM v2 (2q+3 configurations)"] = (e, lo, hi)
        for dm in (0.25, 1.0):
            lo = np.min([env.adaptive(dm)[0][0] for env in envs.values()], 0)
            hi = np.max([env.adaptive(dm)[0][1] for env in envs.values()], 0)
            out[f"HPPSM v2 (dmax = {dm})"] = (e, lo, hi)
    # kappa sensitivity: no error correction (kappa = 0 only) and kappa up to 2 (bound exceeded two-fold)
    for kk, nm in (((0.0,), "HPPSM v2 (kappa = 0 only)"), ((0.0, 0.5, 1.0, 1.5, 2.0), "HPPSM v2 (kappa up to 2)")):
        out[nm] = hppsm(S, "adaptive", kappas=kk)[:3]
    out["HPPSM v2 (AR(1)-adjusted bound)"] = hppsm(S, "adaptive", de=ar1_design_effect())[:3]
    return pd.DataFrame([dict(rep=rep, method=k, term=t, est=e_[i], lo=l_[i], hi=h_[i], truth=TRUE_COEF[t])
                         for k, (e_, l_, h_) in out.items() for i, t in enumerate(TERMS)])


def vertex_rep(rep):
    """Scenario B, HPPSM with full vertex enumeration (vertices + axes); data identical to coverage_rep."""
    S = setup(rep, "B")
    e, l, h, _ = hppsm(S, "vertices+axes")
    return pd.DataFrame([dict(rep=rep, method="HPPSM v2 (vertices + axes)", term=t, est=e[i], lo=l[i], hi=h[i], truth=TRUE_COEF[t])
                         for i, t in enumerate(TERMS)])


def dense_rep(rep, n_random=50):
    S = setup(rep, "B"); core, shifted = make_core(S)
    rows = []
    res = {}
    for k in np.round(np.arange(0, 1.01, 0.1), 2):
        env = Envelope(lambda d, k=k: core(k, d), len(shifted))
        if k in KAPPAS:
            res[("adapt", k)] = env.adaptive(DMAX)[0]; res[("2q3", k)] = env.vertex_2q3(DMAX)[0]
            res[("vert", k)] = env.vertices(DMAX)[0]; res[("va", k)] = envelope_union(env, DMAX, "vertices+axes")
        res[("dense", k)] = env.dense(DMAX, n_random, np.random.default_rng(rep))[0]
    U = lambda tag, ks: (np.min([res[(tag, k)][0] for k in ks], 0), np.max([res[(tag, k)][1] for k in ks], 0))
    kd = [k for (t, k) in res if t == "dense"]
    l1, h1 = U("2q3", KAPPAS); l2, h2 = U("adapt", KAPPAS); l3, h3 = U("dense", kd)
    l4, h4 = U("vert", KAPPAS); l5, h5 = U("va", KAPPAS)
    l3, h3 = np.minimum(l3, np.minimum(l1, l2)), np.maximum(h3, np.maximum(h1, h2))
    for i, t in enumerate(TERMS):
        rows.append(dict(rep=rep, term=t, truth=TRUE_COEF[t], lo_2q3=l1[i], hi_2q3=h1[i], lo_adapt=l2[i], hi_adapt=h2[i],
                         lo_vert=l4[i], hi_vert=h4[i], lo_va=l5[i], hi_va=h5[i], lo_full=l3[i], hi_full=h3[i]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ measurement-error diagnostics
def me_empirical(out):
    """Per variable: standardised mean of U = W - X*, Var(U) relative to the mean bound, correlation of U with
    X*, the log odds ratio of U when added to the oracle model (nondifferentiality), and within-patient lag-1
    autocorrelation of the measurements."""
    U = (LOall + HIall) / 2 - Mall
    B2 = pd.DataFrame(error_variance_bound(LOall.values, HIall.values, NNall.values), columns=VARS)
    rows = []
    base = np.hstack([Mall.values, Zall.values])
    for j, v in enumerate(VARS):
        b, se, _ = logit_fit(yall, np.hstack([base, U[[v]].values]))
        rows.append(dict(variable=v, n_median=float(NNall[v].median()), mean_U_over_sdX=U[v].mean() / Mall[v].std(),
                         ratio_EU2_to_bound=float((U[v] ** 2).mean() / B2[v].mean()),
                         ratio_VarU_to_bound=float(U[v].var() / B2[v].mean()),
                         corr_U_X=float(np.corrcoef(U[v], Mall[v])[0, 1]),
                         reliability_mid=float(1 - U[v].var() / ((LOall[v] + HIall[v]) / 2).var()),
                         logOR_U_per_SD=float(b[-1] * U[v].std()), z_U=float(b[-1] / se[-1])))
    me = pd.DataFrame(rows)
    # within-patient lag-1 autocorrelation from the raw time series (patients with >= 6 measurements)
    cache = os.path.join(out, "p12_lag1.csv")
    if not os.path.exists(cache):
        frames = []
        for s in ["set-a", "set-b", "set-c"]:
            for fn in glob.glob(os.path.join(RAWDIR, s, "*.txt")):
                d = pd.read_csv(fn); d = d[d.Parameter.isin(VARS)]
                d["hours"] = d.Time.str.split(":").apply(lambda x: int(x[0]) + int(x[1]) / 60)
                d = d[(d.hours > 0) & (d.Value > 0)]
                frames.append(d.assign(file=fn))
        L = pd.concat(frames).sort_values(["file", "Parameter", "hours"])
        def lag1(x):
            x = x.values
            if len(x) < 6 or np.std(x) == 0:
                return np.nan
            x = x - x.mean(); return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))
        ac = L.groupby(["Parameter", "file"]).Value.apply(lag1).groupby("Parameter").median()
        ac.rename("lag1_autocorr_median").to_csv(cache)
    ac = pd.read_csv(cache, index_col=0).iloc[:, 0]
    me["lag1_autocorr_median"] = me.variable.map(ac)
    return me


# ------------------------------------------------------------------ application with crisp and graded contours
def application(out):
    X = RAW[RAW.Age.notna() & RAW["In-hospital_death"].notna()]
    ya = X["In-hospital_death"].values.astype(float)
    Za, MEANa, LOa, HIa, _, NNa = covariates(X); Zav = Za.values
    MIDa = (LOa + HIa) / 2; missa = MIDa.isna()
    B2a = pd.DataFrame(error_variance_bound(LOa.fillna(0).values, HIa.fillna(0).values, NNa.fillna(1).values), columns=VARS)
    sd_a = MIDa.std(); cc = ~missa.any(axis=1).values
    res = {"Oracle (true 48-h means)": wald(*logit_terms(ya[cc], MEANa.values[cc], Zav[cc])),
           "Mid-point Wald": wald(*logit_terms(ya[cc], MIDa.values[cc], Zav[cc]))}
    imps = [IterativeImputer(random_state=k, sample_posterior=True, max_iter=10).fit_transform(
        np.hstack([Zav, ya[:, None], MIDa.values]))[:, Zav.shape[1] + 1:] for k in range(5)]
    heavy = [v for v in VARS if missa[v].mean() > 0.01] or VARS[:4]

    def core(kappa, d, alpha=0.05):
        ests, vars_ = [], []
        for A in imps:
            A2 = A.copy()
            for j, v in enumerate(heavy):
                if d[j]:
                    A2[missa[v].values, VARS.index(v)] += d[j] * sd_a[v]
            Xc = regression_calibration(A2, Zav, kappa * B2a.mean().values) if kappa > 0 else A2
            e, s = logit_terms(ya, Xc, Zav); ests.append(e); vars_.append(s ** 2)
        return rubin(ests, vars_, alpha)
    est0 = core(0.0, np.zeros(len(heavy)))
    res["Multiple imputation"] = est0
    lo, hi = [], []
    for k in KAPPAS:
        # q = 12 shifted variables here, so 2^12 vertices x 3 kappas x 5 imputations is not affordable: adaptive search
        l, h = envelope_union(Envelope(lambda d, k=k: core(k, d), len(heavy)), DMAX, "adaptive"); lo.append(l); hi.append(h)
    res["HPPSM v2 (proposed)"] = (est0[0], np.min(lo, 0), np.max(hi, 0))
    scale = np.array([sd_a[v] for v in VARS] + [X.Age.std()])
    app = pd.DataFrame([dict(method=k, term=t, est=e[i] * scale[i], lo=l[i] * scale[i], hi=h[i] * scale[i])
                        for k, (e, l, h) in res.items() for i, t in enumerate(TERMS)])
    orc = app[app.method.str.startswith("Oracle")].set_index("term").est
    app["contains_oracle"] = (app.lo <= app.term.map(orc)) & (app.term.map(orc) <= app.hi)
    app.to_csv(os.path.join(out, "application.csv"), index=False)
    # graded contour: triangular on delta (sup-norm) and on kappa, pi(psi) = 1 - max(|delta|/dmax, kappa);
    # alpha-cut = union over {psi : pi(psi) >= alpha} of the (1 - alpha) core intervals. Crisp: Psi fixed.
    rows = []
    for a in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        for kind in ("graded", "crisp"):
            r = (1 - a) if kind == "graded" else 1.0
            lo, hi = [], []
            for k in sorted(set([0.0, 0.5 * r, r])):
                env = Envelope(lambda d, k=k: core(k, d, a), len(heavy))
                l, h = envelope_union(env, DMAX * r, "adaptive")
                lo.append(l); hi.append(h)
            l, h = np.min(lo, 0), np.max(hi, 0)
            for i, t in enumerate(TERMS):
                rows.append(dict(alpha=a, contour=kind, term=t, lo=l[i] * scale[i], hi=h[i] * scale[i]))
        cfg = core(0.0, np.zeros(len(heavy)), a)
        for i, t in enumerate(TERMS):
            rows.append(dict(alpha=a, contour="MI only", term=t, lo=cfg[1][i] * scale[i], hi=cfg[2][i] * scale[i]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "application_contours.csv"), index=False)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--scen", default="A")
    ap.add_argument("--reps", type=int, default=1000); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="results_p12")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.task == "me":
        print(me_empirical(a.out).round(3).to_string()); me_empirical(a.out).to_csv(f"{a.out}/me_empirical.csv", index=False)
    elif a.task == "app":
        print(application(a.out).round(3).to_string())
    else:
        fn = dict(coverage=lambda r: coverage_rep(r, a.scen), dense=dense_rep, vertex=vertex_rep)[a.task]
        path = f"{a.out}/{a.task}_{a.scen if a.task == 'coverage' else 'B'}_{a.start}.csv"
        done = set(pd.read_csv(path).rep) if os.path.exists(path) else set()
        for r in range(a.start, a.reps):
            if r in done:
                continue
            t0 = time.time(); fn(r).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[{a.task} {a.scen}] rep {r} {time.time()-t0:.1f}s", flush=True)
