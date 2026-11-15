"""
hppsm_v3.py - core library for the HPPSM analyses (Draft 7).

Contents
  * RoystonParmarPH : flexible parametric PH model (JAX likelihood), fitted by L-BFGS-B and then
                      refined by Newton-Raphson steps to a gradient norm below 1e-8, so the maximum
                      likelihood estimate does not depend on optimiser stopping rules.
  * logit_fit       : unpenalised logistic regression (coefficients and SEs per raw unit).
  * rubin           : Rubin's rules.
  * smcfcs_rp       : substantive-model-compatible FCS imputation for the RP model (Bartlett et al. 2015).
  * regression_calibration, error_variance_bound : interval-summarised covariates.
  * Envelope        : union of core intervals over mechanism configurations, computed by
                        - vertex_2q3  : reference, each variable at +/-dmax alone, all +dmax, all -dmax
                        - adaptive    : the above plus, for every endpoint of every target, the vertex
                                        whose signs are read off the single-variable runs (exact when
                                        each endpoint is monotone in each coordinate; Proposition 5)
                        - dense       : all 2^q vertices plus random points in the box (verification)
"""
import itertools
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import t as tdist
jax.config.update("jax_enable_x64", True)
Q95 = 1.959963984540054


# ------------------------------------------------------------------ Royston-Parmar model
def _standardise(A, mu=None, sd=None):
    A = np.asarray(A, float)
    if mu is None:
        mu = A.mean(0); sd = A.std(0); sd = np.where(sd > 0, sd, 1.0)
    return (A - mu) / sd, mu, sd


def _pos3(x):
    return jnp.where(x > 0, x, 0.0) ** 3


def _dpos3(x):
    return 3.0 * jnp.where(x > 0, x, 0.0) ** 2


def rp_basis(u, knots):
    kmin, kmax = knots[0], knots[-1]
    cols = [jnp.ones_like(u), u]
    for k in knots[1:-1]:
        lam = (kmax - k) / (kmax - kmin)
        cols.append(_pos3(u - k) - lam * _pos3(u - kmin) - (1 - lam) * _pos3(u - kmax))
    return jnp.stack(cols, axis=-1)


def rp_dbasis(u, knots):
    kmin, kmax = knots[0], knots[-1]
    cols = [jnp.zeros_like(u), jnp.ones_like(u)]
    for k in knots[1:-1]:
        lam = (kmax - k) / (kmax - kmin)
        cols.append(_dpos3(u - k) - lam * _dpos3(u - kmin) - (1 - lam) * _dpos3(u - kmax))
    return jnp.stack(cols, axis=-1)


def _rp_ll(th, u, d, Zs, Xs, knots, ng):
    g, bz, bx = th[:ng], th[ng:ng + Zs.shape[1]], th[ng + Zs.shape[1]:]
    s = rp_basis(u, knots) @ g; ds = rp_dbasis(u, knots) @ g
    eta = Zs @ bz + Xs @ bx
    return jnp.sum(d * (jnp.log(jnp.clip(ds, 1e-10)) - u + s + eta) - jnp.exp(s + eta))


_rp_nll = jax.jit(lambda *a: -_rp_ll(*a), static_argnums=6)
_rp_grad = jax.jit(jax.grad(lambda *a: -_rp_ll(*a)), static_argnums=6)
_rp_hess = jax.jit(jax.hessian(_rp_ll), static_argnums=6)


def _newton_polish(f, g, h, th, tol=1e-8, max_iter=8):
    """Newton-Raphson refinement of a minimiser of f (h = Hessian of -f) with step halving."""
    fv = f(th)
    for _ in range(max_iter):
        gr = g(th)
        if np.max(np.abs(gr)) < tol:
            break
        H = h(th)
        try:
            step = np.linalg.solve(H, gr)          # H is the log-likelihood Hessian (negative definite)
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        while lam > 1e-4:
            new = th + lam * step
            fn = f(new)
            if np.isfinite(fn) and fn <= fv + 1e-12:
                th, fv = new, fn; break
            lam /= 2
        else:
            break
    return th


def _np_basis(u, knots):
    knots = np.asarray(knots); kmin, kmax = knots[0], knots[-1]
    p3 = lambda x: np.where(x > 0, x, 0.0) ** 3; d3 = lambda x: 3.0 * np.where(x > 0, x, 0.0) ** 2
    B = [np.ones_like(u), u]; dB = [np.zeros_like(u), np.ones_like(u)]
    for k in knots[1:-1]:
        lam = (kmax - k) / (kmax - kmin)
        B.append(p3(u - k) - lam * p3(u - kmin) - (1 - lam) * p3(u - kmax))
        dB.append(d3(u - k) - lam * d3(u - kmin) - (1 - lam) * d3(u - kmax))
    return np.column_stack(B), np.column_stack(dB)


class _RPLik:
    """Analytic log-likelihood, gradient and Hessian of the Royston-Parmar PH model (numpy)."""

    def __init__(self, u, d, Zs, Xs, knots):
        self.B, self.dB = _np_basis(u, knots); self.u, self.d = u, d
        self.D = np.hstack([self.B, Zs, Xs]); self.ng = self.B.shape[1]

    def _parts(self, th):
        ds = self.dB @ th[:self.ng]; lin = self.D @ th
        return ds, lin

    def nll(self, th):
        ds, lin = self._parts(th)
        if np.any(ds <= 0):
            return np.inf
        return -float(np.sum(self.d * (np.log(ds) - self.u + lin) - np.exp(np.minimum(lin, 700))))

    def grad(self, th):                       # gradient of the negative log-likelihood
        ds, lin = self._parts(th); mu = np.exp(np.minimum(lin, 700))
        g = self.D.T @ (self.d - mu)
        g[:self.ng] += self.dB.T @ (self.d / np.clip(ds, 1e-10, None))
        return -g

    def hess(self, th):                       # Hessian of the log-likelihood
        ds, lin = self._parts(th); mu = np.exp(np.minimum(lin, 700))
        H = -(self.D.T * mu) @ self.D
        w = self.d / np.clip(ds, 1e-10, None) ** 2
        H[:self.ng, :self.ng] -= (self.dB.T * w) @ self.dB
        return H


class RoystonParmarPH:
    """log H(t|z,x) = s(log t) + z'b_z + x'b_x; s a natural cubic spline with `df` d.f.
    Covariates are standardised internally (theta and cov on the standardised scale)."""

    def __init__(self, df=3):
        self.df = df

    def fit(self, t, event, Z, X, theta0=None):
        t = np.asarray(t, float); event = np.asarray(event, float); u = np.log(t)
        self.knots = jnp.asarray(np.quantile(u[event == 1], np.linspace(0, 1, self.df + 1)))
        Zs, self.z_mu, self.z_sd = _standardise(Z)
        Xs, self.x_mu, self.x_sd = _standardise(X)
        Zs, Xs = np.ascontiguousarray(Zs), np.ascontiguousarray(Xs)
        ng = self.df + 1
        self.ix_g = np.arange(ng); self.ix_z = np.arange(ng, ng + Zs.shape[1])
        self.ix_x = np.arange(ng + Zs.shape[1], ng + Zs.shape[1] + Xs.shape[1])
        warm = theta0 is not None
        if theta0 is None:
            theta0 = np.zeros(ng + Zs.shape[1] + Xs.shape[1]); theta0[0] = np.log(event.sum() / t.sum()); theta0[1] = 1.0
        L = _RPLik(u, event, Zs, Xs, np.asarray(self.knots))
        f, g, h = L.nll, L.grad, L.hess
        th = None
        if warm:                                   # warm start: Newton-Raphson directly (quadratic convergence)
            th = _newton_polish(f, g, h, np.asarray(theta0, float), max_iter=25)
            if not np.all(np.isfinite(th)) or np.max(np.abs(g(th))) > 1e-6:
                th = None
        if th is None:
            res = minimize(f, theta0, jac=g, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-7})
            th = _newton_polish(f, g, h, res.x)
        self.theta = th
        self.cov = np.linalg.inv(-h(self.theta))
        self.loglik = -f(self.theta)
        return self

    def coef(self, block, j):
        k = (self.ix_z if block == "z" else self.ix_x)[j]
        s = (self.z_sd if block == "z" else self.x_sd)[j]
        return self.theta[k] / s, np.sqrt(self.cov[k, k]) / s

    def lp(self, Z, X):
        return ((np.asarray(Z) - self.z_mu) / self.z_sd) @ self.theta[self.ix_z] + \
               ((np.asarray(X) - self.x_mu) / self.x_sd) @ self.theta[self.ix_x]


# ------------------------------------------------------------------ logistic regression
def logit_fit(y, X):
    """Unpenalised logistic regression by Newton-Raphson. Returns (coefficients, SEs) per raw unit
    (intercept excluded) and the intercept as a third element."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, s = X.mean(0), X.std(0); s = np.where(s > 0, s, 1.0)
    Xd = np.hstack([np.ones((len(y), 1)), (X - mu) / s])
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(50):
        eta = Xd @ b; p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p); H = Xd.T @ (Xd * W[:, None]); gr = Xd.T @ (y - p)
        step = np.linalg.solve(H, gr); b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(Xd @ b)))
    cov = np.linalg.inv(Xd.T @ (Xd * (p * (1 - p))[:, None]))
    coef = b[1:] / s
    return coef, np.sqrt(np.diag(cov))[1:] / s, b[0] - np.sum(b[1:] * mu / s)


# ------------------------------------------------------------------ pooling
def rubin(ests, vars_, alpha=0.05):
    ests, vars_ = np.asarray(ests, float), np.asarray(vars_, float)
    m = len(ests); qb = ests.mean(0); W = vars_.mean(0)
    if m == 1:
        z = tdist.ppf(1 - alpha / 2, 1e6); return qb, qb - z * np.sqrt(W), qb + z * np.sqrt(W)
    B = ests.var(0, ddof=1); Tv = W + (1 + 1 / m) * B
    with np.errstate(divide="ignore", invalid="ignore"):
        dfr = np.where(B > 0, (m - 1) * (1 + W / ((1 + 1 / m) * B)) ** 2, 1e6)
    tq = tdist.ppf(1 - alpha / 2, dfr)
    return qb, qb - tq * np.sqrt(Tv), qb + tq * np.sqrt(Tv)


# ------------------------------------------------------------------ interval-summarised covariates
def error_variance_bound(lo, hi, n):
    """Bound (1): Var(U) <= (n-2) w^2 / (4 n^2), interior measurements treated as independent."""
    return np.maximum(n - 2, 0) * (hi - lo) ** 2 / (4 * np.maximum(n, 1) ** 2)


def regression_calibration(W, Z, sigma_u2):
    """E[X*|W,Z] under W = X* + U with U independent of (X*, Z, Y), E U = 0, Var U = diag(sigma_u2);
    computed from first and second moments. Error variance capped at 80% of Var(W)."""
    V = np.hstack([W, Z]); mu = V.mean(0); S = np.cov(V, rowvar=False)
    q = W.shape[1]
    sigma_u2 = np.minimum(sigma_u2, 0.8 * np.diag(S)[:q])
    C = S[:q, :].copy(); C[:, :q] -= np.diag(sigma_u2)
    return W.mean(0) + (V - mu) @ np.linalg.solve(S, C.T)


# ------------------------------------------------------------------ SMC-FCS for the RP model
def truncnorm_draw(rng, mu, sd, lo, hi):
    """Exact draws from N(mu, sd^2) truncated to [lo, hi] by inversion, computed in the tail that keeps
    precision (no clipping, so no artificial mass at the limits)."""
    from scipy.special import ndtr, ndtri
    mu = np.asarray(mu, float); a = (lo - mu) / sd; b = (hi - mu) / sd
    u = rng.uniform(size=mu.shape)
    x = np.empty_like(mu)
    up = a > 0                                     # whole interval in the upper tail: invert the survival function
    Ga, Gb = ndtr(-a[up]), ndtr(-b[up])
    x[up] = -ndtri(Gb + u[up] * (Ga - Gb))
    dn = ~up
    Fa, Fb = ndtr(a[dn]), ndtr(b[dn])
    x[dn] = ndtri(Fa + u[dn] * (Fb - Fa))
    return np.clip(mu + sd * x, lo, hi)            # clip only guards against rounding at the limits


def smcfcs_rp(T, E, Z, Xmiss, lo, hi, n_cycles=8, m=3, seed=0, max_tries=1000, theta0=None, return_diag=False):
    """Substantive-model-compatible FCS imputation (Bartlett et al. 2015) for the Royston-Parmar model.
    Proposals come from the normal linear covariate model (parameters drawn from their posterior) truncated
    to the variable's support [lo, hi]; they are accepted with probability exp(-H) if censored and
    H exp(1 - H) if died, with H the cumulative hazard under a parameter draw from the current model fit.
    Values not accepted after `max_tries` proposals keep their value from the previous cycle; the number of
    such values is reported in the diagnostics (return_diag=True)."""
    rng = np.random.default_rng(seed)
    Xm = np.asarray(Xmiss, float); M = np.isnan(Xm); n, q = Xm.shape
    miss_cols = [j for j in range(q) if M[:, j].any()]
    out = []; diag = dict(draws=0, not_accepted=0, proposals=0, final_cycle_not_accepted=0)
    for k in range(m):
        X = Xm.copy()
        for j in range(q):
            X[M[:, j], j] = np.nanmean(Xm[:, j])
        th_prev = theta0
        for cyc in range(n_cycles):
            rp = RoystonParmarPH(df=3).fit(T, E, Z, X, theta0=th_prev); th_prev = rp.theta
            th = rng.multivariate_normal(rp.theta, rp.cov)
            s = np.asarray(rp_basis(jnp.asarray(np.log(T)), rp.knots)) @ th[rp.ix_g]
            eta_z = ((Z - rp.z_mu) / rp.z_sd) @ th[rp.ix_z]
            bx = th[rp.ix_x] / rp.x_sd
            for j in miss_cols:
                rows = np.where(M[:, j])[0]
                D = np.hstack([np.ones((n, 1)), Z, np.delete(X, j, axis=1)])
                obs = ~M[:, j]; Do, yo = D[obs], X[obs, j]
                XtX = Do.T @ Do + 1e-8 * np.eye(D.shape[1])
                bh = np.linalg.solve(XtX, Do.T @ yo)
                res = yo - Do @ bh; dfree = max(obs.sum() - D.shape[1], 1)
                s2 = (res @ res) / rng.chisquare(dfree)
                bd = rng.multivariate_normal(bh, s2 * np.linalg.inv(XtX))
                mu = D[rows] @ bd
                eta_other = eta_z[rows] + (X[rows] - rp.x_mu) @ bx - (X[rows, j] - rp.x_mu[j]) * bx[j]
                pending = np.ones(len(rows), bool); draw = X[rows, j].copy()     # fall-back: previous value
                for _ in range(max_tries):
                    if not pending.any():
                        break
                    idx = np.where(pending)[0]
                    prop = truncnorm_draw(rng, mu[idx], np.sqrt(s2), lo[j], hi[j])
                    H = np.exp(np.minimum(s[rows[idx]] + eta_other[idx] + (prop - rp.x_mu[j]) * bx[j], 700.0))
                    acc_p = np.where(E[rows[idx]] == 1, H * np.exp(1.0 - H), np.exp(-H))
                    acc = rng.uniform(size=len(idx)) < acc_p
                    diag["proposals"] += len(idx)
                    draw[idx[acc]] = prop[acc]; pending[idx[acc]] = False
                X[rows, j] = draw
                diag["draws"] += len(rows); diag["not_accepted"] += int(pending.sum())
                if cyc == n_cycles - 1:
                    diag["final_cycle_not_accepted"] += int(pending.sum())
        out.append(X.copy())
    return (out, diag) if return_diag else out


# ------------------------------------------------------------------ envelope over mechanism parameters
class Envelope:
    """Union of core intervals over configurations of the MNAR shift delta (one coordinate per heavily
    missing variable). `core(delta_vector) -> (est, lo, hi)` arrays over the targets. Results are cached,
    so the different search strategies share their evaluations."""

    def __init__(self, core, q):
        self.core, self.q, self.cache = core, q, {}

    def eval(self, d):
        key = tuple(np.round(np.asarray(d, float), 10))
        if key not in self.cache:
            self.cache[key] = self.core(np.asarray(d, float))
        return self.cache[key]

    def _union(self, keys):
        lo = np.min([self.cache[k][1] for k in keys], 0); hi = np.max([self.cache[k][2] for k in keys], 0)
        return lo, hi

    def vertex_2q3(self, dmax):
        q = self.q; cfg = [np.zeros(q)]
        for j in range(q):
            for s in (1, -1):
                d = np.zeros(q); d[j] = s * dmax; cfg.append(d)
        cfg += [np.full(q, dmax), np.full(q, -dmax)]
        keys = [tuple(np.round(c, 10)) for c in cfg]
        for c in cfg:
            self.eval(c)
        return self._union(keys), len(set(keys))

    def adaptive(self, dmax):
        """2q+3 configurations plus the sign-matched vertex for each endpoint of each target."""
        (_, _), _ = self.vertex_2q3(dmax)
        q = self.q
        plus = [self.eval(np.eye(q)[j] * dmax) for j in range(q)]
        minus = [self.eval(-np.eye(q)[j] * dmax) for j in range(q)]
        n_t = len(plus[0][0]) if q else 0
        verts = []
        for t in range(n_t):
            s_lo = np.array([np.sign(plus[j][1][t] - minus[j][1][t]) for j in range(q)])
            s_hi = np.array([np.sign(plus[j][2][t] - minus[j][2][t]) for j in range(q)])
            verts.append(-dmax * np.where(s_lo == 0, 1, s_lo))     # minimises the lower endpoint
            verts.append(dmax * np.where(s_hi == 0, 1, s_hi))      # maximises the upper endpoint
        cfg = [np.zeros(q)] + [np.eye(q)[j] * s * dmax for j in range(q) for s in (1, -1)] + \
              [np.full(q, dmax), np.full(q, -dmax)] + verts
        keys = [tuple(np.round(c, 10)) for c in cfg]
        for c in cfg:
            self.eval(c)
        return self._union(keys), len(set(keys))

    def vertices(self, dmax):
        """Reference configuration and all 2^q vertices of the box (full vertex enumeration).
        Exact for the interval hull whenever each endpoint is monotone in each coordinate (Proposition 5)."""
        q = self.q
        cfg = [np.zeros(q)] + [np.array(v, float) * dmax for v in itertools.product((-1, 1), repeat=q)]
        keys = [tuple(np.round(c, 10)) for c in cfg]
        for c in cfg:
            self.eval(c)
        return self._union(keys), len(set(keys))

    def dense(self, dmax, n_random=100, rng=None):
        """All 2^q vertices plus n_random uniform points in the box (verification only)."""
        q = self.q; rng = rng or np.random.default_rng(0)
        cfg = [np.array(v) * dmax for v in itertools.product((-1, 1), repeat=q)] + \
              [rng.uniform(-dmax, dmax, q) for _ in range(n_random)] + [np.zeros(q)]
        keys = [tuple(np.round(c, 10)) for c in cfg]
        for c in cfg:
            self.eval(c)
        return self._union(keys), len(set(keys))


def coverage_summary(df, oracle="Oracle"):
    """Mean / lowest coverage across terms, Monte Carlo SE of the mean, median width relative to oracle."""
    d = df.copy(); d["cover"] = (d.lo <= d.truth) & (d.truth <= d.hi); d["w"] = d.hi - d.lo
    ow = d[d.method == oracle].groupby("term").w.mean()
    g = d.groupby(["method", "term"]).agg(cvg=("cover", "mean"), w=("w", "mean"), R=("cover", "size")).reset_index()
    g["mcse"] = np.sqrt(g.cvg * (1 - g.cvg) / g.R)
    g["wr"] = g.w / g.term.map(ow)
    s = g.groupby("method").agg(mean_coverage=("cvg", "mean"), lowest_coverage=("cvg", "min"),
                                width_vs_oracle=("wr", "median"), reps=("R", "min"))
    return s, g
