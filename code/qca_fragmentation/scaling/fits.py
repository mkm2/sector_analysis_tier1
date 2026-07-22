"""
Tier 1c scaling extraction (context Tier 1 sec.7).

For a per-rule series y(N) (n_recurrent, max_recurrent = Dmax, or max_basin) we
fit, on exact ln values, three nested models and select by BIC:

    M0:  ln y = c
    M1:  ln y = c + alpha ln N                      (power law y ~ N^alpha)
    M2:  ln y = c + alpha ln N + kappa N            (exponential y ~ e^{kappa N})

The alpha ln N term is ALWAYS carried when reporting kappa, because binomial /
charge-conserving sectors carry sqrt(N)-type prefactors (alpha ~ -1/2) that
otherwise bias kappa.  kappa is reported with a leave-one-out spread.

Growth class label (for the final figure's marker shape):
    constant     -> best model M0
    polynomial   -> best model M1 (or M2 with kappa ~ 0)
    exponential  -> best model M2 with kappa clearly > 0
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

# |kappa| below this is treated as non-exponential (base < e^0.02 ~ 1.02).
# Genuine exponential fragmentation here has kappa >= 0.27 (base 4^{1/5}) up to
# ln 2; a linear sector count (e.g. rule 60: n_sec = N+1) yields a spurious
# kappa ~ 0.008 that must NOT be labelled exponential.
_KAPPA_EPS = 0.02


def _design(Ns: np.ndarray, model: str) -> np.ndarray:
    lnN = np.log(Ns)
    if model == "M0":
        return np.ones((len(Ns), 1))
    if model == "M1":
        return np.column_stack([np.ones_like(Ns, dtype=float), lnN])
    if model == "M2":
        return np.column_stack([np.ones_like(Ns, dtype=float), lnN, Ns.astype(float)])
    raise ValueError(model)


def _fit(Ns: np.ndarray, lny: np.ndarray, model: str):
    X = _design(Ns, model)
    beta, *_ = np.linalg.lstsq(X, lny, rcond=None)
    resid = lny - X @ beta
    rss = float(resid @ resid)
    return beta, rss


def _bic(rss: float, n: int, k: int, floor: float) -> float:
    # Gaussian BIC up to additive const; k = number of fitted params.  The RSS
    # floor prevents n*log(RSS) from being driven by floating-point noise when a
    # model fits essentially perfectly (then all such models tie and the fewest-
    # parameter one wins on the k*log(n) penalty).
    rss = max(rss, floor)
    return n * math.log(rss / n) + k * math.log(n)


def fit_series(Ns: Sequence[int], ys: Sequence[int]) -> Dict:
    """
    Fit M0/M1/M2 to ln y vs N and select by BIC.  Returns a dict with the
    selected model, all parameters, the M2 kappa (+leave-one-out spread), and a
    growth-class label.  Requires >= 3 distinct N; returns {'ok': False} else.
    """
    Ns = np.asarray(list(Ns), dtype=float)
    ys = np.asarray(list(ys), dtype=float)
    good = ys > 0
    Ns, ys = Ns[good], ys[good]
    if len(Ns) < 3 or len(set(Ns.tolist())) < 3:
        return {"ok": False, "reason": "need >=3 positive points"}
    lny = np.log(ys)
    n = len(Ns)

    # RSS noise floor scaled to the data spread: a fit tighter than this is
    # "perfect" for model-selection purposes.
    ss_total = float(((lny - lny.mean()) ** 2).sum())
    floor = 1e-9 * max(1.0, ss_total)

    fits = {}
    for model, k in (("M0", 1), ("M1", 2), ("M2", 3)):
        if n <= k:
            continue
        beta, rss = _fit(Ns, lny, model)
        fits[model] = {"beta": beta, "rss": rss,
                       "bic": _bic(rss, n, k, floor), "k": k}

    best = min(fits, key=lambda m: fits[m]["bic"])

    # M2 kappa (carry alpha ln N) with leave-one-out spread
    kappa = None
    kappa_loo = None
    alpha_full = None
    if "M2" in fits:
        beta2 = fits["M2"]["beta"]
        alpha_full = float(beta2[1])
        kappa = float(beta2[2])
        loo = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if mask.sum() > 3:
                b, _ = _fit(Ns[mask], lny[mask], "M2")
                loo.append(float(b[2]))
        if loo:
            kappa_loo = (float(np.min(loo)), float(np.max(loo)))

    if best == "M0":
        growth = "constant"
    elif best == "M1":
        growth = "polynomial"
    else:  # M2 selected
        growth = "exponential" if (kappa is not None and abs(kappa) > _KAPPA_EPS) else "polynomial"

    out = {
        "ok": True,
        "n_points": n,
        "N_range": [int(Ns.min()), int(Ns.max())],
        "best_model": best,
        "growth_class": growth,
        "kappa": kappa,                 # exponential rate (coeff of N in M2)
        "kappa_loo_range": kappa_loo,
        "alpha_M2": alpha_full,         # power of N alongside kappa
        "base": (math.exp(kappa) if kappa is not None else None),
        "params": {m: fits[m]["beta"].tolist() for m in fits},
        "bic": {m: fits[m]["bic"] for m in fits},
    }
    # convenient power-law slope when M1 is best
    if best == "M1":
        out["alpha"] = float(fits["M1"]["beta"][1])
    return out


# --- pure exponential rate ---------------------------------------------------
# M2 carries an  alpha ln N  term so that binomial prefactors do not bias kappa.
# That is right for the clean, monotone unitary series, but it is unstable on
# the ragged dissipative ones: with few points alpha absorbs the curvature and
# kappa runs away (rule 90's D_max = 1,15,1,7,3,63,2,127 yields alpha = -7.5 and
# a "base" of 11, when D_max <= 2^N forces base <= 2).  For those series we fit
# the two-parameter model  ln y = c + kappa N  instead, which cannot trade the
# rate against a power and stays inside the physical bound.

LN2 = math.log(2.0)

# Named algebraic constants that show up as growth bases, for labelling.
_NAMED_BASES = [
    (2.0, "2"),
    (1.9659482366, "root of $x^3=2x^2-1$"),
    # The square-root forms below all come from parity-doubled recurrences
    # (a(n) = c_2 a(n-2) + c_4 a(n-4)): the characteristic polynomial is a
    # quadratic in x^2, so the base is the square root of a quadratic surd.
    (1.8872076761, "$\\sqrt{(3+\\sqrt{17})/2}$"),
    (1.8477590650, "$\\sqrt{2+\\sqrt2}=2\\cos(\\pi/8)$"),
    (1.8392867552, "tribonacci $\\psi$"),
    (1.7989074399, "$\\sqrt{1+\\sqrt5}$"),
    (1.7548776662, "root of $x^3=2x^2-x+1$"),
    (1.7320508076, "$\\sqrt{3}$"),
    (1.6528916503, "$\\sqrt{1+\\sqrt3}$"),
    (1.6180339887, "golden $\\varphi$"),
    (1.4655712319, "supergolden $\\psi$"),
    (1.4142135624, "$\\sqrt{2}$"),
    (1.3802775691, "root of $x^4=x^3+1$"),
    (1.3247179572, "plastic $\\rho$"),
    (1.2599210499, "$2^{1/3}$"),
    (1.1892071150, "$2^{1/4}$"),
]


def name_base(x: float, tol: float = 1e-6) -> Optional[str]:
    """Return a label if `x` matches a known algebraic constant."""
    if x is None:
        return None
    for v, nm in _NAMED_BASES:
        if abs(x - v) < tol:
            return nm
    return None


def find_integer_recurrence(seq: Sequence[int], max_order: int = 4,
                            coeff_max: int = 30, step: int = 1) -> Dict:
    """Smallest-order EXACT integer linear recurrence a(n) = sum_i c_i a(n-i).

    Solves the first `order` equations over the rationals, keeps the solution
    only if the coefficients are integers, and then VERIFIES it against every
    remaining term (so a returned recurrence is exact on the whole series, not
    a fit).  The growth base is the largest |root| of the characteristic
    polynomial -- an algebraic number, not a regression estimate.

    `step` is the spacing in N between consecutive terms of `seq`.  It is 2 when
    the caller has split a parity-oscillating series into its even and odd
    subsequences, and then the root gives the growth per TWO sites, so the
    reported base is its `step`-th root -- otherwise a parity-split rule would
    be credited with the square of its actual growth base.
    """
    seq = [int(v) for v in seq]
    for order in range(1, max_order + 1):
        if len(seq) < 2 * order + 1:
            break
        M = [[Fraction(seq[n - 1 - i]) for i in range(order)] + [Fraction(seq[n])]
             for n in range(order, 2 * order)]
        for c in range(order):                       # Gauss-Jordan over Q
            p = next((r for r in range(c, order) if M[r][c] != 0), None)
            if p is None:
                break
            M[c], M[p] = M[p], M[c]
            pv = M[c][c]
            M[c] = [x / pv for x in M[c]]
            for r in range(order):
                if r != c and M[r][c] != 0:
                    f = M[r][c]
                    M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        else:
            coeffs = [M[i][order] for i in range(order)]
            if not all(x.denominator == 1 and abs(x) <= coeff_max for x in coeffs):
                continue
            c = [int(x) for x in coeffs]
            if not all(seq[k] == sum(c[i] * seq[k - 1 - i] for i in range(order))
                       for k in range(order, len(seq))):
                continue
            roots = np.roots([1] + [-x for x in c])
            base = float(max(abs(roots))) ** (1.0 / step)
            # A polynomial series has a repeated root at exactly 1, and np.roots
            # splits a triple root into a cluster ~1e-5 wide -- (x-1)^3 comes
            # back as 1.000007.  Snap it, or a polynomial growth law would be
            # reported with a spurious exponential base.  Nothing genuine sits
            # this close to 1: the smallest real base observed is 1.19.
            if abs(base - 1.0) < 1e-4:
                base = 1.0
            return {"ok": True, "order": order, "coeffs": c, "step": step,
                    "base": base, "name": name_base(base)}
    return {"ok": False}


def find_recurrence_by_parity(Ns: Sequence[int], seq: Sequence[int],
                              **kw) -> Dict:
    """Exact recurrence for a parity-oscillating series, per parity.

    A series that jumps between even and odd N obeys no recurrence as a whole,
    but each parity subsequence generally does.  We accept the result only when
    BOTH parities have one and they agree on the base to 1e-9 -- one parity
    alone is half the data and could be a coincidence at these lengths, and
    disagreeing bases mean there is no single growth base to report.
    """
    got = {}
    for name, keep in (("even", 0), ("odd", 1)):
        sub = [y for n, y in zip(Ns, seq) if n % 2 == keep]
        got[name] = find_integer_recurrence(sub, step=2, **kw)
    if not (got["even"]["ok"] and got["odd"]["ok"]):
        return {"ok": False}
    if abs(got["even"]["base"] - got["odd"]["base"]) > 1e-9:
        return {"ok": False}
    out = dict(got["even"])
    out["parity"] = True
    out["coeffs_odd"] = got["odd"]["coeffs"]
    return out


def fit_pure_exponential(Ns: Sequence[int], ys: Sequence[int]) -> Dict:
    """Least-squares ln y = c + kappa N.  Returns kappa, base, rms residual.

    `bounded` is False when the fitted base exceeds 2, which is impossible for
    any series bounded by the Hilbert-space dimension 2^N and therefore marks
    the series as too irregular for an exponential description.
    """
    Ns = np.asarray(list(Ns), dtype=float)
    ys = np.asarray(list(ys), dtype=float)
    good = ys > 0
    Ns, ys = Ns[good], ys[good]
    if len(Ns) < 2 or len(set(Ns.tolist())) < 2:
        return {"ok": False, "reason": "need >=2 positive points"}
    lny = np.log(ys)
    X = np.column_stack([np.ones_like(Ns), Ns])
    beta, *_ = np.linalg.lstsq(X, lny, rcond=None)
    resid = lny - X @ beta
    kappa = float(beta[1])
    return {
        "ok": True,
        "n_points": int(len(Ns)),
        "kappa": kappa,
        "base": math.exp(kappa),
        "rms": float(np.sqrt((resid ** 2).mean())),
        "bounded": kappa <= LN2 + 1e-6,
    }
