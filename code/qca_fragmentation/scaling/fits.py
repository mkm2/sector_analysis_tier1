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
from typing import Dict, List, Sequence

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
