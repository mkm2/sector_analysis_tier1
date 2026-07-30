"""
Tier 1e analysis layer: per-rule sector descriptors and the hyperbola.

THE HYPERBOLA.  Sectors partition the basis, so with K ~ a^N sectors and
D_max ~ b^N,

    K * D_max  >=  sum_k D_k  =  2^N        =>        a * b >= 2,

with equality iff the sector sizes are all comparable.  This is an EXCLUSION
curve for the sector map: no rule may lie below b = 2/a.  It is a theorem about
partitions, so it constrains the WCC map and says nothing whatever about the
monitored-attractor map -- terminal SCCs do not partition, and there
a_att * b_att < 2 is expected, the deficit measuring transient dominance.

Anchors that must sit ON the curve: 204 (a=2, b=1), 51 (a=1, b=2), 150 obc0
(a=1, b=2, alpha=-1/2).  Anchor that must sit ABOVE: 156 (phi * 4^(1/5)).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .. import results_io
from ..core import rules
from .fits import (find_integer_recurrence, find_recurrence_by_parity,
                   fit_pure_exponential, fit_series, name_base)

ANALYTICS = os.path.join(results_io.REPO_ROOT, "analytics")
SECTOR_PATH = os.path.join(ANALYTICS, "sector_map_{bc}.json")

PHI = (1 + 5 ** 0.5) / 2

#: bases within this of 1 are not exponential growth
_KAPPA_BAND = 0.02

#: Rules whose sector law is DERIVED rather than fitted.  (base, alpha, source)
ANALYTIC: Dict[Tuple[int, str], Dict[str, Tuple[float, float, str]]] = {
    (204, "obc0"): {"n_wcc": (2.0, 0.0, "IIII: every state frozen"),
                    "d_max_wcc": (1.0, 0.0, "IIII: every sector a singleton")},
    (204, "pbc"): {"n_wcc": (2.0, 0.0, "IIII: every state frozen"),
                   "d_max_wcc": (1.0, 0.0, "IIII: every sector a singleton")},
    (51, "obc0"): {"n_wcc": (1.0, 0.0, "VVVV: one sector"),
                   "d_max_wcc": (2.0, 0.0, "VVVV: the whole space")},
    (51, "pbc"): {"n_wcc": (1.0, 0.0, "VVVV: one sector"),
                  "d_max_wcc": (2.0, 0.0, "VVVV: the whole space")},
    # R8 sec.4: |S_w| = C(N+1,w) on even w, #sectors = floor((N+1)/2)+1
    (150, "obc0"): {"n_wcc": (1.0, 1.0, "R8: floor((N+1)/2)+1, linear"),
                    "d_max_wcc": (2.0, -0.5, "R8: central binomial C(N+1,.)")},
    # 105 = VIIV fires when the neighbours AGREE, where 150 = IVVI fires when
    # they differ, so it is NOT simply 150's partner: at N = 1 (mod 4) its
    # sectors carry ODD wall number and the multiset differs.  N=9 gives
    # [252,120,120,10,10] = C(10, odd w) against 150's [210,210,45,45,1,1] =
    # C(10, even w), and likewise at N=13.  At all other N the two agree.  The
    # growth is linear either way, which is all the base captures.
    (105, "obc0"): {"n_wcc": (1.0, 1.0, "linear; cf. 150 but odd $w$ at "
                                        "$N\\equiv1\\ (4)$"),
                    "d_max_wcc": (2.0, -0.5, "R8: central binomial")},
}

# R2 sec.3 derives the D_max base of rule 156/198 from theory rather than data,
# and insists it stay that way.  Writing the chain as "rooms" of length l between
# domain walls, a room contributes l+1 states and consumes l+2 sites, so
# D_max = b^N with b(l) = (l+1)^{1/(l+2)}; the saddle point ln x = 1 + 1/x
# (x = l+1) sits at l ~ 2.5911, whose two integer neighbours are exactly the two
# candidate bases 3^{1/4} = 1.31607 and 4^{1/5} = 1.31951.  They differ by 0.26%
# and no finite series can choose between them -- the derivation does, and picks
# the larger.  At finite N the largest sector is a MIXTURE of rooms of length 2
# and 3, which is why neither pure recurrence is exact over the computed range.
#
# All six rules below share the identical WCC D_max series
# 6,9,12,16,20,27,36,48,64,81,108 (N=6..16), so the derivation covers all of them.
_ROOM_BASE = 4 ** 0.2
for _r in (140, 156, 196, 198, 206, 220):
    ANALYTIC.setdefault((_r, "obc0"), {})["d_max_wcc"] = (
        _ROOM_BASE, -0.5, "R2 sec.3: room-packing saddle point, $4^{1/5}$")


#: Largest N reached for EVERY rule.  The headline map fits inside this uniform
#: window so the bases are comparable across the whole rule space; rules that
#: happen to have a longer series (from a targeted extension) would otherwise be
#: fitted on a different domain, which is not a like-for-like comparison.
UNIFORM_N_CAP = 16


def load_series(rule: int, bc: str, n_cap: Optional[int] = None) -> Dict[str, List]:
    """The Tier-1e series for one rule, ergodic units included but flagged."""
    recs = results_io.load_wcc_results(rule, bc)
    if n_cap is not None:
        recs = {N: r for N, r in recs.items() if N <= n_cap}
    out = {"N": [], "n_wcc": [], "d_max_wcc": [], "n_frozen": [],
           "d_max_ratio": [], "ergodic": [], "transient_fraction": [],
           "att_per_sector": [], "n_recurrent": []}
    for N in sorted(recs):
        r = recs[N]
        if r.get("aborted"):
            continue
        out["N"].append(N)
        out["n_wcc"].append(r["n_wcc"])
        out["d_max_wcc"].append(r["d_max_wcc"])
        out["n_frozen"].append(r["n_frozen"])
        out["d_max_ratio"].append(r["d_max_ratio"])
        out["ergodic"].append(bool(r["ergodic_flag"]))
        out["transient_fraction"].append(r.get("transient_fraction"))
        out["att_per_sector"].append(r.get("att_per_sector"))
        out["n_recurrent"].append(r.get("n_recurrent"))
    return out


def _parity_split_helps(Ns, ys) -> bool:
    """Existing project rule: split parities when it halves the rms ln-residual."""
    import numpy as np
    if len(Ns) < 6:
        return False
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    good = ys > 0
    Ns, ys = Ns[good], ys[good]
    if len(Ns) < 6:
        return False
    lny = np.log(ys)

    def rms(n, y):
        if len(n) < 3:
            return None
        k, c = np.polyfit(n, y, 1)
        return float(np.sqrt(np.mean((y - (k * n + c)) ** 2)))

    full = rms(Ns, lny)
    ev = Ns % 2 == 0
    r_ev, r_od = rms(Ns[ev], lny[ev]), rms(Ns[~ev], lny[~ev])
    if full is None or r_ev is None or r_od is None or full == 0:
        return False
    return max(r_ev, r_od) < 0.5 * full


def is_irregular(Ns: List[int], ys: List[int]) -> bool:
    """
    A series that no growth law describes: strongly non-monotone with a large
    residual under the best log-linear fit.  W90/W165 are the type case --
    n_wcc = 2,1,4,4,2,6,2,6,12,1,20 -- where forcing a base out of the fitter
    yields 1.0 for both series and hence the impossible product ab = 1.
    """
    import numpy as np
    if len(ys) < 5:
        return False
    y = np.asarray(ys, float)
    if np.any(y <= 0):
        return True
    down = np.sum(np.diff(y) < 0)
    n = np.asarray(Ns, float)
    k, c = np.polyfit(n, np.log(y), 1)
    rms = float(np.sqrt(np.mean((np.log(y) - (k * n + c)) ** 2)))
    return bool(down >= 2 and rms > 0.35)


def finite_hyperbola(rule: int, bc: str,
                     n_cap: Optional[int] = None) -> Optional[Dict]:
    """
    The theorem at finite N, with no fitting anywhere:

        n_wcc(N) * D_max(N)  >=  2^N      for every N.

    This is an immediate corollary of the A1 sum rule and is the actual content
    of the exclusion curve.  The asymptotic statement ab >= 2 is a corollary
    about bases that only means anything when BOTH series are exponential; when
    one of them is sub-exponential the base product drops the polynomial factor
    and the test degenerates.  So this is the check that should carry the
    validation, and the base plane is the presentation.
    """
    s = load_series(rule, bc, n_cap)
    if not s["N"]:
        return None
    ratios = [(N, k * d / (1 << N))
              for N, k, d in zip(s["N"], s["n_wcc"], s["d_max_wcc"])]
    worst_N, worst = min(ratios, key=lambda t: t[1])
    return {"rule": rule, "bc": bc, "min_ratio": worst, "at_N": worst_N,
            "holds": bool(worst >= 1.0 - 1e-12), "n_units": len(ratios)}


def _saturated(ys: List[int], tail: int = 4) -> bool:
    """A bounded integer series that has stopped changing is CONSTANT, whatever
    the BIC says.  W36's sector count is 9,9,10,10,10,10,10,10,10,10,10; the M2
    fit reads the initial step as curvature and returns base 0.95 -- below 1,
    which is impossible for a count."""
    return len(ys) >= tail + 1 and len(set(ys[-tail:])) == 1


def _volume_fraction_base(Ns: List[int], ys: List[int], tail: int = 6):
    """
    Decide whether D_max = c * 2^N * N^alpha, i.e.\\ whether the base is EXACTLY 2
    with a POWER-LAW prefactor, by asking which model explains the residual
    r(N) = D_max/2^N better: a power of N or an exponential in N.

    This is the other half of R2 sec.3's warning.  There the trap was that M2's
    alpha*ln N absorbs growth and biases kappa DOWNWARD for rule 156; here the
    two-parameter fit has no alpha to absorb an N^{-1/2} prefactor, so it biases
    the base downward for the binomial rules instead -- W134's D_max is exactly
    rule 150's central binomials, base 2 with alpha = -1/2, and the pure
    exponential fit reports 1.9244.  Neither fit is trustworthy on its own; the
    discriminating question is the shape of the prefactor.

    Returns (2.0, alpha, source) when the power-law model wins, else None.
    """
    import numpy as np
    if len(ys) < tail or any(y <= 0 for y in ys):
        return None
    n = np.asarray(Ns[-tail:], float)
    r = np.log([y / (1 << N) for N, y in zip(Ns, ys)][-tail:])

    def rss(x):
        A = np.column_stack([np.ones_like(x), x])
        beta, res, *_ = np.linalg.lstsq(A, r, rcond=None)
        pred = A @ beta
        return float(((r - pred) ** 2).sum()), beta

    rss_pow, beta_pow = rss(np.log(n))      # r = c + alpha ln N  -> base 2
    rss_exp, _ = rss(n)                      # r = c + kappa N     -> base != 2
    if rss_pow < rss_exp:
        return (2.0, float(beta_pow[1]),
                r"$D_{\max}/2^N$ is a power of $N$, so the base is exactly $2$")
    return None


def series_descriptor(rule: int, bc: str, key: str,
                      Ns: List[int], ys: List[int]) -> Optional[Dict]:
    """
    (class, base, alpha, exact, base_lo/hi) for one Tier-1e series.

    Base priority: analytic override > exact integer recurrence (whole series,
    then per parity) > BIC fit.  The leave-one-out kappa spread of the M2 fit
    supplies the uncertainty band the hyperbola check needs.
    """
    an = ANALYTIC.get((rule, bc), {}).get(key)
    f = fit_series(Ns, ys)
    if not f.get("ok"):
        return None
    if an is None and is_irregular(Ns, ys):
        return {"cls": "irregular", "base": None, "alpha": None, "exact": False,
                "source": "non-monotone, no growth law", "parity_split": None,
                "base_lo": None, "base_hi": None, "n_points": f["n_points"],
                "N_range": f["N_range"], "named": None}
    if an is None:
        vf = _volume_fraction_base(Ns, ys)
        if vf is not None:
            return {"cls": "exponential", "base": vf[0], "alpha": vf[1],
                    "exact": True, "source": vf[2], "parity_split": None,
                    "base_lo": None, "base_hi": None,
                    "n_points": f["n_points"], "N_range": f["N_range"],
                    "named": name_base(vf[0])}
        if _saturated(ys):
            return {"cls": "constant", "base": 1.0, "alpha": 0.0, "exact": True,
                    "source": f"saturated at {ys[-1]}", "parity_split": None,
                    "base_lo": None, "base_hi": None,
                    "n_points": f["n_points"], "N_range": f["N_range"],
                    "named": None}
    cls, model = f["growth_class"], f["best_model"]
    if model == "M0":
        base, alpha = 1.0, 0.0
    elif model == "M1":
        base, alpha = 1.0, float(f["params"]["M1"][1])
    else:
        # R2 sec.3, "a second, purely numerical trap": M2's alpha*ln N term
        # absorbs part of the growth, so its kappa is a BIASED rate estimator --
        # for rule 156 it reports 1.2554 against a true 4^{1/5} = 1.3195.  M2 is
        # the right tool for the growth CLASS and for the sub-leading power, but
        # the RATE must come from the two-parameter fit ln y = c + kappa N (or
        # from an exact recurrence, handled below).
        alpha = float(f["alpha_M2"] or 0.0)
        if cls == "polynomial":
            # M2 was selected but its kappa is within the noise band, so the
            # series is a power of N, not an exponential.  Reporting a base of
            # 1.09 next to the class "polynomial" is self-contradictory, and it
            # is what put six linear-sector-count rules (44, 100, 110, 124, 188,
            # 230, whose counts run 7,8,...,17) at a > 1.
            base = 1.0
        else:
            fe = fit_pure_exponential(Ns, ys)
            base = (float(fe["base"]) if fe.get("ok")
                    else float(f["base"] or 1.0))

    lo = hi = None
    if f.get("kappa_loo_range"):
        lo, hi = (math.exp(f["kappa_loo_range"][0]),
                  math.exp(f["kappa_loo_range"][1]))

    exact, source = False, f"BIC {model}"
    parity = None
    rec = find_integer_recurrence(list(ys))
    if rec.get("ok") and rec["base"] > 1.0:
        base, exact, source = rec["base"], True, "integer recurrence"
    elif _parity_split_helps(Ns, ys):
        pr = find_recurrence_by_parity(list(Ns), list(ys))
        if pr.get("ok") and pr.get("base", 0) > 1.0:
            base, exact, source = pr["base"], True, "parity recurrence"
            parity = True
        else:
            en = [n for n in Ns if n % 2 == 0]
            ey = [y for n, y in zip(Ns, ys) if n % 2 == 0]
            # Classify the parity BRANCH, do not merely fit it: rule 44's even-N
            # sector counts are 7,9,11,13,15,17, a straight line, and a pure
            # exponential fit to a straight line still returns a base of 1.09.
            ef = fit_series(en, ey)
            fe = fit_pure_exponential(en, ey)
            if fe.get("ok") and ef.get("ok"):
                if ef["growth_class"] == "exponential":
                    base, source, parity = float(fe["base"]), "parity fit", True
                else:
                    base, alpha, source, parity = 1.0, alpha, \
                        f"parity branch {ef['growth_class']}", True
    # A base recovered from a parity split or an exact recurrence overrides the
    # class BIC assigned to the mixed series: rule 29's sector count is
    # 3,6,5,9,7,14,10,21,15,31,22, which BIC reads as polynomial because the
    # parity oscillation swamps the trend, while each parity branch plainly grows
    # exponentially (base 1.215).  Class and base have to agree.
    if base > 1.0 + _KAPPA_BAND and (exact or parity):
        cls = "exponential"

    if an is not None:
        base, alpha, exact, source = an[0], an[1], True, an[2]
        lo = hi = None

    # Both n_wcc and D_max lie in [1, 2^N], so both bases lie in [1, 2].  A fit
    # outside that window contradicts a theorem, so the theorem wins and the
    # value is clamped -- with the clamp recorded, never silent.  This is what
    # the binomial rules need: W134's D_max is 924, 1716, 3003, 6435, 12870,
    # 24310, i.e. exactly rule 150's central binomials ~ 2^N/sqrt(N), and the M2
    # fit lands at 2.0588 because the N^{-1/2} prefactor pulls the estimate past
    # the boundary.
    clamped = None
    if base > 2.0 + 1e-12:
        clamped, base, exact = f"fit {base:.4f} clamped to 2", 2.0, False
    elif base < 1.0 - 1e-12:
        clamped, base, exact = f"fit {base:.4f} clamped to 1", 1.0, False
    if clamped:
        source = f"{source}; {clamped}"

    return {"cls": cls, "base": float(base), "alpha": float(alpha),
            "clamped": clamped,
            "exact": bool(exact), "source": source, "parity_split": parity,
            "base_lo": lo, "base_hi": hi, "n_points": f["n_points"],
            "N_range": f["N_range"], "named": name_base(float(base))}


def rule_point(rule: int, bc: str,
               n_cap: Optional[int] = None) -> Optional[Dict]:
    s = load_series(rule, bc, n_cap)
    if len(s["N"]) < 3:
        return None
    t = rules.wolfram_to_tuple(rule)
    a = series_descriptor(rule, bc, "n_wcc", s["N"], s["n_wcc"])
    b = series_descriptor(rule, bc, "d_max_wcc", s["N"], s["d_max_wcc"])
    if a is None or b is None:
        return None
    fam = ("unitary" if rules.is_unitary(t)
           else ("classical" if "V" not in t else "mixed"))
    n_max = max(s["N"])
    i = s["N"].index(n_max)
    return {
        "rule": rule, "bc": bc, "tuple": "".join(t), "family": fam,
        "unitary": rules.is_unitary(t),
        "ergodic": bool(s["ergodic"][i]),
        "n_wcc": a, "d_max_wcc": b,
        "N_max": n_max,
        "n_frozen_at_Nmax": s["n_frozen"][i],
        "d_max_ratio_at_Nmax": s["d_max_ratio"][i],
        "transient_fraction_at_Nmax": s["transient_fraction"][i],
        "att_per_sector_at_Nmax": s["att_per_sector"][i],
        "product_ab": (a["base"] * b["base"]
                       if (a["base"] is not None and b["base"] is not None)
                       else None),
    }


#: A slack wider than this makes the test vacuous; such rules are reported as
#: inconclusive rather than counted as passes.
SLACK_VACUOUS = 0.30


def hyperbola_check(pt: Dict, tol: float = 0.0) -> Dict:
    """
    V1: a * b >= 2, with the fitted-base uncertainty (leave-one-out kappa
    spread) propagated into a slack.

    Three things this deliberately does NOT do.  It does not let the slack
    launder a sub-2 product into a silent pass -- `raw_below` is recorded
    independently of `verdict`, and the task requires every offender to be
    listed.  It does not treat a huge slack as a pass: propagating a loose
    leave-one-out band through the other base can produce a slack of 18, which
    would make the constraint meaningless, so those come out `inconclusive`.
    And it does not forget that the inequality is a THEOREM: whatever the fit
    says, b >= 2/a must hold, so `b_lower_bound` is the value the hyperbola
    predicts for the sub-leading base, which is often sharper than the fit.
    """
    a, b = pt["n_wcc"], pt["d_max_wcc"]
    if a["base"] is None or b["base"] is None:      # irregular series
        return {"rule": pt["rule"], "bc": pt["bc"], "product": None,
                "margin": None, "slack": None, "raw_below": False,
                "within_slack": None, "verdict": "irregular", "ok": True,
                "exact_a": False, "exact_b": False,
                "a": a["base"], "b": b["base"], "bound_violations": [],
                "b_lower_bound": None, "b_fit_hi": None, "on_curve": False}
    prod = a["base"] * b["base"]
    slack = tol
    for d, other in ((a, b), (b, a)):
        if not d["exact"] and d.get("base_lo") and d.get("base_hi"):
            slack = max(slack, abs(d["base_hi"] - d["base_lo"]) * other["base"])
    raw_below = bool(prod < 2.0 - 1e-9)
    within = bool(prod >= 2.0 - slack - 1e-9)
    # When either series is sub-exponential its base is 1 by convention, so the
    # product silently drops the polynomial factor and ab >= 2 is no longer the
    # theorem -- for a polynomial sector count the theorem degenerates to
    # b >= 2, approached from below at finite N.  Such rules are reported as
    # degenerate, not as violations; finite_hyperbola() carries the real check.
    degenerate = (a["cls"] in ("constant", "polynomial")
                  or b["cls"] in ("constant", "polynomial"))
    if not raw_below:
        verdict = "on_curve" if abs(prod - 2.0) < 1e-6 else "above"
    elif degenerate:
        verdict = "degenerate_subexponential"
    elif slack > SLACK_VACUOUS:
        verdict = "inconclusive"
    elif within:
        verdict = "below_within_uncertainty"
    else:
        verdict = "VIOLATION"
    # Physical bounds, from the same partition logic as the hyperbola:
    # 1 <= n_wcc <= 2^N and 1 <= D_max <= 2^N force 1 <= a,b <= 2.  A fitted
    # base outside that window is a mis-fit by inspection, no series length
    # argument required, and must be reported rather than plotted as data.
    bounds = []
    if not (1.0 - 1e-9 <= a["base"] <= 2.0 + 1e-9):
        bounds.append(f"a={a['base']:.4f} outside [1,2]")
    if not (1.0 - 1e-9 <= b["base"] <= 2.0 + 1e-9):
        bounds.append(f"b={b['base']:.4f} outside [1,2]")

    b_lo = 2.0 / a["base"] if a["base"] > 0 else None
    return {"rule": pt["rule"], "bc": pt["bc"], "product": prod,
            "bound_violations": bounds,
            "margin": prod - 2.0, "slack": slack,
            "raw_below": raw_below, "within_slack": within,
            "verdict": verdict,
            "ok": verdict != "VIOLATION",
            "exact_a": a["exact"], "exact_b": b["exact"],
            "a": a["base"], "b": b["base"],
            "b_lower_bound": b_lo,
            "b_fit_hi": b.get("base_hi"),
            "on_curve": bool(abs(prod - 2.0) < 1e-6)}


def attractor_deficit(rule: int, bc: str,
                      n_cap: Optional[int] = None) -> Optional[Dict]:
    """
    V4: the monitored-attractor map has NO hyperbola constraint.  Report the
    deficit 2 - a_att*b_att as an observable measuring transient dominance.
    Assert nothing.

    The descriptors are built with the SAME convention as the sector ones
    (series_descriptor: exact recurrence > analytic > two-parameter fit, never
    the biased M2 rate of R2 sec.3).  Using growth_map's conventions here
    instead would make the two maps incomparable and would throw away a free
    visual check -- A2 says WCC and terminal SCC coincide for a unitary rule, so
    those 16 points must land in the same place in both panels.  For that reason
    a unitary rule also inherits the SECTOR analytic overrides (its two series
    are the same series); a dissipative rule must not.
    """
    from .dissipative import load_series as load_att_series
    try:
        s = load_att_series(rule, bc)
    except Exception:
        return None
    if not s or len(s.get("N") or []) < 3:
        return None
    # Same N window as the sector series, or the bases are fitted on different
    # domains and the A2 check fails for a bookkeeping reason (W60/W102 differed
    # by 0.014 purely because the Tier-1a series runs past the sector cap).
    if n_cap is not None:
        keep = [i for i, N in enumerate(s["N"]) if N <= n_cap]
        if len(keep) < 3:
            return None
        s = {k: [v[i] for i in keep] for k, v in s.items()
             if isinstance(v, list) and len(v) == len(s["N"])}
    unitary = rules.is_unitary(rules.wolfram_to_tuple(rule))
    ka = "n_wcc" if unitary else "n_recurrent"
    kb = "d_max_wcc" if unitary else "d_max"
    a = series_descriptor(rule, bc, ka, s["N"], s["n_recurrent"])
    b = series_descriptor(rule, bc, kb, s["N"], s["d_max"])
    if not a or not b or a.get("base") is None or b.get("base") is None:
        return None
    prod = a["base"] * b["base"]
    return {"rule": rule, "bc": bc, "a_att": a["base"], "b_att": b["base"],
            "product": prod, "deficit": 2.0 - prod}


def build(bc: str = "obc0", n_cap: Optional[int] = UNIFORM_N_CAP) -> Dict:
    pts, checks, deficits, finite = [], [], [], []
    for rule in range(256):
        p = rule_point(rule, bc, n_cap)
        if p is None:
            continue
        pts.append(p)
        checks.append(hyperbola_check(p))
        fh = finite_hyperbola(rule, bc, n_cap)
        if fh:
            finite.append(fh)
        d = attractor_deficit(rule, bc, n_cap)
        if d:
            deficits.append(d)
    tally: Dict[str, int] = {}
    for c in checks:
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    out = {"bc": bc, "n_cap": n_cap, "n_rules": len(pts), "points": pts,
           "hyperbola": checks, "attractor_deficits": deficits,
           "verdicts": tally,
           "finite_hyperbola": finite,
           "finite_hyperbola_failures": [f for f in finite if not f["holds"]],
           # every rule whose product is below 2, whatever the slack says --
           # the task forbids letting these pass silently
           "raw_below": [c for c in checks if c["raw_below"]],
           "bound_violations": [c for c in checks if c["bound_violations"]],
           "violations": [c for c in checks if c["verdict"] == "VIOLATION"]}
    os.makedirs(ANALYTICS, exist_ok=True)
    with open(SECTOR_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = SECTOR_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Tier 1e sector map + hyperbola")
    ap.add_argument("--bc", default="obc0", choices=["pbc", "obc0"])
    args = ap.parse_args(argv)
    d = build(args.bc)
    print(f"{d['n_rules']} rules with a Tier-1e descriptor ({args.bc})")
    print("verdicts:", d["verdicts"])
    ff = d["finite_hyperbola_failures"]
    worst = min(d["finite_hyperbola"], key=lambda f: f["min_ratio"])
    print(f"finite-N check n_wcc(N)*Dmax(N) >= 2^N : "
          f"{len(d['finite_hyperbola']) - len(ff)}/{len(d['finite_hyperbola'])} "
          f"hold; tightest is W{worst['rule']} at N={worst['at_N']} "
          f"with ratio {worst['min_ratio']:.4f}")
    for f in ff[:10]:
        print(f"   FAIL W{f['rule']} at N={f['at_N']}: ratio {f['min_ratio']:.6f}")
    print(f"physically impossible fitted bases (must lie in [1,2]): "
          f"{len(d['bound_violations'])}")
    for v in d["bound_violations"]:
        print(f"   W{v['rule']:<4} {'; '.join(v['bound_violations'])}")
    print(f"rules with a*b < 2 (all listed, whatever the slack): "
          f"{len(d['raw_below'])}")
    for v in sorted(d["raw_below"], key=lambda c: c["product"])[:20]:
        print(f"   W{v['rule']:<4} a={v['a']:.4f} b={v['b']:.4f} "
              f"a*b={v['product']:.4f}  slack={v['slack']:.3f}  "
              f"{v['verdict']}   [hyperbola forces b >= "
              f"{v['b_lower_bound']:.4f}; fit hi = "
              f"{v['b_fit_hi'] if v['b_fit_hi'] is None else round(v['b_fit_hi'], 4)}]")
    anchors = {204: 2.0, 51: 2.0, 150: 2.0}
    for r, want in anchors.items():
        c = next((c for c in d["hyperbola"] if c["rule"] == r), None)
        if c:
            print(f"anchor W{r}: a*b = {c['product']:.6f} "
                  f"(want {want}) on_curve={c['on_curve']}")
    c = next((c for c in d["hyperbola"] if c["rule"] == 156), None)
    if c:
        print(f"anchor W156: a*b = {c['product']:.4f} (must be > 2) "
              f"-> {'ABOVE' if c['product'] > 2 else 'NOT ABOVE'}")




# --- fragmentation that survives dissipation ---------------------------------

def open_system_fragmented(d: Dict) -> List[Dict]:
    """
    The V+reset rules that keep an EXPONENTIALLY growing sector count.

    These are the interesting ones physically.  A sector is an enclosure --
    span(WCC) is invariant under every Kraus operator -- so an exponentially
    growing sector count is exact for the unmonitored channel as well as for the
    measured circuit.  A rule in this list therefore exhibits Hilbert-space
    fragmentation as an OPEN quantum system, not merely as a unitary circuit that
    someone later added noise to.

    Selection is deliberately narrow: the rule must contain a reset (so it is
    genuinely dissipative), its coherent correspondent must itself have sector
    structure (otherwise there was nothing to survive), and its own sector count
    must be classified exponential with a base bounded away from 1.
    """
    xy = {p["rule"]: p for p in d["points"]}
    out = []
    for parent in sorted(rules.UNITARY_RULES):
        pp = xy.get(parent)
        if pp is None or pp["n_wcc"]["base"] is None:
            continue
        pa, pb = pp["n_wcc"]["base"], pp["d_max_wcc"]["base"]
        if abs(pa - 1) < 1e-6 and abs(pb - 2) < 1e-6:
            continue                       # parent had nothing to lose
        for c in rules.dissipative_children(parent):
            q = xy.get(c)
            if q is None or q["n_wcc"]["base"] is None:
                continue
            if q["n_wcc"]["cls"] != "exponential":
                continue
            if q["n_wcc"]["base"] <= 1.0 + _KAPPA_BAND:
                continue
            out.append({
                "rule": c, "tuple": q["tuple"], "parent": parent,
                "parent_tuple": pp["tuple"],
                "a": q["n_wcc"]["base"], "b": q["d_max_wcc"]["base"],
                "product": q["product_ab"],
                "a_named": q["n_wcc"]["named"], "b_named": q["d_max_wcc"]["named"],
                "a_exact": q["n_wcc"]["exact"], "a_source": q["n_wcc"]["source"],
                "parent_a": pa, "parent_b": pb,
            })
    return sorted(out, key=lambda r: (-r["a"], r["rule"]))

def sector_growth_class(rule: int, bc: str, d: Dict,
                        n_cap: Optional[int] = UNIFORM_N_CAP) -> str:
    """
    'constant' / 'polynomial' / 'exponential' / 'irregular' for the SECTOR COUNT.

    The base plane cannot express this: a rule with a linear sector count and one
    with a single sector both sit at a = 1, so rule 150 (which has
    floor((N+1)/2)+1 sectors, 9 of them at N=16) is plotted on top of rule 51
    (which has exactly one).  Survival questions have to be asked in terms of the
    growth CLASS, not the base.
    """
    pts = {p["rule"]: p for p in d["points"]}
    q = pts.get(rule)
    if q is None:
        return "unknown"
    if q["n_wcc"]["base"] is None:
        return "irregular"
    ys = load_series(rule, bc, n_cap)["n_wcc"]
    if ys and len(set(ys)) == 1:
        return "constant"
    return q["n_wcc"]["cls"]


def survival_by_parent_class(d: Dict, bc: str = "obc0") -> Dict:
    """
    What survives a reset, resolved by the growth class of the coherent parent.

    This is the question the base plane cannot answer.  Grouping the 14 clusters
    by the parent's sector-growth class and asking what class each child lands in
    separates three quite different fates, where the (a,b) = (1,2) cell had
    conflated them.
    """
    out: Dict[str, Dict] = {}
    detail: List[Dict] = []
    for parent in sorted(rules.UNITARY_RULES):
        kids = rules.dissipative_children(parent)
        if not kids:
            continue
        pcls = sector_growth_class(parent, bc, d)
        row = out.setdefault(pcls, {"parents": [], "children": 0,
                                    "exponential": 0, "polynomial": 0,
                                    "constant": 0, "irregular": 0})
        row["parents"].append(parent)
        for c in kids:
            ccls = sector_growth_class(c, bc, d)
            if ccls == "unknown":
                continue
            row["children"] += 1
            row[ccls] = row.get(ccls, 0) + 1
            detail.append({"rule": c, "parent": parent,
                           "parent_class": pcls, "class": ccls})
    return {"by_parent_class": out, "detail": detail}


if __name__ == "__main__":
    main()
