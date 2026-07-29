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
    (105, "obc0"): {"n_wcc": (1.0, 1.0, "R8: reflection partner of 150"),
                    "d_max_wcc": (2.0, -0.5, "R8: central binomial")},
}


def load_series(rule: int, bc: str) -> Dict[str, List]:
    """The Tier-1e series for one rule, ergodic units included but flagged."""
    recs = results_io.load_wcc_results(rule, bc)
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
    cls, model = f["growth_class"], f["best_model"]
    if model == "M0":
        base, alpha = 1.0, 0.0
    elif model == "M1":
        base, alpha = 1.0, float(f["params"]["M1"][1])
    else:
        base, alpha = float(f["base"] or 1.0), float(f["alpha_M2"] or 0.0)

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
            fe = fit_pure_exponential([n for n in Ns if n % 2 == 0],
                                      [y for n, y in zip(Ns, ys) if n % 2 == 0])
            if fe.get("ok"):
                base, source, parity = float(fe["base"]), "parity fit", True
    if an is not None:
        base, alpha, exact, source = an[0], an[1], True, an[2]
        lo = hi = None

    return {"cls": cls, "base": float(base), "alpha": float(alpha),
            "exact": bool(exact), "source": source, "parity_split": parity,
            "base_lo": lo, "base_hi": hi, "n_points": f["n_points"],
            "N_range": f["N_range"], "named": name_base(float(base))}


def rule_point(rule: int, bc: str) -> Optional[Dict]:
    s = load_series(rule, bc)
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
        "product_ab": a["base"] * b["base"],
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
    prod = a["base"] * b["base"]
    slack = tol
    for d, other in ((a, b), (b, a)):
        if not d["exact"] and d.get("base_lo") and d.get("base_hi"):
            slack = max(slack, abs(d["base_hi"] - d["base_lo"]) * other["base"])
    raw_below = bool(prod < 2.0 - 1e-9)
    within = bool(prod >= 2.0 - slack - 1e-9)
    if not raw_below:
        verdict = "on_curve" if abs(prod - 2.0) < 1e-6 else "above"
    elif slack > SLACK_VACUOUS:
        verdict = "inconclusive"
    elif within:
        verdict = "below_within_uncertainty"
    else:
        verdict = "VIOLATION"
    b_lo = 2.0 / a["base"] if a["base"] > 0 else None
    return {"rule": pt["rule"], "bc": pt["bc"], "product": prod,
            "margin": prod - 2.0, "slack": slack,
            "raw_below": raw_below, "within_slack": within,
            "verdict": verdict,
            "ok": verdict != "VIOLATION",
            "exact_a": a["exact"], "exact_b": b["exact"],
            "a": a["base"], "b": b["base"],
            "b_lower_bound": b_lo,
            "b_fit_hi": b.get("base_hi"),
            "on_curve": bool(abs(prod - 2.0) < 1e-6)}


def attractor_deficit(rule: int, bc: str) -> Optional[Dict]:
    """
    V4: the monitored-attractor map has NO hyperbola constraint.  Report the
    deficit 2 - a_att*b_att as an observable measuring transient dominance.
    Assert nothing.
    """
    from .growth_map import _dissipative_descriptor, _series_descriptor
    from .dissipative import load_series as load_att_series
    t = rules.wolfram_to_tuple(rule)
    try:
        if rules.is_unitary(t):
            s = load_att_series(rule, bc)
            if len(s["N"]) < 3:
                return None
            a = _series_descriptor(s["N"], s["n_recurrent"])
            b = _series_descriptor(s["N"], s["d_max"])
        else:
            d = _dissipative_descriptor(rule, bc)
            a, b = d["n_recurrent"], d["d_max"]
    except Exception:
        return None
    if not a or not b or a.get("base") is None or b.get("base") is None:
        return None
    prod = a["base"] * b["base"]
    return {"rule": rule, "bc": bc, "a_att": a["base"], "b_att": b["base"],
            "product": prod, "deficit": 2.0 - prod}


def build(bc: str = "obc0") -> Dict:
    pts, checks, deficits = [], [], []
    for rule in range(256):
        p = rule_point(rule, bc)
        if p is None:
            continue
        pts.append(p)
        checks.append(hyperbola_check(p))
        d = attractor_deficit(rule, bc)
        if d:
            deficits.append(d)
    tally: Dict[str, int] = {}
    for c in checks:
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    out = {"bc": bc, "n_rules": len(pts), "points": pts,
           "hyperbola": checks, "attractor_deficits": deficits,
           "verdicts": tally,
           # every rule whose product is below 2, whatever the slack says --
           # the task forbids letting these pass silently
           "raw_below": [c for c in checks if c["raw_below"]],
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


if __name__ == "__main__":
    main()
