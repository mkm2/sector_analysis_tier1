"""
Per-rule descriptors and maps for the X-gate (permutation) circuits.

Two maps, and the point of the family is how they are related.  Because the
transition graph is functional, every weak component holds exactly one cycle, so

    n_recurrent = n_wcc   identically, for all 256 rules and every N.

The sector map and the monitored-attractor map therefore share their x-axis
exactly, and differ ONLY in what is plotted upward: the largest SECTOR against
the longest CYCLE.  In the Hadamard family those two axes disagreed in both
coordinates and the disagreement was the story (R9 sec.5); here the horizontal
disagreement is identically zero by a theorem, and what is left is a clean
measurement of how much of a basin is actually periodic.

Fitting convention is inherited unchanged from scaling/sectors.py -- exact
recurrence, then analytic override, then the two-parameter fit, never the biased
M2 rate (R2 sec.3).  The analytic overrides there are keyed on Hadamard
observables and cannot fire here, which is deliberate: nothing derived for the
Hadamard circuits is silently reused for a different gate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from .. import results_io
from ..core import rules
from ..scaling import sectors

ANALYTICS = os.path.join(results_io.REPO_ROOT, "analytics")
XMAP_PATH = os.path.join(ANALYTICS, "xgate_map_{bc}.json")

UNIFORM_N_CAP = 24

#: series -> the key handed to sectors.series_descriptor.  Prefixed so no
#: Hadamard ANALYTIC override can match.
_KEYS = {"n_wcc": "x_n_wcc", "d_max_wcc": "x_d_max_wcc",
         "n_recurrent": "x_n_recurrent", "d_max_recurrent": "x_d_max_recurrent"}


def load_series(rule: int, bc: str,
                n_cap: Optional[int] = None) -> Dict[str, List]:
    recs = results_io.load_xgate_results(rule, bc)
    if n_cap is not None:
        recs = {N: r for N, r in recs.items() if N <= n_cap}
    keys = ("n_wcc", "d_max_wcc", "n_recurrent", "d_max_recurrent",
            "n_frozen", "n_fixed_points", "transient_fraction",
            "transient_depth", "d_max_ratio", "reversible")
    out: Dict[str, List] = {"N": []}
    for k in keys:
        out[k] = []
    for N in sorted(recs):
        out["N"].append(N)
        for k in keys:
            out[k].append(recs[N].get(k))
    return out


def rule_point(rule: int, bc: str,
               n_cap: Optional[int] = UNIFORM_N_CAP) -> Optional[Dict]:
    s = load_series(rule, bc, n_cap)
    if len(s["N"]) < 3:
        return None
    t = rules.wolfram_to_tuple(rule)
    d: Dict[str, Optional[Dict]] = {}
    for key, tag in _KEYS.items():
        d[key] = sectors.series_descriptor(rule, bc, tag, s["N"], s[key])
    if any(v is None for v in d.values()):
        return None
    i = len(s["N"]) - 1
    rev = bool(s["reversible"][i])
    return {
        "rule": rule, "bc": bc, "tuple": "".join(t),
        "reversible": rev,
        "family": ("reversible" if rev else
                   ("V+reset" if "V" in t else "V-free")),
        "n_wcc": d["n_wcc"], "d_max_wcc": d["d_max_wcc"],
        "n_recurrent": d["n_recurrent"],
        "d_max_recurrent": d["d_max_recurrent"],
        "N_max": s["N"][i],
        "transient_fraction_at_Nmax": s["transient_fraction"][i],
        "transient_depth_at_Nmax": s["transient_depth"][i],
        "d_max_ratio_at_Nmax": s["d_max_ratio"][i],
        "n_fixed_points_at_Nmax": s["n_fixed_points"][i],
        "cyclic_fraction_at_Nmax": 1.0 - s["transient_fraction"][i],
    }


def identity_check(bc: str, n_cap: Optional[int] = UNIFORM_N_CAP) -> Dict:
    """n_recurrent == n_wcc for every unit, and the sector sizes partition."""
    n = ok = bad = 0
    worst = None
    for rule in range(256):
        for N, r in results_io.load_xgate_results(rule, bc).items():
            if n_cap is not None and N > n_cap:
                continue
            n += 1
            if r["n_recurrent"] == r["n_wcc"]:
                ok += 1
            else:
                bad += 1
                worst = (rule, N, r["n_wcc"], r["n_recurrent"])
            full = results_io.sizes_from_xgate_record(r, "wcc")
            assert sum(full) == (1 << N), (rule, N)
    return {"units": n, "identity_holds": ok, "identity_fails": bad,
            "first_failure": worst}


def build(bc: str = "obc0", n_cap: Optional[int] = UNIFORM_N_CAP) -> Dict:
    pts, checks = [], []
    for rule in range(256):
        p = rule_point(rule, bc, n_cap)
        if p is None:
            continue
        pts.append(p)
        a = p["n_wcc"]["base"]
        for tag, key in (("sector", "d_max_wcc"),
                         ("attractor", "d_max_recurrent")):
            b = p[key]["base"]
            if a is None or b is None:
                continue
            checks.append({"rule": rule, "map": tag, "product": a * b,
                           "below_2": bool(a * b < 2 - 1e-9)})
    # the finite-N form, which needs no fitting
    finite = []
    for rule in range(256):
        s = load_series(rule, bc, n_cap)
        if not s["N"]:
            continue
        r = [(N, k * dm / (1 << N))
             for N, k, dm in zip(s["N"], s["n_wcc"], s["d_max_wcc"])]
        wn, wv = min(r, key=lambda t: t[1])
        finite.append({"rule": rule, "min_ratio": wv, "at_N": wn,
                       "holds": wv >= 1.0 - 1e-12})
    out = {"bc": bc, "n_cap": n_cap, "n_rules": len(pts), "points": pts,
           "products": checks,
           "identity": identity_check(bc, n_cap),
           "finite_hyperbola": finite,
           "finite_failures": [f for f in finite if not f["holds"]]}
    os.makedirs(ANALYTICS, exist_ok=True)
    with open(XMAP_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = XMAP_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="X-gate maps")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    a = ap.parse_args(argv)
    d = build(a.bc)
    print(f"{d['n_rules']} rules with an X-gate descriptor ({a.bc})")
    idn = d["identity"]
    print(f"n_recurrent == n_wcc : {idn['identity_holds']}/{idn['units']} units"
          f"  (failures {idn['identity_fails']})")
    ff = d["finite_failures"]
    worst = min(d["finite_hyperbola"], key=lambda f: f["min_ratio"])
    print(f"finite-N n_wcc*Dmax >= 2^N : "
          f"{len(d['finite_hyperbola']) - len(ff)}/{len(d['finite_hyperbola'])}"
          f" hold; tightest W{worst['rule']} at N={worst['at_N']} "
          f"ratio {worst['min_ratio']:.4f}")
    for tag in ("sector", "attractor"):
        sub = [c for c in d["products"] if c["map"] == tag]
        below = sum(1 for c in sub if c["below_2"])
        print(f"  {tag:<9} map: {below}/{len(sub)} rules with ab < 2")


if __name__ == "__main__":
    main()
