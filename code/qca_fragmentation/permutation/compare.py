"""
R11: the same 256 symbol tables under two different gates.

The Hadamard family (R1--R9) and the permutation family (R10) share their rule
encoding exactly -- the same tuple (r00, r01, r10, r11), the same brick-wall
order, the same boundary handling, the same compiled step list.  Only the letter
V is read differently.  So one may ask, rule by rule, what the gate change does
to the sector structure.

METHOD, and the one thing that must not be got wrong.  R9 fits its bases on
N <= 16 and R10 on N <= 24.  A base fitted on a longer window is a different
number, so every comparison here rebuilds the X-gate descriptors on the SAME
window as the Hadamard ones (COMMON_N_CAP = 16) using the identical
scaling/sectors.series_descriptor machinery.  Nothing is compared across
windows; where the N=24 numbers are quoted they are labelled as such.

THREE COMPARISONS, in decreasing order of how much they are entitled to say:

  1. The unitary/reversible baseline, 16 rules.  Here both families satisfy
     sectors = attractors (A2 on the Hadamard side, F1 on the X side), so the
     comparison is between two objects of the same type and is completely fair.
  2. The survivors.  R9's headline sets -- the 8 open-system-fragmented rules,
     the 6 with strictly linear sector counts, the pinned-frontier family -- are
     asked what they do under X.
  3. The rule-level correlation across all 256.  This is the weakest of the
     three: it is a scatter of two fitted exponents, and its value is mostly in
     showing how little survives.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .. import results_io
from ..core import rules
from ..scaling import sectors
from . import analysis

COMMON_N_CAP = 16
CMP_PATH = os.path.join(analysis.ANALYTICS, "gate_compare_{bc}.json")

#: R9 sec.6.3: the V+reset rules that keep an exponentially growing sector count
#: even with a reset in the table.  Fragmentation as an open quantum system.
R9_OPEN_FRAGMENTED = (156, 198, 140, 206, 196, 220, 108, 201)

#: R9 sec.6.4: strictly linear sector counts -- not one big sector, but N+1 of
#: them.  These are the pinned-frontier rules.
R9_LINEAR_COUNT = (44, 100, 110, 124, 188, 230)

#: R9 sec.6.5: the frontier family, whose charge is the position of the extremal
#: excitation and which collapses on the ring.
R9_FRONTIER = sectors.FRONTIER_RULES


def has_V(rule: int) -> bool:
    return "V" in rules.wolfram_to_tuple(rule)


# --- the structural relation between the two families ------------------------

def refinement_check(N: int, bc: str = "obc0") -> Dict:
    """
    The theorem this whole report rests on, verified exhaustively at one size.

    Applying the Hadamard to a site produces BOTH outcomes for the target bit,
    so the flip is always one of them: succ_X(x) is an element of succ_H(x) for
    every state and every rule.  The X graph is therefore a subgraph of the H
    graph on the same vertex set, and consequently the X sectors REFINE the H
    sectors -- each X component sits inside a single H component, each H sector
    is a disjoint union of X sectors, and pointwise

        n_wcc^X(N) >= n_wcc^H(N),      D_max^X(N) <= D_max^H(N).

    Nothing about locality or the brick-wall order enters, so it holds for both
    boundary conventions and every N.
    """
    from ..graph import wcc as _wcc
    from . import xca as _xca

    def _find(parent, a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bad_edge, bad_refine = [], []
    for rule in range(256):
        succ = _wcc.make_succ(rule, N, bc)
        T = _xca.step_table(rule, N, bc)
        hp = list(range(1 << N))
        for x in range(1 << N):
            ys = succ(x)
            if T[x] not in ys:
                bad_edge.append(rule)
                break
            for y in ys:
                ra, rb = _find(hp, x), _find(hp, y)
                if ra != rb:
                    hp[ra] = rb
        xp = list(range(1 << N))
        for x in range(1 << N):
            ra, rb = _find(xp, x), _find(xp, int(T[x]))
            if ra != rb:
                xp[ra] = rb
        seen: Dict[int, int] = {}
        for x in range(1 << N):
            k = _find(xp, x)
            h = _find(hp, x)
            if seen.setdefault(k, h) != h:
                bad_refine.append(rule)
                break
    return {"N": N, "bc": bc, "rules": 256,
            "edge_failures": sorted(set(bad_edge)),
            "refine_failures": sorted(set(bad_refine)),
            "edges_ok": 256 - len(set(bad_edge)),
            "refines_ok": 256 - len(set(bad_refine))}


def pointwise_order(bc: str = "obc0", n_cap: int = COMMON_N_CAP) -> Dict:
    """The two inequalities implied by refinement, over the whole stored sweep."""
    bad_n, bad_d, units = [], [], 0
    for rule in range(256):
        h = sectors.load_series(rule, bc, n_cap)
        x = analysis.load_series(rule, bc, n_cap)
        for N in sorted(set(h["N"]) & set(x["N"])):
            i, j = h["N"].index(N), x["N"].index(N)
            units += 1
            if x["n_wcc"][j] < h["n_wcc"][i]:
                bad_n.append((rule, N))
            if x["d_max_wcc"][j] > h["d_max_wcc"][i]:
                bad_d.append((rule, N))
    return {"units": units, "n_wcc_failures": bad_n, "d_max_failures": bad_d}


def vfree_identity(bc: str = "obc0") -> Dict:
    """
    A rule with no V never invokes the gate, so the two engines must agree
    exactly on those 81 rules -- a cross-implementation test, not a physics
    claim.  Compares the full sector-size multiset at every shared N.
    """
    same = diff = 0
    bad = []
    for rule in range(256):
        if has_V(rule):
            continue
        hs = results_io.load_wcc_results(rule, bc)
        xs = results_io.load_xgate_results(rule, bc)
        for N in sorted(set(hs) & set(xs)):
            if hs[N].get("aborted"):
                continue
            a = sorted(results_io.sizes_from_wcc_record(hs[N]))
            b = sorted(results_io.sizes_from_xgate_record(xs[N], "wcc"))
            if a == b:
                same += 1
            else:
                diff += 1
                bad.append((rule, N))
    return {"rules": 256 - sum(1 for r in range(256) if has_V(r)),
            "units_same": same, "units_diff": diff, "failures": bad[:10]}


# --- Hilbert-space fragmentation, in the standard vocabulary -----------------

#: bases within this of an integer are treated as that integer
_BAND = 0.02


def hsf_phase(a: Optional[float], b: Optional[float]) -> str:
    """
    The HSF phase of a rule, read off the (a, b) point.

    With n_wcc ~ a^N sectors and D_max ~ b^N, the standard vocabulary is:

      unfragmented   a = 1, b = 2      one Krylov sector of full weight
      weak           a > 1, b = 2      exponentially many, largest still 2^N
                                       up to a sub-exponential factor
      strong         a > 1, 1 < b < 2  largest sector an exponentially
                                       vanishing fraction of the space
      shattered      b = 1             every sector sub-exponential

    'degenerate' is not a phase: it is the fit-degenerate corner a = 1 with
    b < 2, which the sum rule forbids asymptotically (a*b >= 2) and which
    therefore always means a sub-exponential sector count that the base cannot
    express (R9 sec.5).
    """
    if a is None or b is None:
        return "irregular"
    if b <= 1.0 + _BAND:
        return "shattered"
    if a <= 1.0 + _BAND:
        return "unfragmented" if b >= 2.0 - _BAND else "degenerate"
    return "weak" if b >= 2.0 - _BAND else "strong"


def phase_table(rows: List[Dict]) -> Dict:
    """Phase under each gate, and the cross-tabulation."""
    ph = [(r["rule"], hsf_phase(r["a_h"], r["b_h"]),
           hsf_phase(r["a_x"], r["b_x"])) for r in rows]
    cross: Dict[str, Dict[str, int]] = {}
    for _, h, x in ph:
        cross.setdefault(h, {})
        cross[h][x] = cross[h].get(x, 0) + 1
    return {"per_rule": [{"rule": r, "h": h, "x": x} for r, h, x in ph],
            "H": dict(Counter(h for _, h, _ in ph)),
            "X": dict(Counter(x for _, _, x in ph)),
            "cross": cross}


def sector_orbit_split(rule: int, N: int, bc: str = "obc0") -> Dict:
    """
    How each Hadamard sector of `rule` breaks into X-gate orbits.

    This is the refinement theorem made concrete, and for rules 156 and 198 it
    has a striking form: every Hadamard sector splits into orbits that are ALL
    THE SAME LENGTH, so

        dim(H sector) = (number of X orbits) * (orbit length),

    i.e. the X dynamics acts on the sector as a free Z_L action and the orbit
    label is an extra conserved charge carried on top of whatever labels the
    Hadamard sector.  Not universal -- 108, 201, 54, 150, 105 all have sectors
    with orbits of differing length -- so it is reported, not assumed.
    """
    import numpy as np
    from ..graph import wcc as _wcc
    from . import xca as _xca

    succ = _wcc.make_succ(rule, N, bc)
    par = list(range(1 << N))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for x in range(1 << N):
        for y in succ(x):
            ra, rb = find(x), find(y)
            if ra != rb:
                par[ra] = rb
    lab = [find(x) for x in range(1 << N)]

    T = _xca.step_table_np(rule, N, bc)
    seen = np.zeros(1 << N, np.int8)
    orbits = []
    for x in range(1 << N):
        if seen[x]:
            continue
        path = []
        y = x
        while seen[y] == 0:
            seen[y] = 1
            path.append(y)
            y = int(T[y])
        if seen[y] == 1:
            orbits.append(path[path.index(y):])
        for z in path:
            seen[z] = 2

    dim = Counter(lab)
    by: Dict[int, List[int]] = {}
    for o in orbits:
        by.setdefault(lab[o[0]], []).append(len(o))
    rows = Counter()
    unequal = 0
    for k, lens in by.items():
        if len(set(lens)) > 1:
            unequal += 1
            continue
        rows[(dim[k], len(lens), lens[0])] += 1
    return {"rule": rule, "N": N, "bc": bc,
            "n_h_sectors": len(dim), "n_x_orbits": len(orbits),
            "unequal_sectors": unequal,
            "factorises": unequal == 0,
            "rows": sorted((d, n, L, c) for (d, n, L), c in rows.items()),
            "orbit_counts": sorted({n for _, n, _ in rows}),
            "orbit_lengths": sorted({L for _, _, L in rows})}


def class_table(rows: List[Dict], key_h: str, key_x: str) -> Dict[str, Dict[str, int]]:
    """Cross-tabulation of growth CLASS under the two gates."""
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        out.setdefault(r[key_h], {})
        out[r[key_h]][r[key_x]] = out[r[key_h]].get(r[key_x], 0) + 1
    return out


def x_point(rule: int, bc: str, n_cap: int = COMMON_N_CAP) -> Optional[Dict]:
    """An X-gate descriptor built on the Hadamard window."""
    return analysis.rule_point(rule, bc, n_cap)


def h_point(rule: int, bc: str, d9: Dict) -> Optional[Dict]:
    for p in d9["points"]:
        if p["rule"] == rule:
            return p
    return None


def build(bc: str = "obc0", n_cap: int = COMMON_N_CAP) -> Dict:
    d9 = sectors.load(bc) or sectors.build(bc)
    rows = []
    for rule in range(256):
        h, x = h_point(rule, bc, d9), x_point(rule, bc, n_cap)
        if h is None or x is None:
            continue
        t = rules.wolfram_to_tuple(rule)
        rows.append({
            "rule": rule, "tuple": "".join(t),
            "unitary": bool(rules.is_unitary(t)),
            "family": x["family"],
            # Hadamard
            "a_h": h["n_wcc"]["base"], "b_h": h["d_max_wcc"]["base"],
            "cls_a_h": h["n_wcc"]["cls"], "cls_b_h": h["d_max_wcc"]["cls"],
            "named_a_h": h["n_wcc"]["named"], "named_b_h": h["d_max_wcc"]["named"],
            # X gate, same window
            "a_x": x["n_wcc"]["base"], "b_x": x["d_max_wcc"]["base"],
            "cls_a_x": x["n_wcc"]["cls"], "cls_b_x": x["d_max_wcc"]["cls"],
            "named_a_x": x["n_wcc"]["named"], "named_b_x": x["d_max_wcc"]["named"],
            "b_rec_x": x["d_max_recurrent"]["base"],
            "cls_b_rec_x": x["d_max_recurrent"]["cls"],
        })
    out = {"bc": bc, "n_cap": n_cap, "rows": rows,
           "n_rules": len(rows),
           "correlation": correlation(rows),
           "series": series_compare(bc, n_cap),
           "refinement": [refinement_check(N, bc) for N in (8, 9)],
           "pointwise": pointwise_order(bc, n_cap),
           "vfree_identity": vfree_identity(bc),
           "class_sector_count": class_table(rows, "cls_a_h", "cls_a_x"),
           "class_sector_size": class_table(rows, "cls_b_h", "cls_b_x"),
           "phases": phase_table(rows),
           "split_156": sector_orbit_split(156, 14, bc),
           "split_198": sector_orbit_split(198, 14, bc)}
    os.makedirs(analysis.ANALYTICS, exist_ok=True)
    with open(CMP_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = CMP_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def correlation(rows: List[Dict]) -> Dict:
    """Pearson r between the two gates, on each coordinate, all rules and the
    unitary subset separately."""
    out = {}
    for tag, sub in (("all", rows),
                     ("unitary", [r for r in rows if r["unitary"]]),
                     ("irreversible", [r for r in rows if not r["unitary"]]),
                     # the 81 V-free rules are the SAME circuit under both
                     # gates, so including them inflates every correlation;
                     # this is the row that means something
                     ("has_V", [r for r in rows if has_V(r["rule"])])):
        for co, kh, kx in (("a", "a_h", "a_x"), ("b", "b_h", "b_x")):
            v = [(r[kh], r[kx]) for r in sub
                 if r[kh] is not None and r[kx] is not None]
            if len(v) < 3:
                continue
            h = np.array([p[0] for p in v])
            x = np.array([p[1] for p in v])
            r_ = (float(np.corrcoef(h, x)[0, 1])
                  if h.std() > 0 and x.std() > 0 else None)
            out[f"{tag}_{co}"] = {"n": len(v), "pearson": r_,
                                  "mean_abs_diff": float(np.mean(np.abs(h - x))),
                                  "identical": int(np.sum(np.abs(h - x) < 1e-6))}
    return out


def series_compare(bc: str, n_cap: int = COMMON_N_CAP) -> List[Dict]:
    """
    The comparison that needs no fitting at all: the sector COUNT at the largest
    common N, and whether the two families even agree on the order of magnitude.
    """
    out = []
    for rule in range(256):
        hs = sectors.load_series(rule, bc, n_cap)
        xs = analysis.load_series(rule, bc, n_cap)
        if not hs["N"] or not xs["N"]:
            continue
        N = min(hs["N"][-1], xs["N"][-1])
        if N not in hs["N"] or N not in xs["N"]:
            continue
        i, j = hs["N"].index(N), xs["N"].index(N)
        out.append({"rule": rule, "N": N,
                    "n_wcc_h": hs["n_wcc"][i], "n_wcc_x": xs["n_wcc"][j],
                    "d_max_h": hs["d_max_wcc"][i], "d_max_x": xs["d_max_wcc"][j],
                    "same_count": hs["n_wcc"][i] == xs["n_wcc"][j],
                    "same_dmax": hs["d_max_wcc"][i] == xs["d_max_wcc"][j]})
    return out


def baseline(d: Dict) -> List[Dict]:
    """The 16 rules that are unitary under H and reversible under X."""
    return [r for r in d["rows"] if r["unitary"]]


def survivors(d: Dict) -> Dict[str, List[Dict]]:
    """R9's headline sets, asked what they do under X."""
    by = {r["rule"]: r for r in d["rows"]}
    ser = {s["rule"]: s for s in d["series"]}
    out = {}
    for name, group in (("open_fragmented", R9_OPEN_FRAGMENTED),
                        ("linear_count", R9_LINEAR_COUNT),
                        ("frontier", R9_FRONTIER)):
        rowset = []
        for rule in group:
            r = by.get(rule)
            if r is None:
                continue
            s = ser.get(rule, {})
            rowset.append({**r, "N": s.get("N"),
                           "n_wcc_h": s.get("n_wcc_h"),
                           "n_wcc_x": s.get("n_wcc_x")})
        out[name] = rowset
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="R11 gate comparison")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    a = ap.parse_args(argv)
    d = build(a.bc)
    print(f"{d['n_rules']} rules with a descriptor under BOTH gates "
          f"(common window N <= {d['n_cap']}, {a.bc})")
    print("\n== correlation between the gates ==")
    for k, v in sorted(d["correlation"].items()):
        pr = "n/a" if v["pearson"] is None else f"{v['pearson']:+.3f}"
        print(f"  {k:<18} n={v['n']:<4} pearson {pr}  "
              f"mean|diff| {v['mean_abs_diff']:.3f}  "
              f"identical {v['identical']}")
    print("\n== structure ==")
    for rc in d["refinement"]:
        print(f"  N={rc['N']}: X edges are H edges for {rc['edges_ok']}/256; "
              f"X sectors refine H sectors for {rc['refines_ok']}/256")
    pw = d["pointwise"]
    print(f"  pointwise over {pw['units']} units: n_wcc^X >= n_wcc^H fails "
          f"{len(pw['n_wcc_failures'])}x; D_max^X <= D_max^H fails "
          f"{len(pw['d_max_failures'])}x")
    vf = d["vfree_identity"]
    print(f"  V-free cross-implementation: {vf['units_same']} units identical, "
          f"{vf['units_diff']} different")
    print("\n== HSF phase ==")
    ph = d["phases"]
    print("  H:", ph["H"])
    print("  X:", ph["X"])
    for h, row in sorted(ph["cross"].items()):
        print(f"  {h:<13} -> " + "  ".join(f"{k}={v}" for k, v in sorted(row.items())))
    sp = d["split_156"]
    print(f"\n== rule 156 at N={sp['N']}: {sp['n_h_sectors']} H sectors split "
          f"into {sp['n_x_orbits']} X orbits ==")
    print(f"  every sector factorises (equal orbit lengths): {sp['factorises']}")
    print(f"  orbit counts seen {sp['orbit_counts']}")
    print(f"  orbit lengths seen {sp['orbit_lengths']}")
    print("\n== growth class of the sector COUNT: H (rows) vs X (cols) ==")
    for h, row in sorted(d["class_sector_count"].items()):
        print(f"  {h:<13}" + "  ".join(f"{k}={v}" for k, v in sorted(row.items())))
    same_c = sum(1 for s in d["series"] if s["same_count"])
    same_d = sum(1 for s in d["series"] if s["same_dmax"])
    print(f"\n== fit-free ==\n  same sector count at the common N: "
          f"{same_c}/{len(d['series'])};  same D_max: {same_d}")

    print("\n== unitary / reversible baseline ==")
    for r in baseline(d):
        print(f"  {r['rule']:<4}{r['tuple']}  H:(a={r['a_h']:.4f},"
              f"b={r['b_h']:.4f})  X:(a={r['a_x']:.4f},b={r['b_x']:.4f})  "
              f"cycle b={r['b_rec_x']:.4f}")

    sv = survivors(d)
    for name, rowset in sv.items():
        print(f"\n== R9 {name} under X ==")
        for r in rowset:
            print(f"  {r['rule']:<4}{r['tuple']}  H count {r['cls_a_h']:<12}"
                  f"({r['n_wcc_h']})  ->  X count {r['cls_a_x']:<12}"
                  f"({r['n_wcc_x']})   X cycle {r['cls_b_rec_x']}")


if __name__ == "__main__":
    main()
