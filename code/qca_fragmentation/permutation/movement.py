"""
Two movements in the X-gate rule space, and the census that explains them.

MOVEMENT 1, WCC -> recurrent.  The same rule seen in the two maps.  Because the
graph is functional the horizontal coordinate is *identical* in both (F1), so
the move is a pure vertical drop from the base of the largest SECTOR to the base
of the longest CYCLE.  In the Hadamard family (R9 F3) the move had both
components and the horizontal part was the interesting one; here it is zero by
theorem and the vertical part is the entire content.

MOVEMENT 2, parent -> children.  Every rule has a coherent parent: switch the
resets off, D, E -> I (rules.coherent_part).  The parent is one of the 16
reversible rules, and the 240 others are its descendants -- 3^k - 1 of them for
a parent with k identity slots, so the identity 204 alone accounts for the whole
V-free family of 80.  This is R9's F6 clustering applied to the permutation
circuits, where it can be pushed further than R9 could: here the parent's
recurrent set is a permutation of the whole space, and one can ask whether the
child's attractors are literally orbits OF THE PARENT.  For 179 of the 240 they
are (INHERITANCE below).

The two movements are not independent.  A cycle sits inside a sector, so
D_rec <= D_wcc pointwise and movement 1 always points down; and the only source
of a long cycle is a parent that has one, so movement 2 controls what movement 1
can leave behind.
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
from . import analysis, xca

INHERIT_PATH = os.path.join(analysis.ANALYTICS, "xgate_inheritance_{bc}.json")

#: N at which the inheritance question is decided exhaustively.  Two sizes, one
#: of each parity, because the brick-wall sublattices are only independent sets
#: at even N.
INHERIT_N = (12, 13)


def children(parent: int) -> List[int]:
    """
    Every rule whose coherent part is `parent`, the rule itself excluded.

    Deliberately NOT rules.dissipative_children, which additionally demands a V
    (it was written for the Hadamard family, where a rule without a V has no
    coherent dynamics worth the name).  Here the V-free rules are ordinary
    members of the identity's cluster and dropping them would lose 80 of 240.
    """
    return [r for r in range(256)
            if r != parent and rules.coherent_parent(r) == parent]


# --- movement 1: WCC -> recurrent --------------------------------------------

def movement(d: Dict) -> List[Dict]:
    """Per rule: the vertical drop from the sector map to the cycle map."""
    out = []
    for p in d["points"]:
        bw, br = p["d_max_wcc"]["base"], p["d_max_recurrent"]["base"]
        out.append({
            "rule": p["rule"], "tuple": p["tuple"], "family": p["family"],
            "a": p["n_wcc"]["base"],
            "b_wcc": bw, "b_rec": br,
            "cls_wcc": p["d_max_wcc"]["cls"], "cls_rec": p["d_max_recurrent"]["cls"],
            "drop": (None if bw is None or br is None else bw - br),
            "stays": (bw is not None and br is not None and abs(bw - br) < 1e-9),
            "cyclic_fraction": p["cyclic_fraction_at_Nmax"],
        })
    return sorted(out, key=lambda r: (-(r["drop"] if r["drop"] is not None else -1),
                                      r["rule"]))


def cycle_class_census(d: Dict) -> Dict:
    """
    Growth class of the longest cycle, by family.

    This is the table that answers "why is there anything exponential left".
    Sectors partition the basis and the sum rule forces exponential structure
    into one of the two sector coordinates; cycles carry no such obligation,
    because the recurrent set is not the whole space and is usually a vanishing
    fraction of it.  So the cycle map is free to collapse, and mostly does.
    """
    cls = ("constant", "polynomial", "exponential", "irregular")
    out: Dict[str, Dict[str, int]] = {}
    for p in d["points"]:
        row = out.setdefault(p["family"], {c: 0 for c in cls})
        row[p["d_max_recurrent"]["cls"]] = \
            row.get(p["d_max_recurrent"]["cls"], 0) + 1
    for f, row in out.items():
        row["rules"] = sum(row[c] for c in cls if c in row)
    return out


def above_curve(d: Dict, plane: str = "cycle", tail: int = 6) -> List[Dict]:
    """
    The rules whose point lies ON or ABOVE $ab=2$ in the requested plane.

    In the sector plane that is unremarkable -- the curve is an exclusion curve
    there, so everything must be on or above it.  In the CYCLE plane it is the
    interesting question: nothing forces a rule up there, since cycles do not
    partition the basis (see the report's "three sizes" section), so a rule with
    a*b_rec >= 2 has an exponentially long attractor AND exponentially many of
    them.  Returns the two sequences as well, since the position is a claim
    about them and the reader should be able to check it.
    """
    key = "d_max_wcc" if plane == "sector" else "d_max_recurrent"
    ykey = "n_wcc" if plane == "sector" else "n_recurrent"
    out = []
    for p in d["points"]:
        a, b = p["n_wcc"]["base"], p[key]["base"]
        if a is None or b is None or a * b < 2.0 - 1e-9:
            continue
        s = analysis.load_series(p["rule"], p["bc"], analysis.UNIFORM_N_CAP)
        out.append({
            "rule": p["rule"], "tuple": p["tuple"], "family": p["family"],
            "a": a, "b": b, "product": a * b,
            "cls": p[key]["cls"],
            "on_curve": bool(abs(a * b - 2.0) < 1e-9),
            "N_tail": s["N"][-tail:],
            "count_tail": s[ykey][-tail:],
            "size_tail": s[key][-tail:],
        })
    return sorted(out, key=lambda r: (-r["product"], r["rule"]))


def above_curve_groups(d: Dict, plane: str = "cycle", tail: int = 6):
    """above_curve() with rules sharing BOTH sequences collapsed into one row."""
    groups: Dict = {}
    for r in above_curve(d, plane, tail):
        k = (tuple(r["count_tail"]), tuple(r["size_tail"]))
        groups.setdefault(k, []).append(r)
    out = []
    for (cnt, size), rs in groups.items():
        out.append({**rs[0], "rules": [x["rule"] for x in rs],
                    "tuples": [x["tuple"] for x in rs],
                    "count_tail": list(cnt), "size_tail": list(size)})
    return sorted(out, key=lambda r: (-r["product"], r["rules"][0]))


def growing_cycles(d: Dict) -> List[Dict]:
    """The rules whose longest cycle grows at all, with the series tail."""
    out = []
    for p in d["points"]:
        c = p["d_max_recurrent"]["cls"]
        if c in ("constant",):
            continue
        s = analysis.load_series(p["rule"], p["bc"], analysis.UNIFORM_N_CAP)
        out.append({"rule": p["rule"], "tuple": p["tuple"],
                    "family": p["family"], "cls": c,
                    "base": p["d_max_recurrent"]["base"],
                    "parent": rules.coherent_parent(p["rule"]),
                    "tail": s["d_max_recurrent"][-5:],
                    "N_tail": s["N"][-5:]})
    order = {"exponential": 0, "irregular": 1, "polynomial": 2}
    return sorted(out, key=lambda r: (order.get(r["cls"], 9), -(r["base"] or 0),
                                      r["rule"]))


# --- the V-free collapse ------------------------------------------------------

#: the 16 rules whose every symbol is a reset: the new bit does not depend on
#: the old one at all, so the update is a boolean function of the two neighbours
RESET_ONLY = [r for r in range(256)
              if all(s in "DE" for s in rules.wolfram_to_tuple(r))]

_BOOL_NAMES = {
    (0, 0, 0, 0): "0", (1, 1, 1, 1): "1",
    (0, 0, 0, 1): r"$l\wedge r$", (1, 1, 1, 0): r"$\neg(l\wedge r)$",
    (0, 1, 1, 1): r"$l\vee r$", (1, 0, 0, 0): r"$\neg(l\vee r)$",
    (0, 1, 0, 1): "$r$", (1, 0, 1, 0): r"$\neg r$",
    (0, 0, 1, 1): "$l$", (1, 1, 0, 0): r"$\neg l$",
    (0, 1, 1, 0): r"$l\oplus r$", (1, 0, 0, 1): r"$\neg(l\oplus r)$",
    (0, 0, 1, 0): r"$l\wedge\neg r$", (1, 1, 0, 1): r"$\neg l\vee r$",
    (0, 1, 0, 0): r"$\neg l\wedge r$", (1, 0, 1, 1): r"$l\vee\neg r$",
}


def _is_affine(tt) -> bool:
    """f(l,r) = c0 + c1 l + c2 r over GF(2), i.e. XOR-like."""
    c0 = tt[0]
    c2 = tt[1] ^ c0
    c1 = tt[2] ^ c0
    return bool(tt[3] == (c0 ^ c1 ^ c2) and (c1 or c2))


def reset_only_table(d: Dict) -> List[Dict]:
    """
    The 16 all-reset rules, classified by the boolean function they implement.

    This is the sharp end of the V-free question.  With every symbol a reset the
    new bit is f(l, r) with no dependence on the old bit, so the rule IS a
    boolean function of two variables.  Of the sixteen such functions, fourteen
    are constant, a (possibly negated) projection, or monotone -- and all of
    those collapse to bounded cycles.  The two that are neither, XOR and XNOR,
    are exactly rules 90 and 165, are GF(2)-linear, and are exactly the two
    V-free rules in the whole family whose cycle length is not bounded.
    """
    pts = {p["rule"]: p for p in d["points"]}
    out = []
    for r in RESET_ONLY:
        t = rules.wolfram_to_tuple(r)
        tt = tuple(1 if s == "E" else 0 for s in t)     # E -> 1, D -> 0
        s = analysis.load_series(r, d["bc"], analysis.UNIFORM_N_CAP)
        out.append({"rule": r, "tuple": "".join(t),
                    "f": _BOOL_NAMES.get(tt, "?"), "affine": _is_affine(tt),
                    "cls": pts[r]["d_max_recurrent"]["cls"],
                    "cyc_tail": s["d_max_recurrent"][-4:],
                    "cyc_max": max(s["d_max_recurrent"])})
    return out


# --- movement 2: parent -> children ------------------------------------------

def parent_clusters(d: Dict, plane: str = "cycle") -> List[Dict]:
    """
    [{parent, parent_xy, children:[(rule, xy)], ...}] for the requested plane.

    plane 'sector' plots (a, b_wcc), plane 'cycle' plots (a, b_rec).  The
    x-coordinate is the same in both -- that is F1 -- so the two panels differ
    only in the height of each point, parent included.
    """
    key = "d_max_wcc" if plane == "sector" else "d_max_recurrent"
    pts = {p["rule"]: p for p in d["points"]}

    def xy(r):
        p = pts.get(r)
        if p is None or p["n_wcc"]["base"] is None or p[key]["base"] is None:
            return None
        return (p["n_wcc"]["base"], p[key]["base"])

    out = []
    for parent in sorted(rules.UNITARY_RULES):
        kids = [(c, xy(c)) for c in children(parent)]
        kids = [(c, v) for c, v in kids if v is not None]
        out.append({"parent": parent,
                    "tuple": "".join(rules.wolfram_to_tuple(parent)),
                    "parent_xy": xy(parent),
                    "children": kids,
                    "n_children": len(children(parent))})
    return sorted(out, key=lambda c: (-c["n_children"], c["parent"]))


def parent_summary(d: Dict, inherit: Optional[Dict] = None) -> List[Dict]:
    """Per parent: what its cluster does in both planes."""
    pts = {p["rule"]: p for p in d["points"]}
    inh = {int(k): v for k, v in (inherit or {}).get("rules", {}).items()}
    out = []
    for parent in sorted(rules.UNITARY_RULES):
        kids = children(parent)
        pp = pts[parent]
        ke = sum(1 for c in kids if pts[c]["d_max_recurrent"]["cls"] == "exponential")
        se = sum(1 for c in kids if pts[c]["n_wcc"]["cls"] == "exponential")
        drops = [pts[c]["d_max_wcc"]["base"] - pts[c]["d_max_recurrent"]["base"]
                 for c in kids
                 if pts[c]["d_max_wcc"]["base"] is not None
                 and pts[c]["d_max_recurrent"]["base"] is not None]
        ninh = sum(1 for c in kids if inh.get(c, {}).get("inherits"))
        out.append({
            "parent": parent, "tuple": pp["tuple"],
            "n_children": len(kids),
            "parent_cls_sector": pp["n_wcc"]["cls"],
            "parent_cls_cycle": pp["d_max_recurrent"]["cls"],
            "parent_b_cycle": pp["d_max_recurrent"]["base"],
            "children_exp_sector_count": se,
            "children_exp_cycle": ke,
            "median_drop": float(np.median(drops)) if drops else None,
            "inherits": ninh, "inherit_of": len(kids) if inh else 0,
        })
    return out


# --- inheritance: are the child's attractors orbits of the parent? -----------

def _cycles(rule: int, N: int, bc: str):
    """(list of cycles as state lists, the step table)."""
    T = xca.step_table_np(rule, N, bc)
    seen = np.zeros(1 << N, np.int8)
    cyc = []
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
            cyc.append(path[path.index(y):])
        for z in path:
            seen[z] = 2
    return cyc, T


def inheritance(N: int, bc: str = "obc0") -> Dict:
    """
    For every non-reversible rule, decide at size N whether

        (i)  the resets act as the IDENTITY on the recurrent set, i.e. the child
             map and the parent map agree there, and hence
        (ii) every attractor of the child is an orbit of its reversible parent.

    (i) implies (ii); we check (i), which is the stronger and cheaper statement,
    and record the child's and the parent's longest cycle either way.  This is
    the question R9 could not ask: there the coherent parent has no single-valued
    successor, so "the child's attractor is an orbit of the parent" is not even
    a well-formed sentence.
    """
    out: Dict[int, Dict] = {}
    plen: Dict[int, int] = {}
    for parent in sorted(rules.UNITARY_RULES):
        pc, Tp = _cycles(parent, N, bc)
        plen[parent] = max(len(c) for c in pc)
        for c in children(parent):
            cyc, T = _cycles(c, N, bc)
            rec = np.array(sorted(x for cc in cyc for x in cc), dtype=np.int64)
            agree = bool(len(rec) and np.array_equal(T[rec], Tp[rec]))
            out[c] = {"parent": parent, "inherits": agree,
                      "child_max_cycle": max(len(x) for x in cyc),
                      "parent_max_cycle": plen[parent],
                      "n_cycles": len(cyc), "n_recurrent": int(len(rec))}
    n = len(out)
    ok = sum(1 for v in out.values() if v["inherits"])
    longer = [r for r, v in out.items()
              if v["child_max_cycle"] > v["parent_max_cycle"]]
    return {"N": N, "bc": bc, "rules": {str(k): v for k, v in out.items()},
            "n_children": n, "n_inherits": ok,
            "n_longer_than_parent": len(longer),
            "longer_than_parent": sorted(longer),
            "parent_max_cycle": {str(k): v for k, v in plen.items()}}


def build_inheritance(bc: str = "obc0", Ns=INHERIT_N) -> Dict:
    out = {"bc": bc, "by_N": {}}
    for N in Ns:
        out["by_N"][str(N)] = inheritance(N, bc)
    # a rule inherits iff it does so at every N tested
    keys = [str(N) for N in Ns]
    merged = {}
    for r in out["by_N"][keys[0]]["rules"]:
        merged[r] = {
            "parent": out["by_N"][keys[0]]["rules"][r]["parent"],
            "inherits": all(out["by_N"][k]["rules"][r]["inherits"]
                            for k in keys),
            "child_max_cycle": {k: out["by_N"][k]["rules"][r]["child_max_cycle"]
                                for k in keys},
            "parent_max_cycle": {k: out["by_N"][k]["rules"][r]["parent_max_cycle"]
                                 for k in keys},
        }
    out["rules"] = merged
    out["n_children"] = len(merged)
    out["n_inherits"] = sum(1 for v in merged.values() if v["inherits"])
    out["n_longer_than_parent"] = sum(
        1 for v in merged.values()
        if any(v["child_max_cycle"][k] > v["parent_max_cycle"][k] for k in keys))
    out["longer_than_parent"] = sorted(
        int(r) for r, v in merged.items()
        if any(v["child_max_cycle"][k] > v["parent_max_cycle"][k] for k in keys))
    os.makedirs(analysis.ANALYTICS, exist_ok=True)
    with open(INHERIT_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load_inheritance(bc: str = "obc0") -> Optional[Dict]:
    p = INHERIT_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R10 movement analyses")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    ap.add_argument("--rebuild-inheritance", action="store_true")
    a = ap.parse_args(argv)
    d = analysis.load(a.bc) or analysis.build(a.bc)

    print("== cycle growth class, by family ==")
    for fam, row in sorted(cycle_class_census(d).items()):
        print(f"  {fam:<11} " + "  ".join(
            f"{k}={row[k]}" for k in
            ("constant", "polynomial", "exponential", "irregular", "rules")))

    mv = movement(d)
    stay = [m for m in mv if m["stays"]]
    dr = [m["drop"] for m in mv if m["drop"] is not None]
    print(f"\n== movement WCC -> recurrent ==\n  {len(dr)} rules with both bases; "
          f"{len(stay)} stay put; median drop {np.median(dr):.4f}; "
          f"max drop {max(dr):.4f} (X{mv[0]['rule']})")
    neg = [m for m in mv if m["drop"] is not None and m["drop"] < -1e-9]
    print(f"  drops that go UP (must be none): {len(neg)}")

    print("\n== all-reset rules: the V-free collapse ==")
    for row in reset_only_table(d):
        print(f"  X{row['rule']:<4}{row['tuple']}  f={row['f']:<22}"
              f"affine={str(row['affine']):<6}{row['cls']:<12}"
              f"max cycle {row['cyc_max']}")

    inh = (build_inheritance(a.bc) if a.rebuild_inheritance
           else (load_inheritance(a.bc) or build_inheritance(a.bc)))
    print(f"\n== inheritance (N={list(inh['by_N'])}) ==")
    print(f"  resets act as the identity on the recurrent set: "
          f"{inh['n_inherits']}/{inh['n_children']} children")
    print(f"  children whose longest cycle exceeds the parent's: "
          f"{inh['n_longer_than_parent']} {inh['longer_than_parent']}")

    print("\n== per parent ==")
    for row in parent_summary(d, inh):
        print(f"  X{row['parent']:<4}{row['tuple']}  kids={row['n_children']:<3}"
              f"cycle={row['parent_cls_cycle']:<12}"
              f"exp-cycle kids={row['children_exp_cycle']:<3}"
              f"exp-sector kids={row['children_exp_sector_count']:<3}"
              f"inherit={row['inherits']}/{row['inherit_of']}")


if __name__ == "__main__":
    main()
