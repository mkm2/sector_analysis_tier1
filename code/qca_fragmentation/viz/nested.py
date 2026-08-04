"""
Nested fragmentation: can a rule fragment twice?

THE QUESTION.  A rule fragments once when the number of sectors (weak
components) grows exponentially with N.  It fragments a second time, INSIDE a
sector, when one weak component contains exponentially many terminal SCCs.  Only
the thirteen rules of R9 sec.7.3 can do the second thing at all --- every other
rule has exactly one attractor per sector.  So: is there a rule that does both?

The answer is no, and the way it fails is a clean dichotomy (R13).

WHAT IS MEASURED, per rule and per N:
  n_wcc      number of sectors
  n_att      number of terminal SCCs in total
  max_att    the most attractors any single sector contains
  big_att    attractors inside the LARGEST sector
  dist       attractors per sector, descending -- the prettiest of the four
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from . import spy

ANALYTICS = os.path.join(results_io.REPO_ROOT, "analytics")
STORE = os.path.join(ANALYTICS, "nested_fragmentation_{bc}.json")

#: R9 sec.7.3's twelve, plus rule 37 which is an exception only at N = 2 (mod 3).
MULTI_RULES = (203, 217, 219, 36, 44, 100, 104, 233, 29, 71, 235, 249, 37)

#: The three exact recurrences the total attractor counts obey.  All are
#: "Narayana-like": a linear recurrence with two unit coefficients and no others.
FAMILIES = {
    (1, 0, 1): ("supergolden $\\psi$", 1.4655712318767682,
                "a_n = a_{n-1} + a_{n-3}"),
    (1, 0, 0, 1): ("$x^4=x^3+1$", 1.3802775690976143,
                   "a_n = a_{n-1} + a_{n-4}"),
    (0, 1, 1): ("plastic $\\rho$", 1.3247179572447454,
                "a_n = a_{n-2} + a_{n-3}"),
}


def census(rule: int, Ns: Sequence[int], bc: str = "obc0") -> Dict:
    """The four series, plus the per-sector distribution at the largest N."""
    nw, natt, mx, big, dist = [], [], [], [], None
    for N in Ns:
        sb, term = spy.scc_blocks(rule, N, bc)
        wb = spy.wcc_blocks(rule, N, bc)          # largest sector first
        own = spy.block_of(wb, 1 << N)
        per = Counter(own[b[0]] for b, t in zip(sb, term) if t)
        nw.append(len(wb))
        natt.append(sum(term))
        mx.append(max(per.values()) if per else 0)
        big.append(per.get(0, 0))
        dist = sorted(per.values(), reverse=True)
    return {"rule": rule, "bc": bc, "N": list(Ns), "n_wcc": nw,
            "n_att": natt, "max_att": mx, "big_att": big, "dist": dist}


def build(Ns=range(6, 17), bc: str = "obc0", rules_in=MULTI_RULES) -> Dict:
    out = {"bc": bc, "N": list(Ns),
           "rows": [census(r, Ns, bc) for r in rules_in]}
    os.makedirs(ANALYTICS, exist_ok=True)
    with open(STORE.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = STORE.format(bc=bc)
    return json.load(open(p)) if os.path.exists(p) else None


# --- classification ----------------------------------------------------------

def growth(Ns: Sequence[int], ys: Sequence[int], rule: int = 0,
           bc: str = "obc0", key: str = "y") -> Dict:
    """
    Growth class of one series, using the project's parity-aware descriptor.

    Several of these series oscillate with N (rule 29's sector count runs
    3, 6, 5, 9, 7, 14, ...), so a naive ratio test misreads them in both
    directions -- which is exactly why the shared descriptor is used here rather
    than a fresh heuristic.
    """
    from ..scaling import sectors as S
    d = S.series_descriptor(rule, bc, key, list(Ns), list(ys))
    if d is None:
        return {"cls": "unknown", "base": None}
    return {"cls": d["cls"], "base": d["base"], "exact": d["exact"],
            "alpha": d["alpha"]}


def classify(row: Dict) -> Dict:
    """(sector growth, in-sector growth) for one rule, and the joint verdict."""
    Ns = row["N"]
    a = growth(Ns, row["n_wcc"], row["rule"], row["bc"], "n_wcc")
    b = growth(Ns, row["max_att"], row["rule"], row["bc"], "max_att")
    t = growth(Ns, row["n_att"], row["rule"], row["bc"], "n_att")
    both = a["cls"] == "exponential" and b["cls"] == "exponential"
    return {"rule": row["rule"], "sectors": a, "in_sector": b, "total": t,
            "doubly_exponential_fragmentation": both}


def exclusion_check(bc: str = "obc0", data: Optional[Dict] = None) -> Dict:
    """
    THE RESULT.  No rule fragments exponentially in both directions.

    Rules outside MULTI_RULES have one attractor per sector, so their in-sector
    count is identically 1 and cannot grow; the check therefore only has to
    inspect the thirteen, which it does explicitly.
    """
    d = data or load(bc) or build(bc=bc)
    rows = [classify(r) for r in d["rows"]]
    bad = [r["rule"] for r in rows if r["doubly_exponential_fragmentation"]]
    exp_sectors = [r["rule"] for r in rows if r["sectors"]["cls"] == "exponential"]
    exp_inside = [r["rule"] for r in rows
                  if r["in_sector"]["cls"] == "exponential"]
    return {"bc": bc, "rows": rows, "violations": bad,
            "exponential_sectors": exp_sectors,
            "exponential_in_sector": exp_inside,
            "disjoint": not (set(exp_sectors) & set(exp_inside))}


def narayana(n: int) -> List[int]:
    """1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, ... -- a(k) = a(k-1) + a(k-3)."""
    a = [1, 1, 1]
    while len(a) < n:
        a.append(a[-1] + a[-3])
    return a[:n]
