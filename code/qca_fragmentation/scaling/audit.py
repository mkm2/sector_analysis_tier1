"""
Is the scaling dataset good enough to fit?  (run before trusting any growth law)

A growth law is only as good as the series under it, and the failure modes are
quiet ones: a hole in the N grid, a series too short for the recurrence search,
a rule that stopped early because it went ergodic.  None of these raise an
error anywhere in the pipeline -- they just silently weaken a fit.  This module
checks all 256 rules x both boundary conditions and reports:

  MISSING    no data at all (expected only for ergodic-flagged units, which are
             excluded from the fits by design -- those are listed separately)
  GAPS       an interior N with no record: the series cannot be fitted honestly
             and NO recurrence check is valid across the hole
  SHORT      fewer points than find_integer_recurrence needs for order k
             (it needs 2k+1 terms, so order 3 needs 7 and order 4 needs 9)
  PARITY     effective length once split by parity, which is what actually
             limits the exact-base search for the ~43 oscillating pbc rules

Exit status is non-zero if any GAPS are found, so this can gate a report build.

    python -m qca_fragmentation.scaling.audit
    python -m qca_fragmentation.scaling.audit --json analytics/scaling_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from .. import results_io
from ..core import rules
from .summary import load_series

# find_integer_recurrence needs 2*order+1 terms; these are the orders we care to
# be able to detect (order 4 covers every recurrence seen so far, with margin).
MIN_FOR_ORDER = {2: 5, 3: 7, 4: 9}


def audit_unit(rule: int, bc: str) -> Dict:
    s = load_series(rule, bc)
    Ns = s["N"]
    t = "".join(rules.wolfram_to_tuple(rule))
    row = {"rule": rule, "tuple": t, "bc": bc,
           "unitary": rules.is_unitary(rules.wolfram_to_tuple(rule)),
           "n_points": len(Ns), "N_min": Ns[0] if Ns else None,
           "N_max": Ns[-1] if Ns else None,
           "ergodic_from": s["ergodic_from"], "gaps": [],
           "n_even": 0, "n_odd": 0, "max_order": 0, "max_order_parity": 0}
    if not Ns:
        return row
    row["gaps"] = [n for n in range(Ns[0], Ns[-1] + 1) if n not in Ns]
    row["n_even"] = sum(1 for n in Ns if n % 2 == 0)
    row["n_odd"] = sum(1 for n in Ns if n % 2 == 1)
    row["max_order"] = max((k for k, m in MIN_FOR_ORDER.items()
                            if len(Ns) >= m), default=0)
    par = min(row["n_even"], row["n_odd"])
    row["max_order_parity"] = max((k for k, m in MIN_FOR_ORDER.items()
                                   if par >= m), default=0)
    return row


def audit(rule_range=range(256), bcs=("pbc", "obc0")) -> List[Dict]:
    return [audit_unit(r, bc) for r in rule_range for bc in bcs]


def report(rows: List[Dict], out=sys.stdout) -> int:
    gaps = [r for r in rows if r["gaps"]]
    missing = [r for r in rows if r["n_points"] == 0]
    miss_erg = [r for r in missing if r["ergodic_from"] is not None]
    miss_bad = [r for r in missing if r["ergodic_from"] is None]
    have = [r for r in rows if r["n_points"]]

    def _p(*a):
        print(*a, file=out)

    for kind, sel in (("dissipative", lambda r: not r["unitary"]),
                      ("unitary", lambda r: r["unitary"])):
        rs = [r for r in have if sel(r)]
        if not rs:
            continue
        lo = min(r["N_min"] for r in rs)
        hi = max(r["N_max"] for r in rs)
        short = [r for r in rs if r["max_order"] < 4]
        pshort = [r for r in rs if r["max_order_parity"] < 2]
        _p(f"{kind}: {len(rs)} units with data, N={lo}..{hi}, "
           f"{min(r['n_points'] for r in rs)}..{max(r['n_points'] for r in rs)} points")
        _p(f"   recurrence search: {len(rs) - len(short)}/{len(rs)} units long "
           f"enough for order 4, {len(pshort)} too short for order 2 per parity")

    _p(f"\nunits with data      : {len(have)}/{len(rows)}")
    _p(f"missing, ergodic     : {len(miss_erg)}  (excluded from fits by design)")
    _p(f"missing, UNEXPLAINED : {len(miss_bad)}")
    for r in miss_bad:
        _p(f"   W{r['rule']} {r['bc']}")
    _p(f"interior gaps        : {len(gaps)}")
    for r in gaps:
        _p(f"   W{r['rule']} ({r['tuple']}) {r['bc']}: missing N={r['gaps']}")
    return 1 if (gaps or miss_bad) else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the rows here")
    args = ap.parse_args(argv)
    rows = audit()
    rc = report(rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"description":
                       "Per-(rule, bc) coverage audit of the Tier-1c scaling "
                       "dataset: N grid, interior gaps, and the largest "
                       "recurrence order the series length can support.",
                       "rows": rows}, f, indent=1)
        print(f"\nwrote {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
