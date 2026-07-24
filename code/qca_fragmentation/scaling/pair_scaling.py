"""
Tier-1d coherent-core SANDWICH scaling (task Sec.4) from the results_tier1d store.

For each coherent rule we have three per-N series that bracket the unmonitored
channel's fixed-point dimension:

    lower  K_M(N)           = # terminal SCCs                (monitored)
    exact  cesaro_within(N) = within-sector dim Fix(Phi)     (N <= 6, dense)
    upper  fix_upper(N)     = |P_rec|(N), diagonal blocks     (channel bound)

Each series is fitted with the Tier-1c machinery (exact integer recurrence first,
then the two-parameter exponential, with the parity/period split active), and a
per-rule verdict is emitted:

    coincide     -- lower, exact and upper bases agree      (graph is tight)
    interval     -- lower < upper bases, exact inside       (a growth window)
    undetermined -- a series has no growth law on the sizes reached

The module also reports the certificate coverage (how many units certify per N)
and resolves the census non-monotonicity by N-parity, using the pair graph as a
proxy for protected coherence at ALL N (including odd N and beyond the dense
ceiling of the Tier-1b census).

Outputs
    figures/pair_sandwich.csv           one row per (rule, bc)
    analytics/pair_scaling.json         coverage + parity resolution
    reports/tex/tab_pair_sandwich.tex   the coherent-core table for R7
"""

from __future__ import annotations

import collections
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .. import results_io
from ..core import rules
from ..quantum.weak_charges import WALL_HADAMARD_CORE
from .dissipative import fit_with_period
from .fits import find_integer_recurrence, find_recurrence_by_period

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")
ANALYTICS_DIR = os.path.join(results_io.REPO_ROOT, "analytics")

# Coherent-core anchors shown explicitly in the R7 table.
TABLE_RULES = [22, 23, 146, 151, 178, 28, 50, 27, 18, 19]


def load_pair_series(rule: int, bc: str) -> Dict:
    recs = results_io.load_pair_results(rule, bc)
    Ns = sorted(recs)
    out = {"N": Ns}
    for key in ("km", "cesaro_rank", "fix_upper", "pair_offdiag",
                "certified", "n_weak", "n_strong", "weak_grades_coherence",
                "bounded_only"):
        out[key] = [recs[n].get(key) for n in Ns]
    return out


# The pair-graph bounds live in the DOUBLED space, so |P_rec| and the
# within-sector dim Fix legitimately grow up to base 4 (not the base-2 ceiling of
# single-space counts).  We therefore read the fit rate directly and accept any
# finite base up to this ceiling, rather than through fit_with_period's
# `bounded` gate (which is tuned for #attractors / D_max <= 2^N).
_DOUBLED_CEILING = 4.3


def _base_of(Ns, ys) -> Dict:
    """Growth base of a bracket series with the Tier-1c machinery: a parity/period
    split (the pair-graph bounds oscillate by N-parity, e.g. rule 22's |P_rec|),
    an exact integer recurrence per residue class if one holds, else the
    two-parameter exponential rate -- accepting doubled-space bases up to 4."""
    pairs = [(n, y) for n, y in zip(Ns, ys) if y is not None]
    if len(pairs) < 3:
        return {"base": None, "kind": "n/a", "name": None, "period": 1}
    ns = [p[0] for p in pairs]
    yy = [p[1] for p in pairs]
    f = fit_with_period(ns, yy)
    period = f["period"]
    rec = (find_recurrence_by_period(ns, yy, period=period) if period > 1
           else find_integer_recurrence(yy))
    if rec.get("ok") and rec.get("base") is not None:
        return {"base": rec["base"], "kind": "exact", "name": rec.get("name"),
                "period": period}
    # raw exponential rate (per residue class if the series splits by parity)
    if period > 1:
        ks = [f[p + "_exp"]["kappa"] for p in ("even", "odd")
              if f[p + "_exp"].get("ok")]
        kappa = max(ks, key=abs) if ks else None
    else:
        fe = f["all_exp"]
        kappa = fe["kappa"] if fe.get("ok") else None
    if kappa is not None:
        base = float(np.exp(kappa))
        if 0 < base <= _DOUBLED_CEILING:
            return {"base": base, "kind": "fit", "name": None, "period": period}
    return {"base": None, "kind": "irregular", "name": None, "period": period}


def summarize_pair_rule(rule: int, bc: str) -> Dict:
    s = load_pair_series(rule, bc)
    t = "".join(rules.wolfram_to_tuple(rule))
    Ns = s["N"]
    row = {
        "rule": rule, "tuple": t, "bc": bc,
        "n_points": len(Ns),
        "N_min": Ns[0] if Ns else None,
        "N_max": Ns[-1] if Ns else None,
        "wall_hadamard": (t[1] == "V" and t[2] == "V"),
    }
    lo = _base_of(Ns, s["km"])
    ex = _base_of(Ns, s["cesaro_rank"])
    up = _base_of(Ns, s["fix_upper"])
    row["km_base"], row["km_base_kind"] = lo["base"], lo["kind"]
    row["cesaro_base"], row["cesaro_base_kind"], row["cesaro_base_name"] = \
        ex["base"], ex["kind"], ex["name"]
    row["fix_upper_base"], row["fix_upper_base_kind"] = up["base"], up["kind"]

    bl, bu = lo["base"], up["base"]
    if bl is None or bu is None:
        verdict = "undetermined"
    elif abs(bl - bu) < 1e-6:
        verdict = "coincide"
    else:
        verdict = "interval"
    row["sandwich"] = verdict

    # certificate + coherence at the largest N reached
    row["certified_Nmax"] = s["certified"][-1] if Ns else None
    row["any_certified"] = any(c is True for c in s["certified"])
    row["n_weak_Nmax"] = next((w for w in reversed(s["n_weak"]) if w is not None),
                              None)
    row["grades_coherence"] = any(g is True for g in s["weak_grades_coherence"])
    # protected-coherence flag per N (pair_offdiag > 0), by parity
    row["coherent_by_N"] = {int(n): (o or 0) > 0
                            for n, o in zip(Ns, s["pair_offdiag"])
                            if o is not None}
    return row


def coherent_rules() -> List[int]:
    from ..pair_sweep import coherent_rules as _cr
    return _cr()


def summarize(bcs=("pbc",)) -> List[Dict]:
    rows = []
    for rule in coherent_rules():
        for bc in bcs:
            if results_io.load_pair_results(rule, bc):
                rows.append(summarize_pair_rule(rule, bc))
    return rows


def certificate_coverage(bc: str = "pbc") -> Dict:
    """How many dissipative units certify exact at each N (all computed rules)."""
    cov = collections.defaultdict(lambda: [0, 0])   # N -> [certified, total]
    pdir = results_io.PAIR_RESULTS_DIR
    files = os.listdir(pdir) if os.path.isdir(pdir) else []
    for f in files:
        if not f.endswith(f"_{bc}.jsonl"):
            continue
        for line in open(os.path.join(pdir, f)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("bounded_only"):
                continue
            c = r.get("certified")
            if c is None:
                continue
            cov[r["N"]][1] += 1
            if c:
                cov[r["N"]][0] += 1
    return {str(n): {"certified": v[0], "total": v[1]}
            for n, v in sorted(cov.items())}


def parity_resolution(bc: str = "pbc") -> Dict:
    """Resolve the coherence-census non-monotonicity by N-parity.

    Two indicators over all 240 dissipative rules:
      * EXACT (N<=6): # rules with within-sector dim Fix > K_M, i.e. genuine
        protected coherence beyond the classical pointer count.  This includes
        the ODD N the even-only Tier-1b census could not reach.
      * SUPPORT (all N): # rules whose off-diagonal P_rec is non-empty -- an
        upper bound; it shows coherence support PERSISTS at every N (it does not
        vanish and reappear), so the even-only census understates the picture.
    """
    exact = collections.defaultdict(int)
    support = collections.defaultdict(int)
    pdir = results_io.PAIR_RESULTS_DIR
    files = os.listdir(pdir) if os.path.isdir(pdir) else []
    for f in files:
        if not f.endswith(f"_{bc}.jsonl"):
            continue
        for line in open(os.path.join(pdir, f)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("bounded_only"):
                continue
            N = r["N"]
            if (r.get("pair_offdiag") or 0) > 0:
                support[N] += 1
            cz, km = r.get("cesaro_rank"), r.get("km")
            if cz is not None and km is not None and cz > km:
                exact[N] += 1
    return {
        "exact_by_N": {str(n): exact[n] for n in sorted(exact)},
        "exact_even": {str(n): exact[n] for n in sorted(exact) if n % 2 == 0},
        "exact_odd": {str(n): exact[n] for n in sorted(exact) if n % 2 == 1},
        "support_by_N": {str(n): support[n] for n in sorted(support)},
    }


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [k for k in rows[0].keys() if k != "coherent_by_N"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(x, nd=4):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def latex_table(rows: List[Dict], path: str, bc: str = "pbc",
                only: Optional[List[int]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    only = TABLE_RULES if only is None else only
    sel = [r for r in rows if r["bc"] == bc and r["rule"] in only]
    sel.sort(key=lambda r: only.index(r["rule"]))
    lines = [
        r"\begin{tabular}{@{}rl rrr l c c@{}}",
        r"\toprule",
        r"W & tuple & $b_{K_M}$ & $b_{\mathrm{Fix}}$ & $b_{|P_{\mathrm{rec}}|}$ "
        r"& verdict & cert & weak \\",
        r"\midrule",
    ]
    for r in sel:
        cz = _fmt(r["cesaro_base"])
        if r["cesaro_base_name"]:
            cz += f" {{\\footnotesize {r['cesaro_base_name']}}}"
        cert = {True: r"\checkmark", False: r"$\times$", None: "--"}[
            r["certified_Nmax"]]
        weak = "yes" if r["grades_coherence"] else "--"
        lines.append(
            f"{r['rule']} & \\texttt{{{r['tuple']}}} & "
            f"{_fmt(r['km_base'])} & {cz} & {_fmt(r['fix_upper_base'])} & "
            f"{r['sandwich']} & {cert} & {weak} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="pbc")
    args = ap.parse_args(argv)
    rows = summarize((args.bc,))
    write_csv(rows, os.path.join(FIGURES_DIR, "pair_sandwich.csv"))
    latex_table(rows, os.path.join(TEX_DIR, "tab_pair_sandwich.tex"), bc=args.bc)

    par = parity_resolution(args.bc)
    doc = {
        "bc": args.bc,
        "n_rules": len(rows),
        "certificate_coverage": certificate_coverage(args.bc),
        "parity_resolution": par,
        "sandwich_verdicts": dict(collections.Counter(r["sandwich"] for r in rows)),
        "n_grading_weak_charge": sum(1 for r in rows if r["grades_coherence"]),
        "tier1d_version": results_io.TIER1D_VERSION,
    }
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    with open(os.path.join(ANALYTICS_DIR, "pair_scaling.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)

    print(f"{args.bc}: {len(rows)} coherent rules")
    print(f"  sandwich verdicts: {doc['sandwich_verdicts']}")
    print(f"  certificate coverage: {doc['certificate_coverage']}")
    print(f"  exact within-sector coherent count (dimFix>K_M): {par['exact_by_N']}")
    print(f"    even N: {par['exact_even']}   odd N: {par['exact_odd']}")
    print(f"  coherence-support persists (offdiag>0): {par['support_by_N']}")
    print(f"  weak charge grades coherence: {doc['n_grading_weak_charge']} rules")


if __name__ == "__main__":
    main()
