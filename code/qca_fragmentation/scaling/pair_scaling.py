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
from .fits import find_integer_recurrence, fit_pure_exponential

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


def _base_of(Ns, ys) -> Dict:
    """Exact-recurrence base if one exists, else the two-parameter exp base."""
    pairs = [(n, y) for n, y in zip(Ns, ys) if y is not None]
    if len(pairs) < 3:
        return {"base": None, "kind": "n/a", "name": None}
    ns = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rec = find_integer_recurrence(ys)
    if rec.get("ok") and rec["base"] is not None:
        return {"base": rec["base"], "kind": "exact", "name": rec.get("name")}
    fit = fit_pure_exponential(ns, ys)
    if fit.get("ok") and fit.get("bounded"):
        return {"base": float(np.exp(fit["kappa"])), "kind": "fit", "name": None}
    return {"base": None, "kind": "irregular", "name": None}


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


def parity_resolution(rows: List[Dict]) -> Dict:
    """Count coherent rules (pair_offdiag > 0) per N, split by N parity."""
    per_N = collections.defaultdict(int)
    for row in rows:
        for n, is_coh in row["coherent_by_N"].items():
            if is_coh:
                per_N[n] += 1
    even = {n: c for n, c in sorted(per_N.items()) if n % 2 == 0}
    odd = {n: c for n, c in sorted(per_N.items()) if n % 2 == 1}
    return {"per_N": {str(n): c for n, c in sorted(per_N.items())},
            "even_N": {str(n): c for n, c in even.items()},
            "odd_N": {str(n): c for n, c in odd.items()}}


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

    doc = {
        "bc": args.bc,
        "n_rules": len(rows),
        "certificate_coverage": certificate_coverage(args.bc),
        "parity_resolution": parity_resolution(rows),
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
    print(f"  coherent-by-N (offdiag>0): {doc['parity_resolution']['per_N']}")
    print(f"    even N: {doc['parity_resolution']['even_N']}")
    print(f"    odd  N: {doc['parity_resolution']['odd_N']}")
    print(f"  weak charge grades coherence: {doc['n_grading_weak_charge']} rules")


if __name__ == "__main__":
    main()
