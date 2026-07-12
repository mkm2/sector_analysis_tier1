"""
Aggregate Tier-1a sweep results into per-rule scaling summaries (Tier 1c).

Reads results/{rule}_{bc}.jsonl, builds the series y(N) for
  n_recurrent  (number of sectors),
  d_max        (largest sector = max_recurrent),
  max_basin    (== d_max for unitary rules),
fits each with scaling.fits.fit_series (BIC model selection), and emits a tidy
summary (list of dict rows) plus a CSV.  Ergodic-flagged units are excluded from
the fit; the smallest N at which the rule first went ergodic is recorded.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

from .. import results_io
from ..core import rules
from .fits import fit_series


def load_series(rule: int, bc: str):
    """Return dict with N grid and y-series (ergodic units excluded)."""
    recs = results_io.load_results(rule, bc)
    Ns, n_rec, d_max, max_basin = [], [], [], []
    ergodic_from: Optional[int] = None
    for N in sorted(recs):
        rec = recs[N]
        if rec.get("ergodic_flag"):
            if ergodic_from is None:
                ergodic_from = N
            continue
        sizes = results_io.sizes_from_record(rec, "sizes_recurrent")
        basins = results_io.sizes_from_record(rec, "sizes_basins")
        Ns.append(N)
        n_rec.append(rec["n_recurrent"])
        d_max.append(sizes[0] if sizes else 0)
        max_basin.append(basins[0] if basins else (sizes[0] if sizes else 0))
    return {"N": Ns, "n_recurrent": n_rec, "d_max": d_max,
            "max_basin": max_basin, "ergodic_from": ergodic_from}


def summarize_rule(rule: int, bc: str) -> Dict:
    s = load_series(rule, bc)
    row = {
        "rule": rule,
        "tuple": "".join(rules.wolfram_to_tuple(rule)),
        "bc": bc,
        "hsf": rules.wolfram_to_hsf(rule),
        "n_points": len(s["N"]),
        "N_min": s["N"][0] if s["N"] else None,
        "N_max": s["N"][-1] if s["N"] else None,
        "ergodic_from": s["ergodic_from"],
        "n_sectors_at_Nmax": s["n_recurrent"][-1] if s["n_recurrent"] else None,
        "d_max_at_Nmax": s["d_max"][-1] if s["d_max"] else None,
    }
    for key in ("n_recurrent", "d_max", "max_basin"):
        fit = fit_series(s["N"], s[key])
        if fit.get("ok"):
            row[f"{key}_growth"] = fit["growth_class"]
            row[f"{key}_kappa"] = fit["kappa"]
            row[f"{key}_base"] = fit["base"]
            row[f"{key}_alpha"] = fit.get("alpha_M2")
            row[f"{key}_model"] = fit["best_model"]
        else:
            row[f"{key}_growth"] = "n/a"
            row[f"{key}_kappa"] = None
            row[f"{key}_base"] = None
            row[f"{key}_alpha"] = None
            row[f"{key}_model"] = None
    return row


def summarize(ruleset: List[int], bcs=("obc0", "pbc")) -> List[Dict]:
    rows = []
    for rule in ruleset:
        for bc in bcs:
            if results_io.load_results(rule, bc):
                rows.append(summarize_rule(rule, bc))
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")


def _fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def latex_summary_table(rows: List[Dict], path: str):
    """Emit a booktabs table of the per-rule scaling summary (\\input-able)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = {"exponential": 0, "polynomial": 1, "constant": 2, "n/a": 3}
    rows = sorted(rows, key=lambda r: (r["bc"],
                                       order.get(r["d_max_growth"], 9),
                                       -(r["d_max_kappa"] or 0)))
    lines = [
        r"\begin{tabular}{@{}rl l r rr l l@{}}",
        r"\toprule",
        r"W & tuple & bc & $N_{\max}$ & \#sec & $D_{\max}$ & "
        r"\#sec growth & $D_{\max}$ growth (base) \\",
        r"\midrule",
    ]
    for r in rows:
        if r["ergodic_from"] is not None and r["n_points"] == 0:
            growth_n = growth_d = "ergodic"
            base = ""
        else:
            growth_n = r["n_recurrent_growth"]
            gd = r["d_max_growth"]
            base = f" ({_fmt(r['d_max_base'],3)})" if gd == "exponential" else ""
            growth_d = gd
        erg = "" if r["ergodic_from"] is None else f"$^{{\\dagger{r['ergodic_from']}}}$"
        lines.append(
            f"{r['rule']} & \\texttt{{{r['tuple']}}} & {r['bc']}{erg} & "
            f"{_fmt(r['N_max'])} & {_fmt(r['n_sectors_at_Nmax'])} & "
            f"{_fmt(r['d_max_at_Nmax'])} & {growth_n} & {growth_d}{base} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["unitary", "unitary-all"],
                    default="unitary-all")
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR, "unitary_summary.csv"))
    args = ap.parse_args(argv)
    ruleset = (rules.UNITARY_RULES if args.which == "unitary-all"
               else rules.unitary_reflection_reps())
    rows = summarize(ruleset)
    write_csv(rows, args.out)
    latex_summary_table(rows, os.path.join(TEX_DIR, "tab_unitary_summary.tex"))
    # pretty print
    for r in rows:
        print(f"W{r['rule']:3d} {r['tuple']} {r['bc']:5s} "
              f"N<= {r['N_max']} sec@Nmax={r['n_sectors_at_Nmax']} "
              f"Dmax@Nmax={r['d_max_at_Nmax']} | "
              f"n_sec:{r['n_recurrent_growth']}(kappa={r['n_recurrent_kappa']}) "
              f"Dmax:{r['d_max_growth']}(base={r['d_max_base']})"
              f"{' ERG@'+str(r['ergodic_from']) if r['ergodic_from'] else ''}")
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
