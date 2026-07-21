"""
Tier-1c scaling extraction for the 240 DISSIPATIVE rules (PLAN.md T6/T7).

The unitary counterpart (`scaling/summary.py`) fits the sector count and the
largest sector.  For a dissipative rule the same two series mean something
different -- the graph is directed, so

    n_recurrent  = number of ATTRACTORS (recurrent classes),
    d_max        = size of the largest attractor,
    max_basin    = size of the largest basin of attraction,
    transient_depth = longest condensation-DAG path (relaxation-time proxy),

and we extract the exponential base of each, exactly as in the unitary case:
BIC selection over  ln y = c (+ alpha ln N) (+ kappa N),  base = e^kappa.

PARITY.  Unlike the unitary sweep, many dissipative rules oscillate strongly
between even and odd N -- a pbc chain of odd length cannot host the two Neel
states, so e.g. rule 22 has 3 attractors at even N and 2 at odd N, while its
largest attractor is a single state at even N but grows like 2N at odd N.  A
single fit over N=6..13 then reports a meaningless average.  We therefore fit
three times (all N, even N, odd N), detect the split by comparing residuals,
and report the parity-resolved rate when the series is split.  `kappa_eff` is
the number to plot: the joint fit when the series is parity-clean, otherwise
the parity branch with the larger |kappa| (the dominant growth).

Outputs
    figures/dissipative_summary.csv     one row per (rule, bc)
    reports/tex/tab_diss_scaling.tex    the representative-rule table for R5
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results_io
from ..core import rules
from .fits import find_integer_recurrence, fit_pure_exponential, fit_series
from .summary import load_series

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")
ANALYTICS_DIR = os.path.join(results_io.REPO_ROOT, "analytics")

SERIES_KEYS = ("n_recurrent", "d_max", "max_basin", "transient_depth")

# A joint fit whose ln-residuals exceed this (rms) is suspect...
_RESID_BAD = 0.15
# ...and is declared parity-split if splitting by parity cuts the residual by
# at least this factor.
_RESID_GAIN = 2.0
# Scatter (rms of ln y) that survives the parity split: the series is then not
# described by any smooth growth law and is reported as "irregular".
_RESID_NOISY = 0.35


def dissipative_rules() -> List[int]:
    return [r for r in range(256)
            if not rules.is_unitary(rules.wolfram_to_tuple(r))]


def _rms_resid(Ns: Sequence[float], ys: Sequence[float], fit: Dict) -> float:
    """rms residual of the SELECTED model on ln y."""
    if not fit.get("ok"):
        return float("nan")
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    good = ys > 0
    Ns, ys = Ns[good], ys[good]
    beta = np.asarray(fit["params"][fit["best_model"]], float)
    cols = [np.ones_like(Ns)]
    if len(beta) > 1:
        cols.append(np.log(Ns))
    if len(beta) > 2:
        cols.append(Ns)
    pred = np.column_stack(cols) @ beta
    r = np.log(ys) - pred
    return float(np.sqrt((r * r).mean()))


def fit_with_parity(Ns: Sequence[int], ys: Sequence[int]) -> Dict:
    """Fit a series over all N and over each parity separately.

    The growth CLASS comes from the BIC selection (`fit_series`); the reported
    RATE comes from the two-parameter `fit_pure_exponential`, because the
    alpha*ln N term of M2 is unstable on the ragged dissipative series (see
    fits.fit_pure_exponential).  A series that is still scattered after the
    parity split is labelled "irregular" and carries no rate at all.
    """
    Ns = list(Ns)
    ys = list(ys)
    out: Dict = {"all": fit_series(Ns, ys), "all_exp": fit_pure_exponential(Ns, ys)}
    for name, keep in (("even", 0), ("odd", 1)):
        sn = [n for n in Ns if n % 2 == keep]
        sy = [y for n, y in zip(Ns, ys) if n % 2 == keep]
        out[name] = (fit_series(sn, sy) if len(sn) >= 3
                     else {"ok": False, "reason": "too few"})
        out[name + "_exp"] = fit_pure_exponential(sn, sy)

    r_all = _rms_resid(Ns, ys, out["all"])
    r_par = [_rms_resid([n for n in Ns if n % 2 == k],
                        [y for n, y in zip(Ns, ys) if n % 2 == k], out[p])
             for k, p in ((0, "even"), (1, "odd"))]
    r_par = [r for r in r_par if r == r]  # drop nan
    split = False
    if r_all == r_all and r_all > _RESID_BAD and r_par:
        split = max(r_par) * _RESID_GAIN < r_all
    out["parity_split"] = split
    out["resid_all"] = r_all
    out["resid_eff"] = max(r_par) if (split and r_par) else r_all

    order = {"exponential": 0, "polynomial": 1, "constant": 2}
    if split:
        exps = [out[p + "_exp"] for p in ("even", "odd") if out[p + "_exp"].get("ok")]
        classes = [out[p]["growth_class"] for p in ("even", "odd") if out[p].get("ok")]
    else:
        exps = [out["all_exp"]] if out["all_exp"].get("ok") else []
        classes = [out["all"]["growth_class"]] if out["all"].get("ok") else []

    growth = min(classes, key=lambda c: order.get(c, 9)) if classes else "n/a"
    kappa = max((e["kappa"] for e in exps), key=abs) if exps else None

    # Guards: an unbounded rate, or residual scatter that survives the parity
    # split, means the series has no exponential description at all.
    unbounded = any(not e["bounded"] for e in exps)
    noisy = out["resid_eff"] == out["resid_eff"] and out["resid_eff"] > _RESID_NOISY
    if unbounded or noisy:
        growth, kappa = "irregular", None

    out["growth_eff"] = growth
    out["kappa_eff"] = kappa
    out["irregular"] = growth == "irregular"
    return out


def attractor_class(n_rec: Optional[int], d_max: Optional[int]) -> str:
    """Coarse graph-level attractor structure at the largest computed N."""
    if not n_rec or not d_max:
        return "n/a"
    if d_max == 1:
        return "unique-point" if n_rec == 1 else "multistable"
    return "extended" if n_rec == 1 else "multi-extended"


def load_coherent_census(path: Optional[str] = None) -> Dict:
    """The Tier-1b coherent-attractor census (analytics/coherent_attractor_census.json).

    Returns {} if the census has not been generated; callers must degrade
    gracefully (the scaling extraction itself does not depend on it).
    """
    path = path or os.path.join(ANALYTICS_DIR, "coherent_attractor_census.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def summarize_rule(rule: int, bc: str, coherent: Optional[set] = None) -> Dict:
    s = load_series(rule, bc)
    t = "".join(rules.wolfram_to_tuple(rule))
    row = {
        "rule": rule,
        "tuple": t,
        "bc": bc,
        "n_points": len(s["N"]),
        "N_min": s["N"][0] if s["N"] else None,
        "N_max": s["N"][-1] if s["N"] else None,
        "n_att_at_Nmax": s["n_recurrent"][-1] if s["n_recurrent"] else None,
        "d_max_at_Nmax": s["d_max"][-1] if s["d_max"] else None,
        "depth_at_Nmax": s["transient_depth"][-1] if s["transient_depth"] else None,
    }
    row["attractor_class"] = attractor_class(row["n_att_at_Nmax"],
                                             row["d_max_at_Nmax"])
    row["coherent_attractor"] = (rule in coherent) if coherent is not None else None
    row["wall_hadamard"] = (t[1] == "V" and t[2] == "V")
    for key in SERIES_KEYS:
        f = fit_with_parity(s["N"], s[key])
        row[f"{key}_growth"] = f["growth_eff"]
        row[f"{key}_kappa"] = f["kappa_eff"]
        row[f"{key}_base"] = (np.exp(f["kappa_eff"])
                              if f["kappa_eff"] is not None else None)
        row[f"{key}_parity_split"] = f["parity_split"]
        row[f"{key}_irregular"] = f["irregular"]
        row[f"{key}_resid"] = f["resid_eff"]
        row[f"{key}_model"] = f["all"].get("best_model")
        for p in ("even", "odd"):
            e = f[p + "_exp"]
            row[f"{key}_kappa_{p}"] = e["kappa"] if e.get("ok") else None
        # EXACT base: an integer linear recurrence satisfied by the whole
        # series pins the growth base as an algebraic number.  Only meaningful
        # for a parity-clean series (a split series has two interleaved laws).
        rec = (find_integer_recurrence(s[key])
               if (s[key] and not f["parity_split"]) else {"ok": False})
        row[f"{key}_rec_order"] = rec["order"] if rec["ok"] else None
        row[f"{key}_rec_coeffs"] = (",".join(map(str, rec["coeffs"]))
                                    if rec["ok"] else None)
        row[f"{key}_base_exact"] = rec["base"] if rec["ok"] else None
        row[f"{key}_base_name"] = rec.get("name") if rec["ok"] else None
    return row


def summarize(bcs=("pbc", "obc0")) -> List[Dict]:
    census = load_coherent_census()
    # a rule counts as "coherent" if its attractor is non-basis-spanned at ANY
    # of the censused sizes (the census is per-N; see analytics/ JSON)
    coherent = set()
    for _N, entry in census.get("by_N", {}).items():
        coherent.update(int(r) for r in entry.get("coherent_rules", []))
    # ...and only for the bc the census was actually run under: it is a pbc
    # census, and pasting its verdicts onto obc0 rows would be a fabrication.
    census_bc = census.get("bc")
    rows = []
    for rule in dissipative_rules():
        for bc in bcs:
            if results_io.load_results(rule, bc):
                rows.append(summarize_rule(
                    rule, bc, coherent if (census and bc == census_bc) else None))
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# Rules shown explicitly in R5's scaling table: the ground-truth cases, the
# fastest growers, and the wall-Hadamard core.
TABLE_RULES = [0, 4, 22, 28, 50, 146, 178, 200, 232, 3, 19, 35]


def latex_table(rows: List[Dict], path: str, bc: str = "pbc",
                only: Optional[Sequence[int]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    only = TABLE_RULES if only is None else only
    sel = [r for r in rows if r["bc"] == bc and r["rule"] in only]
    sel.sort(key=lambda r: only.index(r["rule"]))
    lines = [
        r"\begin{tabular}{@{}rl rr l l l@{}}",
        r"\toprule",
        r"W & tuple & \#att & $D_{\max}$ & \#att growth (base) & "
        r"$D_{\max}$ growth (base) & attractor class \\",
        r"\midrule",
    ]
    for r in sel:
        def g(key):
            cls = r[f"{key}_growth"]
            s = cls
            if cls == "exponential":
                exact, nm, b = (r[f"{key}_base_exact"], r[f"{key}_base_name"],
                                r[f"{key}_base"])
                if exact:
                    s += f" ({exact:.4f}" + (f", {nm}" if nm else "") + ")"
                elif b:
                    s += f" ({b:.3f})"
            if r[f"{key}_parity_split"]:
                s += r"$^{\ast}$"
            return s
        lines.append(
            f"{r['rule']} & \\texttt{{{r['tuple']}}} & "
            f"{_fmt(r['n_att_at_Nmax'])} & {_fmt(r['d_max_at_Nmax'])} & "
            f"{g('n_recurrent')} & {g('d_max')} & {r['attractor_class']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR,
                                                  "dissipative_summary.csv"))
    args = ap.parse_args(argv)
    rows = summarize()
    write_csv(rows, args.out)
    latex_table(rows, os.path.join(TEX_DIR, "tab_diss_scaling.tex"))

    import collections
    for bc in ("pbc", "obc0"):
        sub = [r for r in rows if r["bc"] == bc]
        gn = collections.Counter(r["n_recurrent_growth"] for r in sub)
        gd = collections.Counter(r["d_max_growth"] for r in sub)
        ps = sum(1 for r in sub if r["n_recurrent_parity_split"]
                 or r["d_max_parity_split"])
        print(f"{bc}: {len(sub)} rules | #att growth {dict(gn)} | "
              f"Dmax growth {dict(gd)} | parity-split {ps}")
        exact = collections.Counter()
        for r in sub:
            for key in ("n_recurrent", "d_max"):
                if r[f"{key}_growth"] == "exponential" and r[f"{key}_base_exact"]:
                    exact[round(r[f"{key}_base_exact"], 6)] += 1
        print(f"    exact algebraic bases ({sum(exact.values())} series): "
              + ", ".join(f"{b} x{c}" for b, c in exact.most_common()))
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
