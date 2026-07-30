"""Tables for R10 (X-gate / permutation circuits)."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict

import numpy as np

from .. import results_io
from ..core import rules
from . import analysis

TEXDIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")


def _sci(v: float, sig: int = 3) -> str:
    """LaTeX scientific notation; plain 'e-05' does not render in math mode."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e4:
        return f"{v:.{sig}g}"
    m, e = f"{v:.{sig - 1}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def tab_coverage(bc: str) -> str:
    per_n: Counter = Counter()
    for rule in range(256):
        for N in results_io.load_xgate_results(rule, bc):
            per_n[N] += 1
    rows = [f"${N}$ & ${per_n[N]}$ & ${2 ** N:,}$".replace(",", r"\,") + r" \\"
            for N in sorted(per_n)]
    return ("\\begin{tabular}{rrr}\n\\hline\n"
            "$N$ & rules covered & $2^N$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_families(bc: str, d: Dict) -> str:
    """The headline contrast: exponential sectors vs exponential cycles."""
    rows = []
    for f in ("reversible", "V+reset", "V-free"):
        P = [p for p in d["points"] if p["family"] == f]
        if not P:
            continue
        bw = sum(1 for p in P if (p["d_max_wcc"]["base"] or 1) > 1.02)
        br = sum(1 for p in P if (p["d_max_recurrent"]["base"] or 1) > 1.02)
        cf = np.median([p["cyclic_fraction_at_Nmax"] for p in P])
        td = np.median([p["transient_depth_at_Nmax"] for p in P])
        rows.append(f"{f} & ${len(P)}$ & ${bw}$ & ${br}$ & "
                    f"${_sci(cf)}$ & ${td:.0f}$ \\\\")
    return ("\\begin{tabular}{lrrrrr}\n\\hline\n"
            "family & rules & exp.\\ sectors & exp.\\ cycles & "
            "median cyclic frac. & median depth \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_reversible(bc: str, d: Dict) -> str:
    """The 16 reversible rules: sectors ARE cycles."""
    pts = {p["rule"]: p for p in d["points"]}
    rows = []
    for r in sorted(rules.UNITARY_RULES):
        p = pts.get(r)
        if not p:
            continue
        s = analysis.load_series(r, bc, analysis.UNIFORM_N_CAP)
        a, b = p["n_wcc"]["base"], p["d_max_wcc"]["base"]
        rows.append(
            f"${r}$ & \\texttt{{{p['tuple']}}} & ${a:.5f}$ & ${b:.5f}$ & "
            f"${a * b:.4f}$ & ${s['n_wcc'][-1]:,}$ & ${s['d_max_wcc'][-1]:,}$ "
            f"\\\\".replace(",", r"\,"))
    return ("\\begin{tabular}{rlrrrrr}\n\\hline\n"
            "rule & tuple & $a$ & $b$ & $ab$ & $n_{\\rm wcc}$ & "
            "$D_{\\max}$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_validation(bc: str, d: Dict) -> str:
    idn = d["identity"]
    fh = d["finite_hyperbola"]
    worst = min(fh, key=lambda f: f["min_ratio"])
    nb_sec = sum(1 for c in d["products"]
                 if c["map"] == "sector" and c["below_2"])
    n_sec = sum(1 for c in d["products"] if c["map"] == "sector")
    nb_att = sum(1 for c in d["products"]
                 if c["map"] == "attractor" and c["below_2"])
    n_att = sum(1 for c in d["products"] if c["map"] == "attractor")
    rows = [
        f"sector sizes partition, $\\sum_k D_k = 2^N$ & ${idn['units']}$ & "
        f"${idn['units']}$ & asserted in the analysis \\\\",
        f"$n_{{\\rm recurrent}} = n_{{\\rm wcc}}$ (F1) & ${idn['units']}$ & "
        f"${idn['identity_holds']}$ & ${idn['identity_fails']}$ failures \\\\",
        f"finite-$N$ $n_{{\\rm wcc}}D_{{\\max}}\\geq2^N$ & ${len(fh)}$ & "
        f"${len(fh) - len(d['finite_failures'])}$ & tightest "
        f"${worst['min_ratio']:.4f}$ at $N={worst['at_N']}$ \\\\",
        f"sector map $ab\\geq2$ & ${n_sec}$ & ${n_sec - nb_sec}$ & "
        f"${nb_sec}$ below \\\\",
        f"cycle map $ab\\geq2$ (\\emph{{not}} required) & ${n_att}$ & "
        f"${n_att - nb_att}$ & ${nb_att}$ below, as expected \\\\",
    ]
    return ("\\begin{tabular}{lrrl}\n\\hline\n"
            "check & applicable & passing & note \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def write_all(bc: str = "obc0") -> None:
    d = analysis.load(bc) or analysis.build(bc)
    os.makedirs(TEXDIR, exist_ok=True)
    for name, txt in ((f"tab_r10_coverage_{bc}", tab_coverage(bc)),
                      (f"tab_r10_families_{bc}", tab_families(bc, d)),
                      (f"tab_r10_reversible_{bc}", tab_reversible(bc, d)),
                      (f"tab_r10_validation_{bc}", tab_validation(bc, d))):
        p = os.path.join(TEXDIR, f"{name}.tex")
        with open(p, "w") as f:
            f.write(txt)
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R10 tables")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    write_all(ap.parse_args(argv).bc)


if __name__ == "__main__":
    main()
