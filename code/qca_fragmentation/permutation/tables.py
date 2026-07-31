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
        # by growth CLASS, not by base: a linearly growing cycle also has a
        # base above 1 under a naive fit, and mixing the two is exactly the
        # error the kappa-band guard in scaling/sectors.py now prevents.
        bw = sum(1 for p in P if p["d_max_wcc"]["cls"] == "exponential")
        br = sum(1 for p in P if p["d_max_recurrent"]["cls"] == "exponential")
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


def tab_cycle_classes(bc: str, d: Dict) -> str:
    """Growth class of the longest CYCLE, by family: the collapse, tabulated."""
    from . import movement as mv
    cen = mv.cycle_class_census(d)
    rows = []
    for f in ("reversible", "V+reset", "V-free"):
        r = cen.get(f)
        if not r:
            continue
        rows.append(f"{f} & ${r['rules']}$ & ${r['constant']}$ & "
                    f"${r['polynomial']}$ & ${r['exponential']}$ & "
                    f"${r['irregular']}$ \\\\")
    tot = {k: sum(cen[f].get(k, 0) for f in cen)
           for k in ("rules", "constant", "polynomial", "exponential",
                     "irregular")}
    rows.append("\\hline\nall & $%d$ & $%d$ & $%d$ & $%d$ & $%d$ \\\\"
                % (tot["rules"], tot["constant"], tot["polynomial"],
                   tot["exponential"], tot["irregular"]))
    return ("\\begin{tabular}{lrrrrr}\n\\hline\n"
            "family & rules & bounded & linear & exponential & irregular "
            "\\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_growing_cycles(bc: str, d: Dict) -> str:
    """Every rule whose longest cycle grows with $N$ at all."""
    from . import movement as mv
    rows = []
    for g in mv.growing_cycles(d):
        if g["cls"] == "polynomial":
            continue
        tail = ",\\,".join(f"{v:,}".replace(",", r"\,") for v in g["tail"])
        base = "--" if g["base"] is None else f"{g['base']:.4f}"
        rows.append(f"${g['rule']}$ & \\texttt{{{g['tuple']}}} & "
                    f"{g['cls']} & ${base}$ & ${g['parent']}$ & ${tail}$ \\\\")
    return ("\\begin{tabular}{rllrrl}\n\\hline\n"
            "rule & tuple & class & base & parent & longest cycle, "
            "$N=20\\ldots24$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_reset_only(bc: str, d: Dict) -> str:
    """The 16 all-reset rules as boolean functions of the two neighbours."""
    from . import movement as mv
    rows = []
    for r in mv.reset_only_table(d):
        rows.append(f"${r['rule']}$ & \\texttt{{{r['tuple']}}} & {r['f']} & "
                    f"{'yes' if r['affine'] else 'no'} & {r['cls']} & "
                    f"${r['cyc_max']:,}$ \\\\".replace(",", r"\,"))
    return ("\\begin{tabular}{rllllr}\n\\hline\n"
            "rule & tuple & $x_i\\mapsto f(l,r)$ & GF(2)-affine & cycle class "
            "& longest cycle seen \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_movement(bc: str, d: Dict) -> str:
    """Movement 1, WCC -> recurrent, summarised by family."""
    from . import movement as mv
    rows = []
    M = mv.movement(d)
    for f in ("reversible", "V+reset", "V-free"):
        v = [m for m in M if m["family"] == f and m["drop"] is not None]
        if not v:
            continue
        dr = [m["drop"] for m in v]
        rows.append(f"{f} & ${len(v)}$ & ${sum(1 for m in v if m['stays'])}$ & "
                    f"${np.median(dr):.3f}$ & ${max(dr):.3f}$ & "
                    f"${_sci(np.median([m['cyclic_fraction'] for m in v]))}$ "
                    f"\\\\")
    return ("\\begin{tabular}{lrrrrr}\n\\hline\n"
            "family & rules with both bases & no move & median drop & "
            "max drop & median cyclic frac. \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_parents(bc: str, d: Dict) -> str:
    """Movement 2, parent -> children, one row per reversible parent."""
    from . import movement as mv
    inh = mv.load_inheritance(bc) or mv.build_inheritance(bc)
    rows = []
    for p in mv.parent_summary(d, inh):
        if not p["n_children"]:
            continue
        md = "--" if p["median_drop"] is None else f"{p['median_drop']:.3f}"
        rows.append(
            f"${p['parent']}$ & \\texttt{{{p['tuple']}}} & "
            f"${p['n_children']}$ & {p['parent_cls_cycle']} & "
            f"${p['children_exp_sector_count']}$ & ${p['children_exp_cycle']}$ "
            f"& ${p['inherits']}/{p['inherit_of']}$ & ${md}$ \\\\")
    return ("\\begin{tabular}{rlrlrrrr}\n\\hline\n"
            "parent & tuple & children & parent cycle & exp.\\ sectors & "
            "exp.\\ cycles & inherit & median drop \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def write_all(bc: str = "obc0") -> None:
    d = analysis.load(bc) or analysis.build(bc)
    os.makedirs(TEXDIR, exist_ok=True)
    for name, txt in ((f"tab_r10_coverage_{bc}", tab_coverage(bc)),
                      (f"tab_r10_families_{bc}", tab_families(bc, d)),
                      (f"tab_r10_reversible_{bc}", tab_reversible(bc, d)),
                      (f"tab_r10_cycle_classes_{bc}", tab_cycle_classes(bc, d)),
                      (f"tab_r10_growing_{bc}", tab_growing_cycles(bc, d)),
                      (f"tab_r10_resetonly_{bc}", tab_reset_only(bc, d)),
                      (f"tab_r10_movement_{bc}", tab_movement(bc, d)),
                      (f"tab_r10_parents_{bc}", tab_parents(bc, d)),
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
