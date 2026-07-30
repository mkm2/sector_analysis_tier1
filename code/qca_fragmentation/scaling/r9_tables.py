"""
Tables for R9 (Tier 1e: sectors and basins).

Every table caption in R9 states which EXPERIMENT its numbers describe.  The
convention enforced here:

    sector / n_wcc / d_max_wcc      exact for BOTH monitored and unmonitored
    n_recurrent / basins / depth    MONITORED only (terminal SCCs)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional

from .. import results_io
from ..core import rules
from . import sectors

TEXDIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")
PHI = (1 + 5 ** 0.5) / 2


def _fmt(n) -> str:
    if n is None:
        return "---"
    if isinstance(n, float):
        return f"{n:.4f}"
    s = f"{n:,}".replace(",", r"\,")
    return s


def tab_coverage(bc: str) -> str:
    """Rules x N coverage of the Tier-1e store."""
    per_n: Counter = Counter()
    per_rule_max: Dict[int, int] = {}
    n_erg = 0
    for rule in range(256):
        recs = results_io.load_wcc_results(rule, bc)
        if not recs:
            continue
        per_rule_max[rule] = max(recs)
        for N, r in recs.items():
            per_n[N] += 1
            n_erg += bool(r.get("ergodic_flag"))
    rows = []
    for N in sorted(per_n):
        rows.append(f"${N}$ & ${per_n[N]}$ & ${2 ** N:,}$".replace(",", r"\,")
                    + r" \\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{rrr}\n\\hline\n"
            "$N$ & rules covered & $2^N$ \\\\\n\\hline\n"
            f"{body}\n\\hline\n\\end{{tabular}}\n")


def tab_anchors(bc: str, d: Dict) -> str:
    """V2/V3: the anchors of the exclusion curve."""
    want = {204: ("$2$", "$1$", "on"), 51: ("$1$", "$2$", "on"),
            150: ("$1$", "$2$", "on"), 156: (r"$\varphi$", "$4^{1/5}$", "above")}
    rows = []
    for r in (204, 51, 150, 156):
        c = next((x for x in d["hyperbola"] if x["rule"] == r), None)
        if c is None:
            continue
        wa, wb, wh = want[r]
        mark = {"on_curve": r"\cmark\ on", "above": r"\cmark\ above",
                "inconclusive": r"\xmark\ inconclusive",
                "degenerate_subexponential": r"\xmark\ degenerate",
                "irregular": r"\xmark\ irregular",
                "below_within_uncertainty": r"\xmark\ below",
                "VIOLATION": r"\xmark\ VIOLATION"}[c["verdict"]]
        rows.append(
            f"${r}$ & \\texttt{{{''.join(rules.wolfram_to_tuple(r))}}} & "
            f"{wa} & {wb} & {wh} & ${c['a']:.4f}$ & ${c['b']:.4f}$ & "
            f"${c['product']:.4f}$ & {mark} \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{rlccrrrrl}\n\\hline\n"
            "rule & tuple & $a$ want & $b$ want & where & $a$ & $b$ & $ab$ & "
            "verdict \\\\\n\\hline\n" f"{body}\n\\hline\n\\end{{tabular}}\n")


def tab_verdicts(bc: str, d: Dict) -> str:
    order = ["on_curve", "above", "degenerate_subexponential", "irregular",
             "below_within_uncertainty", "inconclusive", "VIOLATION"]
    label = {"on_curve": r"on the curve, $ab=2$",
             "above": r"above, $ab>2$",
             "degenerate_subexponential":
                 r"sub-exponential factor: $ab$ test degenerate",
             "irregular": r"irregular series, no growth law",
             "below_within_uncertainty":
                 r"below but inside the fit uncertainty",
             "inconclusive": r"below, slack too wide to conclude",
             "VIOLATION": r"\textbf{violation}"}
    rows = [f"{label[k]} & ${d['verdicts'].get(k, 0)}$ \\\\" for k in order]
    rows.append(r"\hline")
    rows.append(f"total & ${d['n_rules']}$ \\\\")
    return ("\\begin{tabular}{lr}\n\\hline\n"
            "verdict & rules \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_below(bc: str, d: Dict) -> str:
    """Every rule with a*b < 2, listed as the task requires."""
    rows = []
    for c in sorted(d["raw_below"], key=lambda x: x["product"]):
        tup = "".join(rules.wolfram_to_tuple(c["rule"]))
        hi = c["b_fit_hi"]
        hi_cell = "---" if hi is None else f"${hi:.4f}$"
        rows.append(
            f"${c['rule']}$ & \\texttt{{{tup}}} & ${c['a']:.4f}$ & "
            f"${c['b']:.4f}$ & ${c['product']:.4f}$ & ${c['slack']:.3f}$ & "
            f"${c['b_lower_bound']:.4f}$ & {hi_cell} \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{rlrrrrrr}\n\\hline\n"
            "rule & tuple & $a$ & $b$ & $ab$ & slack & $2/a$ & fit $b_{\\rm hi}$ "
            "\\\\\n\\hline\n" f"{body}\n\\hline\n\\end{{tabular}}\n")


def tab_growth_classes(bc: str, d: Dict) -> str:
    """Growth-class census for n_wcc and d_max_wcc across the rule space."""
    fams = ["unitary", "classical", "mixed"]
    out = []
    for key, lab in (("n_wcc", r"$\#$sectors"),
                     ("d_max_wcc", r"$D_{\max}$")):
        cnt = {f: Counter() for f in fams}
        for p in d["points"]:
            cnt[p["family"]][p[key]["cls"]] += 1
        classes = ["constant", "polynomial", "exponential", "irregular"]
        for f in fams:
            row = " & ".join(f"${cnt[f].get(c, 0)}$" for c in classes)
            n_ex = sum(1 for p in d["points"]
                       if p["family"] == f and p[key]["exact"])
            out.append(f"{lab} & {f} & {row} & ${n_ex}$ \\\\")
    return ("\\begin{tabular}{llrrrrr}\n\\hline\n"
            "series & family & const & poly & exp & irreg & exact base "
            "\\\\\n\\hline\n" + "\n".join(out) + "\n\\hline\n\\end{tabular}\n")


def tab_exact_bases(bc: str, d: Dict, limit: int = 22) -> str:
    """The exact-base census: rules whose sector law is an integer recurrence."""
    rows = []
    seen = set()
    cand = [p for p in d["points"] if p["n_wcc"]["base"] is not None]
    for p in sorted(cand, key=lambda x: -x["n_wcc"]["base"]):
        a = p["n_wcc"]
        if not a["exact"] or a["base"] <= 1.0:
            continue
        key = (round(a["base"], 8), a["named"])
        if key in seen and len(rows) > limit // 2:
            continue
        seen.add(key)
        rows.append(
            f"${p['rule']}$ & \\texttt{{{p['tuple']}}} & ${a['base']:.5f}$ & "
            f"{a['named'] or '---'} & {a['source']} & "
            f"${p['d_max_wcc']['base']:.4f}$ & ${p['product_ab']:.4f}$ \\\\")
        if len(rows) >= limit:
            break
    return ("\\begin{tabular}{rlrllrr}\n\\hline\n"
            "rule & tuple & $a$ & name & source & $b$ & $ab$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_transient(bc: str, d: Dict, limit: int = 16) -> str:
    """MONITORED transient dominance and attractors per enclosure."""
    pts = [p for p in d["points"]
           if p["transient_fraction_at_Nmax"] is not None]
    pts.sort(key=lambda p: -(p["transient_fraction_at_Nmax"] or 0))
    rows = []
    for p in pts[:limit]:
        rows.append(
            f"${p['rule']}$ & \\texttt{{{p['tuple']}}} & ${p['N_max']}$ & "
            f"${p['transient_fraction_at_Nmax']:.4f}$ & "
            f"${p['att_per_sector_at_Nmax']:.3f}$ & "
            f"${p['d_max_ratio_at_Nmax']:.4f}$ \\\\")
    return ("\\begin{tabular}{rlrrrr}\n\\hline\n"
            "rule & tuple & $N$ & transient frac. & att./sector & "
            "$D_{\\max}^{\\rm wcc}/2^N$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_validation(bc: str) -> str:
    from .sector_figure import validation_stats
    st = validation_stats(bc)
    n_basin = st["basin_ok"] + st["basin_bad"] + st["basin_unknown"]
    rows = [
        f"A1 sum rule $\\sum_k D_k = 2^N$ & ${st['n_units']}$ & "
        f"${st['n_units']}$ & max residual ${st['sum_rule_max_abs_residual']}$ \\\\",
        f"A2 unitary $n_{{\\rm wcc}}=n_{{\\rm rec}}=n_{{\\rm scc}}$ & "
        f"${st['a2_total']}$ & ${st['a2_ok']}$ & size multisets identical \\\\",
        f"basin sum rule (MONITORED) & ${n_basin}$ & ${st['basin_ok']}$ & "
        f"${st['basin_unknown']}$ still unavailable \\\\",
        f"V1 hyperbola $ab\\geq2$ & ${st['n_units'] and len(st['hyperbola_margins'])}$ "
        f"& ${len(st['hyperbola_margins']) - st['hyperbola_violations']}$ & "
        f"${st['hyperbola_violations']}$ violations \\\\",
    ]
    return ("\\begin{tabular}{lrrl}\n\\hline\n"
            "check & applicable & passing & note \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_basin_recompute(bc: str) -> str:
    """The independent basin recomputation (user request), and what remains."""
    n, ok, agree, rt = 0, 0, 0, 0.0
    nmax = 0
    for rule in range(256):
        for N, r in results_io.load_basin_results(rule, bc).items():
            n += 1
            ok += bool(r["basin_sum_check"])
            a = r.get("agrees_with_tier1a") or {}
            agree += bool(a and all(a.values()))
            rt += r.get("runtime") or 0.0
            nmax = max(nmax, N)
    # what is still not checkable, and why -- the distinction matters: a gap in
    # the archive is a defect, an undefined check is not
    cats = Counter()
    for rule in range(256):
        arch = results_io.load_results(rule, bc)
        rb = results_io.load_basin_results(rule, bc)
        for N, rec in results_io.load_wcc_results(rule, bc).items():
            if rec.get("basin_sum_check") is not None:
                continue
            ar = arch.get(N)
            if ar is None:
                cats["N beyond the Tier-1a sweep"] += 1
            elif ar.get("ergodic_flag") or ar.get("sizes_basins") is None:
                cats["Tier-1a unit ergodic: basins never computed"] += 1
            else:
                cats["truncated and NOT yet recomputed"] += 1
    rows = [
        f"units recomputed & ${n}$ \\\\",
        f"largest $N$ recomputed & ${nmax}$ \\\\",
        f"basin sum rule holds & ${ok}/{n}$ \\\\",
        f"agrees with the surviving Tier-1a fields & ${agree}/{n}$ \\\\",
        f"recomputation time & ${rt / 3600:.2f}$~h \\\\",
        r"\hline",
    ]
    for k in ("truncated and NOT yet recomputed",
              "Tier-1a unit ergodic: basins never computed",
              "N beyond the Tier-1a sweep"):
        rows.append(f"{k} & ${cats.get(k, 0)}$ \\\\")
    return ("\\begin{tabular}{lr}\n\\hline\n"
            "quantity & value \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def write_all(bc: str = "obc0") -> None:
    d = sectors.load(bc) or sectors.build(bc)
    os.makedirs(TEXDIR, exist_ok=True)
    tables = [
        (f"tab_r9_coverage_{bc}", tab_coverage(bc)),
        (f"tab_r9_anchors_{bc}", tab_anchors(bc, d)),
        (f"tab_r9_verdicts_{bc}", tab_verdicts(bc, d)),
        (f"tab_r9_below_{bc}", tab_below(bc, d)),
        (f"tab_r9_classes_{bc}", tab_growth_classes(bc, d)),
        (f"tab_r9_exact_{bc}", tab_exact_bases(bc, d)),
        (f"tab_r9_transient_{bc}", tab_transient(bc, d)),
        (f"tab_r9_validation_{bc}", tab_validation(bc)),
        (f"tab_r9_basins_{bc}", tab_basin_recompute(bc)),
        (f"tab_r9_dissip_{bc}", tab_dissipation_clusters(bc, d)),
    ]
    for name, txt in tables:
        p = os.path.join(TEXDIR, f"{name}.tex")
        with open(p, "w") as f:
            f.write(txt)
        print("wrote", p)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R9 tables")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    a = ap.parse_args(argv)
    write_all(a.bc)




def tab_dissipation_clusters(bc: str, d: Dict) -> str:
    """
    The 160 V+reset rules grouped by coherent correspondent (D, E -> I), and
    what adding the reset does to their position in the sector plane.
    """
    from .sector_figure import _plane_points
    xy = _plane_points(d, "sector")
    rows, tot = [], dict(k=0, c=0, s=0)
    recs = []
    for p in sorted(rules.UNITARY_RULES):
        kids = [c for c in rules.dissipative_children(p) if c in xy]
        if not kids or p not in xy:
            continue
        pa, pb = xy[p]
        n12 = sum(1 for c in kids
                  if abs(xy[c][0] - 1) < 1e-6 and abs(xy[c][1] - 2) < 1e-6)
        st = sum(1 for c in kids
                 if abs(xy[c][0] - pa) < 1e-6 and abs(xy[c][1] - pb) < 1e-6)
        trivial = abs(pa - 1) < 1e-6 and abs(pb - 2) < 1e-6
        recs.append((p, pa, pb, len(kids), n12, st, trivial))
        tot["k"] += len(kids)
        tot["c"] += n12
        tot["s"] += st
    recs.sort(key=lambda r: (r[6], -(abs(r[1] - 1) + abs(r[2] - 2)), r[0]))
    for p, pa, pb, nk, n12, st, trivial in recs:
        tup = "".join(rules.wolfram_to_tuple(p))
        keeps = "---" if trivial else r"\textbf{0}"
        rows.append(
            f"${p}$ & \\texttt{{{tup}}} & "
            f"${pa:.4f}$ & ${pb:.4f}$ & ${nk}$ & ${n12}$ & ${st}$ & "
            f"{keeps} \\\\")
    nt = [r for r in recs if not r[6]]
    body = "\n".join(rows)
    foot = (r"\hline" "\n"
            f"all & & & & ${tot['k']}$ & ${tot['c']}$ & ${tot['s']}$ & \\\\")
    return ("\\begin{tabular}{rlrrrrrc}\n\\hline\n"
            "parent & tuple & $a$ & $b$ & children & $\\to(1,2)$ & "
            "at parent & keeps structure \\\\\n\\hline\n"
            f"{body}\n{foot}\n\\hline\n\\end{{tabular}}\n")

if __name__ == "__main__":
    main()
