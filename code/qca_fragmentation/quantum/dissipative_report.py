"""
Dissipative-rule analysis for Report R5.

Consumes the Tier-1a dissipative sweep (results/*.jsonl) and the Tier-1b engine
(quantum/peripheral.py) to produce:

  1. a census of the 240 dissipative rules by graph-level attractor structure,
  2. a curated Tier-1b attractor-type table (mixing / limit-cycle / coherent),
  3. the exact Cesaro-rank "protected coherence gap" for weakly dissipative rules,
  4. the context sec.8 dissipative regression checks.

LaTeX tables are written under reports/tex/ for R5 to \input.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List

from ..core import rules
from ..graph import scc
from .. import results_io
from ..scaling.summary import load_series, TEX_DIR
from . import peripheral as pp


# ---- graph-level attractor-structure class ---------------------------------

def structure_class(rule: int, bc: str) -> str:
    """Classify a rule's attractor structure from its largest computed N."""
    s = load_series(rule, bc)
    if not s["N"]:
        return "ergodic/none"
    nrec = s["n_recurrent"][-1]
    dmax = s["d_max"][-1]
    if nrec == 1 and dmax == 1:
        return "unique fixed point"
    if nrec == 1 and dmax > 1:
        return "single extended attractor"
    if dmax == 1:
        # several point attractors
        return "multistable (point attractors)"
    return "multiple extended attractors"


def census(bc: str) -> Counter:
    c = Counter()
    for r in range(256):
        t = rules.wolfram_to_tuple(r)
        if rules.is_unitary(t):
            continue
        if results_io.load_results(r, bc):
            c[structure_class(r, bc)] += 1
    return c


# ---- Tier-1b curated survey -------------------------------------------------

CURATED = [0, 1, 22, 28, 29, 50, 76, 178, 200, 232]


def tier1b_row(rule: int, N: int, bc: str) -> Dict:
    t = rules.wolfram_to_tuple(rule)
    R = scc.recurrent_classes(rule, N, bc, t)
    kinds = Counter()
    sizes = Counter()
    dmax_dk = 1
    for cls in R:
        at = pp.classify_attractor(cls, N, t, bc)
        kinds[at.kind] += 1
        sizes[at.size] += 1
        dmax_dk = max(dmax_dk, at.d_k)
    return {"rule": rule, "tuple": "".join(t), "n_recurrent": len(R),
            "kinds": dict(kinds), "sizes": dict(sizes), "max_dk": dmax_dk}


def cesaro_gap(rule: int, N: int, bc: str) -> Dict:
    """Full-channel fixed-point dimension (Cesaro rank) vs. the graph-resolved
    per-class fixed dimension.  The graph-resolved dimension is the sum of the
    per-recurrent-class fixed-point dimensions (each >= 1); the gap is the extra
    CROSS-class / weak-symmetry protected coherence the graph cannot see."""
    t = rules.wolfram_to_tuple(rule)
    R = scc.recurrent_classes(rule, N, bc, t)
    per_class = 0
    for cls in R:
        at = pp.classify_attractor(cls, N, t, bc)
        per_class += max(at.mult_one, 1)  # >=1 steady state per closed class
    cr = pp.cesaro_rank(N, t, bc)
    return {"rule": rule, "tuple": "".join(t), "n_recurrent": len(R),
            "classical_dim": per_class, "cesaro_rank": cr,
            "coherence_gap": cr - per_class}


# ---- context sec.8 dissipative regressions ---------------------------------

def check_regressions() -> List[str]:
    out = []
    # rule 22 pbc even N: exactly 3 point attractors
    ok22 = all(len(scc.recurrent_classes(22, N, "pbc", rules.wolfram_to_tuple(22))) == 3
               for N in (4, 6, 8))
    out.append(f"rule 22 pbc even N -> 3 point attractors: {'PASS' if ok22 else 'FAIL'}")
    # rules 28/29 wall transparency: compare recurrent-class counts
    n28 = len(scc.recurrent_classes(28, 10, "pbc", rules.wolfram_to_tuple(28)))
    n29 = len(scc.recurrent_classes(29, 10, "pbc", rules.wolfram_to_tuple(29)))
    out.append(f"rule 28/29 (wall transparency) recurrent classes at N=10 pbc: "
               f"28->{n28}, 29->{n29}")
    return out


# ---- LaTeX emission ---------------------------------------------------------

def _tex_census(path: str):
    lines = [r"\begin{tabular}{@{}l rr@{}}", r"\toprule",
             r"attractor structure & obc0 & pbc \\", r"\midrule"]
    co, cp = census("obc0"), census("pbc")
    keys = ["unique fixed point", "multistable (point attractors)",
            "single extended attractor", "multiple extended attractors"]
    for k in keys:
        lines.append(f"{k} & {co.get(k,0)} & {cp.get(k,0)} \\\\")
    lines += [r"\midrule",
              f"total dissipative rules & {sum(co.values())} & {sum(cp.values())} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _tex_tier1b(path: str, N=8, bc="pbc"):
    lines = [r"\begin{tabular}{@{}rl r l l@{}}", r"\toprule",
             r"W & tuple & \#recur & attractor kinds & max $d_k$ \\", r"\midrule"]
    for r in CURATED:
        row = tier1b_row(r, N, bc)
        kinds = ", ".join(f"{v}$\\times${k}" for k, v in row["kinds"].items())
        lines.append(f"{r} & \\texttt{{{row['tuple']}}} & {row['n_recurrent']} & "
                     f"{kinds} & {row['max_dk']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def faithfulness_census(N: int = 4, bc: str = "pbc"):
    """Split the dissipative rules by whether the Tier-1a (dephased) graph is a
    faithful picture of the coherent channel's attractor support."""
    faithful, unfaithful = [], []
    for r in range(256):
        t = rules.wolfram_to_tuple(r)
        if rules.is_unitary(t):
            continue
        d = pp.graph_faithfulness(N, t, bc)
        (faithful if d["faithful"] else unfaithful).append((r, d))
    return faithful, unfaithful


def _tex_faithful(path: str, N=4, bc="pbc"):
    f, uf = faithfulness_census(N, bc)
    lines = [r"\begin{tabular}{@{}rl rrr l@{}}", r"\toprule",
             r"W & tuple & \#rec.\ states & $\dim\mathrm{Fix}(\Phi)$ & "
             r"leaked weight & graph faithful? \\", r"\midrule"]
    yes = "yes"
    no = r"\textbf{no}"
    for r in [0, 200, 232, 28, 76, 22, 50, 178, 106]:
        t = rules.wolfram_to_tuple(r)
        d = pp.graph_faithfulness(N, t, bc)
        verdict = yes if d["faithful"] else no
        tup = "".join(t)
        lines.append(f"{r} & \\texttt{{{tup}}} & {d['n_recurrent_states']} & "
                     f"{d['dim_fix']} & {d['leak_flow']:.3f} & {verdict} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{6}}{{@{{}}l}}{{census at $N={N}$, {bc}: "
              f"{len(f)} faithful / {len(uf)} unfaithful of 240 dissipative rules}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _tex_cesaro(path: str, N=4, bc="pbc"):
    lines = [r"\begin{tabular}{@{}rl rrr@{}}", r"\toprule",
             r"W & tuple & classical dim & Cesaro rank & coherence gap \\",
             r"\midrule"]
    for r in [0, 1, 22, 28, 50, 178, 200, 232]:
        g = cesaro_gap(r, N, bc)
        lines.append(f"{r} & \\texttt{{{g['tuple']}}} & {g['classical_dim']} & "
                     f"{g['cesaro_rank']} & {g['coherence_gap']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def transient_depth_figure(bc: str, rules_list=(22, 28, 178, 50, 200, 232)):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    MARK = ["o", "s", "^", "D", "v", "P"]
    figdir = os.path.join(results_io.REPO_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, color="#e6e6e2", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plotted = False
    for i, r in enumerate(rules_list):
        s = load_series(r, bc)
        if not s["N"]:
            continue
        ax.plot(s["N"], s["transient_depth"], marker=MARK[i % len(MARK)], ms=5,
                lw=1.8, color=PALETTE[i % len(PALETTE)],
                label=f"W{r} ({''.join(rules.wolfram_to_tuple(r))})",
                markeredgecolor="white", markeredgewidth=0.6)
        plotted = True
    ax.set_xlabel("chain length $N$")
    ax.set_ylabel("transient depth")
    ax.set_title(f"Relaxation (transient depth) vs $N$ ({bc})", fontsize=11, loc="left")
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(figdir, f"fig_transient_depth_{bc}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    os.makedirs(TEX_DIR, exist_ok=True)
    print("=== context sec.8 dissipative regressions ===")
    for line in check_regressions():
        print("  " + line)
    print("=== census (obc0) ===", dict(census("obc0")))
    print("=== census (pbc)  ===", dict(census("pbc")))
    _tex_census(os.path.join(TEX_DIR, "tab_diss_census.tex"))
    _tex_tier1b(os.path.join(TEX_DIR, "tab_diss_tier1b.tex"))
    _tex_cesaro(os.path.join(TEX_DIR, "tab_diss_cesaro.tex"))
    _tex_faithful(os.path.join(TEX_DIR, "tab_diss_faithful.tex"))
    f, uf = faithfulness_census()
    print(f"=== graph faithfulness (N=4,pbc): {len(f)} faithful / {len(uf)} unfaithful "
          f"of 240 dissipative rules ===")
    for bc in ("pbc", "obc0"):
        transient_depth_figure(bc)
    print("wrote R5 tables to", TEX_DIR)


if __name__ == "__main__":
    main()
