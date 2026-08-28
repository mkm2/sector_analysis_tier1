"""Figure and tables for R31: the X-gate orbit-length distribution.

(a) the orbit histogram, formula against the exhaustive permutation;
(b) D_max under X against D_max under the Hadamard -- exponential collapses to
    subexponential, and the X curve is not even monotone;
(c) the per-segment cycle spectrum, which is the object that replaces R30's
    scalar kernel.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import orbit_sizes as O
from . import sector_sizes as H

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = os.path.join(os.path.dirname(HERE), "figures")
TABLES = os.path.join(os.path.dirname(HERE), "reports", "tex")
COLOR = {156: "#1f4e9c", 198: "#6fa8dc", 108: "#b03030", 201: "#e08b3a"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def figure(out: Optional[str] = None, N_meas: int = 16):
    out = out or os.path.join(FIGURES, "fig_orbit_sizes")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    # (a) formula vs the exhaustive permutation
    ax = axes[0]
    _style(ax)
    for rule, off in ((156, -0.22), (108, +0.22)):
        meas = O.hist_bruteforce(rule, N_meas, "obc0")
        sig = sorted(meas)
        ax.bar(np.array(sig) + off, [meas[s] for s in sig], 0.44,
               color=COLOR[rule],
               label=f"$W_{{{rule}}}$ measured ($2^{{N}}$ permutation)")
        form = [O.hist(rule, N_meas, "obc0")[s] for s in sig]
        ax.plot(np.array(sig) + off, form, "k_", ms=6, mew=1.1,
                label="lcm assembly" if rule == 156 else None)
        assert form == [meas[s] for s in sig]
    ax.set_yscale("log")
    ax.set_xlim(0, 45)
    ax.set_xlabel(r"orbit length $\ell$")
    ax.set_ylabel(r"$n_N(\ell)$")
    ax.set_title(rf"(a)  orbit lengths, obc0, $N={N_meas}$", fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right")

    # (b) D_max: X against H
    ax = axes[1]
    _style(ax)
    Ns = list(range(6, 19))
    for rule in (156, 108):
        ax.plot(Ns, [max(H.hist(rule, n, "obc0")) for n in Ns], "--", lw=1.4,
                color=COLOR[rule], label=f"$W_{{{rule}}}$  Hadamard")
        ax.plot(Ns, [max(O.hist(rule, n, "obc0")) for n in Ns], "o-", ms=3.4,
                lw=1.4, color=COLOR[rule], label=f"$W_{{{rule}}}$  $X$")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$D_{\max}$")
    ax.set_title(r"(b)  the largest sector: $\varphi^N\to$ subexponential",
                 fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")

    # (c) the segment cycle spectrum
    ax = axes[2]
    _style(ax)
    for l in range(4, 15):
        ct = O._hyperbolic_type(l)
        ax.plot([l] * len(ct), ct, "o", ms=2.6, color=COLOR[108], alpha=0.55)
    ls = np.arange(4, 15)
    ax.plot(ls, 3 * ls - 7, "-", lw=1.1, color="#333333",
            label=r"$3\ell-7$, the longest cycle")
    ax.plot(ls, ls + 1, ":", lw=1.4, color=COLOR[156],
            label=r"$W_{156}/W_{198}$: a single cycle, $\ell+1$")
    ax.set_xlabel(r"free sites in the segment, $\ell$")
    ax.set_ylabel("cycle lengths in $C(\\ell)$")
    ax.set_title(r"(c)  the kernel is now a CYCLE TYPE", fontsize=9.5)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def write_tables(outdir: Optional[str] = None,
                 Ns: Sequence[int] = (8, 10, 12, 14, 16, 18)):
    outdir = outdir or TABLES
    os.makedirs(outdir, exist_ok=True)
    for bc in ("obc0", "pbc"):
        rows = []
        for rule in O.RULES:
            cells = []
            for N in Ns:
                x = O.hist(rule, N, bc)
                h = H.hist(rule, N, bc)
                cells.append(r"%d/%d & %d/%d" % (sum(x.values()),
                                                 sum(h.values()),
                                                 max(x), max(h)))
            rows.append(r"$W_{%d}$ & %s \\" % (rule, " & ".join(cells)))
        head = " & ".join(r"\multicolumn{2}{c}{$N=%d$}" % n for n in Ns)
        sub = " & ".join([r"$n_X/n_H$ & $D_X/D_H$"] * len(Ns))
        with open(os.path.join(outdir, f"tab_r31_census_{bc}.tex"), "w") as fh:
            fh.write("\\begin{tabular}{l%s}\n\\toprule\nrule & %s \\\\\n"
                     " & %s \\\\\n\\midrule\n" % ("ll" * len(Ns), head, sub)
                     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    rows = []
    for l in range(1, 13):
        ct = O._hyperbolic_type(l)
        rows.append(r"$%d$ & $%d$ & $%s$ & $%d$ \\"
                    % (l, sum(ct), ",\\,".join(str(c) for c in ct),
                       l + 1))
    with open(os.path.join(outdir, "tab_r31_kernel.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llll}\n\\toprule\n$\\ell$ & $|C|$ & "
                 "$C(\\ell)$, symmetric pair & $C(\\ell)$, ascent pair \\\\\n"
                 "\\midrule\n" + "\n".join(rows)
                 + "\n\\bottomrule\n\\end{tabular}\n")
    return outdir


def main(argv: Optional[List[str]] = None) -> int:
    out = figure()
    write_tables()
    print("wrote", out + ".pdf", "and the R31 tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
