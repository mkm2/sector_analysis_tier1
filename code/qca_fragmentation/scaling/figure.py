"""
Tier-1c figures for the unitary sweep (context Tier 1 sec.7).

Produces, from results/{rule}_{bc}.jsonl:
  fig_nsectors_{bc}.pdf   ln(#sectors) vs N for the fragmented rules,
  fig_dmax_{bc}.pdf       ln(D_max)    vs N with fitted exponential slopes,
  fig_growth_scatter.pdf  (kappa of #sectors, kappa of D_max) per rule.

Design follows the dataviz skill: fixed-order CVD-validated categorical hues,
distinct markers as secondary encoding, thin marks, a legend plus direct labels,
recessive grid, no dual axes.  Vector PDF for the report + PNG for quick view.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from ..core import rules
from .fits import fit_series
from .summary import load_series

# CVD-validated categorical hues in fixed order (dataviz reference palette, light).
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT = "#0b0b0b"
MUTED = "#8a8a86"

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

# The fragmented unitary rules worth plotting (distinct behaviours; 198 == 156
# by reflection so only 156 is shown).
FRAGMENTED = [150, 201, 108, 156, 60, 102, 105]


def _style(ax):
    ax.grid(True, color="#e6e6e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _label(rule: int) -> str:
    t = "".join(rules.wolfram_to_tuple(rule))
    h = rules.wolfram_to_hsf(rule)
    return f"W{rule} ({t})"


def fig_series(bc: str, key: str, ylabel: str, title: str, out: str,
               rules_list: List[int] = None):
    rules_list = rules_list or FRAGMENTED
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _style(ax)
    for i, rule in enumerate(rules_list):
        s = load_series(rule, bc)
        if len(s["N"]) < 2:
            continue
        Ns = np.array(s["N"], float)
        y = np.array(s[key], float)
        y = np.where(y > 0, y, np.nan)
        c = PALETTE[i % len(PALETTE)]
        ax.plot(Ns, np.log(y), marker=MARKERS[i % len(MARKERS)], ms=5,
                lw=1.8, color=c, label=_label(rule), markeredgecolor="white",
                markeredgewidth=0.6)
        # direct label at the last point
        xr, yr = Ns[-1], np.log(y[np.isfinite(y)][-1])
        ax.annotate(f"W{rule}", (xr, yr), textcoords="offset points",
                    xytext=(6, 0), color=c, fontsize=8, va="center")
    ax.set_xlabel("chain length $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_growth_scatter(bc: str, out: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    _style(ax)
    marker_for = {"exponential": "o", "polynomial": "s", "constant": "^"}
    color_for = {"exponential": "#e34948", "polynomial": "#2a78d6",
                 "constant": "#8a8a86"}
    seen = set()
    for rule in rules.UNITARY_RULES:
        s = load_series(rule, bc)
        if len(s["N"]) < 3:
            continue
        fn = fit_series(s["N"], s["n_recurrent"])
        fd = fit_series(s["N"], s["d_max"])
        if not (fn.get("ok") and fd.get("ok")):
            continue
        kn = fn["kappa"] or 0.0
        kd = fd["kappa"] or 0.0
        gc = fd["growth_class"]
        ax.scatter(kn, kd, s=70, color=color_for.get(gc, "#000"),
                   marker=marker_for.get(gc, "o"), edgecolor="white",
                   linewidth=0.8, zorder=3,
                   label=gc if gc not in seen else None)
        seen.add(gc)
        ax.annotate(f"W{rule}", (kn, kd), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color=TEXT)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$\kappa$ of #sectors  (exp. rate of sector count)")
    ax.set_ylabel(r"$\kappa$ of $D_{\max}$  (exp. rate of largest sector)")
    ax.set_title(f"Fragmentation growth-rate map ({bc})", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, title="D_max growth", loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for bc in ("obc0", "pbc"):
        fig_series(bc, "n_recurrent", r"$\ln(\#\,\mathrm{sectors})$",
                   f"Sector count vs system size ({bc})",
                   os.path.join(FIGURES_DIR, f"fig_nsectors_{bc}.pdf"))
        fig_series(bc, "d_max", r"$\ln D_{\max}$",
                   f"Largest sector vs system size ({bc})",
                   os.path.join(FIGURES_DIR, f"fig_dmax_{bc}.pdf"))
        fig_growth_scatter(bc, os.path.join(FIGURES_DIR, f"fig_growth_scatter_{bc}.pdf"))
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
