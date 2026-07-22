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
from .fits import find_integer_recurrence, fit_pure_exponential, fit_series
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
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# The growth BASE, not the rate kappa = ln(base), is the informative coordinate.
# Every series here is bounded by the Hilbert-space dimension, so the base lives
# in [1, 2] -- a scale on which the interesting values are recognisable numbers
# (the golden ratio, the plastic number, 4^{1/5}) rather than the anonymous
# kappa in [0, ln 2].  Marking them turns the axis into a reference the reader
# can read a rule's mechanism off directly.
BASE_LIM = (0.98, 2.04)

# Constants that actually occur in the unitary sweep; the dissipative map passes
# its own list (there are many more there).  Kept short on purpose: one faint
# line per constant is a reference, sixteen is a grid.
UNITARY_REF = [(4 ** 0.2, r"$4^{1/5}$"),
               (1.6180339887, r"$\varphi$"),
               (1.7548776662, r"$x^3\!=\!2x^2\!-\!x\!+\!1$"),
               (2.0, r"$2$")]


def _base_of(Ns, ys):
    """Growth base of a series, preferring the exact algebraic value.

    NOT `fit_series`'s kappa: that comes from the M2 model c + alpha ln N +
    kappa N, whose ln N term absorbs part of the growth and biases the base
    downwards -- it put W156/198 at 1.281, well below its true asymptote
    4^{1/5} = 1.3195 (QCA_Circuits.pdf App. B).  The 2-parameter fit is
    unbiased for this purpose, and an exact recurrence beats both.
    Returns (base, exact?).
    """
    rec = find_integer_recurrence(ys)
    if rec["ok"] and rec["base"] > 1.0:
        return rec["base"], True
    f = fit_pure_exponential(Ns, ys)
    return (f["base"] if f.get("ok") else 1.0), False


def base_reference(ax, refs, axis="both"):
    """Faint lines at named growth bases, labelled on the top/right spines."""
    for v, name in refs:
        if axis in ("x", "both"):
            ax.axvline(v, color=MUTED, lw=0.6, ls=":", zorder=1, alpha=0.7)
            # rotated, inside the axes at the bottom: the names are long enough
            # ("x^3=2x^2-x+1") that a horizontal label along the top collides
            # with both the title and its neighbours
            ax.annotate(name, (v, 0.0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(-3, 6),
                        rotation=90, ha="right", va="bottom",
                        fontsize=7, color=MUTED)
        if axis in ("y", "both"):
            ax.axhline(v, color=MUTED, lw=0.6, ls=":", zorder=1, alpha=0.7)
            ax.annotate(name, (1.0, v), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=7.5, color=MUTED)


def fig_growth_scatter(bc: str, out: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    _style(ax)
    marker_for = {"exponential": "o", "polynomial": "s", "constant": "^"}
    color_for = {"exponential": "#e34948", "polynomial": "#2a78d6",
                 "constant": "#8a8a86"}
    base_reference(ax, UNITARY_REF)
    seen = set()
    pts = []
    for rule in rules.UNITARY_RULES:
        s = load_series(rule, bc)
        if len(s["N"]) < 3:
            continue
        fd = fit_series(s["N"], s["d_max"])
        if not fd.get("ok"):
            continue
        bn, en = _base_of(s["N"], s["n_recurrent"])
        bd, ed = _base_of(s["N"], s["d_max"])
        gc = fd["growth_class"]
        ax.scatter(bn, bd, s=70, color=color_for.get(gc, "#000"),
                   marker=marker_for.get(gc, "o"), edgecolor="white",
                   linewidth=0.8, zorder=3,
                   label=gc if gc not in seen else None)
        seen.add(gc)
        pts.append((bn, bd, rule, en and ed))

    # Direct labels.  Several rules land on the SAME point -- 156/198 are
    # reflection partners, 60/102 share the domain-wall law, 201/108 are
    # spin-flip partners -- so labels are clustered and stacked rather than
    # offset individually, which just piles them on top of each other.
    clusters: List[list] = []
    for bn, bd, rule, exact in sorted(pts, key=lambda p: (-p[1], p[0])):
        for cl in clusters:
            if abs(cl[0][0] - bn) < 0.04 and abs(cl[0][1] - bd) < 0.04:
                cl.append((bn, bd, rule, exact))
                break
        else:
            clusters.append([(bn, bd, rule, exact)])
    for cl in clusters:
        bn, bd = cl[0][0], cl[0][1]
        for k, (_, _, rule, exact) in enumerate(cl):
            ax.annotate(f"W{rule}" + ("" if exact else r"$^\dagger$"), (bn, bd),
                        textcoords="offset points", xytext=(7, 4 - 11 * k),
                        fontsize=7.5, color=TEXT, zorder=5)
    ax.set_xlim(*BASE_LIM)
    ax.set_ylim(*BASE_LIM)
    ax.set_xlabel(r"growth base of $\#$sectors")
    ax.set_ylabel(r"growth base of $D_{\max}$")
    ax.set_title(f"Fragmentation growth-base map ({bc})", fontsize=11, loc="left", pad=16)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, title="$D_{\\max}$ growth",
                  loc="lower right")
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
