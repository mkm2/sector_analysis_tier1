"""
Journal-figure candidates: the WCC plane, the SCC plane, and the flow between
them, condensed from three separate figures into one.

R9 currently spends three figures on this --- \\fn{fig\\_sector\\_map} (two
panels), \\fn{fig\\_monitored\\_map} (one) and \\fn{fig\\_sector\\_vs\\_attractor}
(two) --- which is far too much space for one idea: every rule has a position in
the sector plane and a position in the attractor plane, and dissipation moves it
from the first to the second.  This module draws that idea four ways so the
layouts can be compared side by side.

Shared design rules, so the variants differ only in layout:
  * marker area grows with the number of rules stacked in a cell (77% of the
    sector map sits in one cell, so one marker per rule hides the distribution);
  * colour is the family;
  * arrows are LIGHT -- they carry the movement without competing with the
    points, and their width grows with how many rules make the same move;
  * a rule that does not move is drawn as a dot with no arrow.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import ConnectionPatch      # noqa: E402

from .. import results_io                           # noqa: E402
from ..scaling import sectors                       # noqa: E402

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

FAM = {"unitary": "#1f4e9c", "mixed": "#c62828", "classical": "#7aa87a"}
FAMLAB = {"unitary": "unitary (V only), 16",
          "mixed": "V + reset, 160",
          "classical": "V-free, 80"}
ORDER = ("classical", "mixed", "unitary")           # draw order = z order
ARROW = "#8a8f98"
MUTED = "#9a9a95"
LIM = (0.93, 2.11)


def moves(bc: str = "obc0", data: Optional[Dict] = None):
    """
    [(a_wcc, b_wcc, a_att, b_att, family, count)] with identical moves merged.
    """
    d = data or sectors.load(bc) or sectors.build(bc)
    pts = {p["rule"]: p for p in d["points"]}
    c: Counter = Counter()
    for x in d["attractor_deficits"]:
        p = pts.get(x["rule"])
        if not p or p["n_wcc"]["base"] is None:
            continue
        c[(round(p["n_wcc"]["base"], 3), round(p["d_max_wcc"]["base"], 3),
           round(x["a_att"], 3), round(x["b_att"], 3), p["family"])] += 1
    return [(k[0], k[1], k[2], k[3], k[4], n) for k, n in c.items()]


def _style(ax, *, xlabel=None, ylabel=None, curve=True, ticks=True):
    ax.grid(True, color="#ececea", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if curve:
        a = np.linspace(LIM[0], LIM[1], 400)
        ax.plot(a, 2.0 / a, color="#555555", lw=0.9, ls=":", zorder=2)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5, length=2)
    if not ticks:
        ax.set_xticklabels([]), ax.set_yticklabels([])


def _scatter(ax, cells: Counter, *, base=14.0, per=11.0, alpha=0.9):
    for fam in ORDER:
        for (x, y, f), n in cells.items():
            if f != fam:
                continue
            ax.scatter(x, y, s=base + per * np.sqrt(n), facecolor=FAM[fam],
                       edgecolor="white", linewidth=0.7, alpha=alpha,
                       zorder=4 + ORDER.index(fam))


def _legend(ax, loc="lower left", extra=(), fontsize=7.2):
    hs = [Line2D([], [], marker="o", ls="", markersize=6,
                 markerfacecolor=FAM[k], markeredgecolor="white",
                 label=FAMLAB[k]) for k in ("unitary", "mixed", "classical")]
    hs += list(extra)
    lg = ax.legend(handles=hs, fontsize=fontsize, loc=loc, framealpha=0.94,
                   borderpad=0.4, handletextpad=0.5, labelspacing=0.35)
    lg.get_frame().set_linewidth(0.5)
    return lg


def _wcc_cells(mv):
    c: Counter = Counter()
    for a, b, _, _, f, n in mv:
        c[(a, b, f)] += n
    return c


def _att_cells(mv):
    c: Counter = Counter()
    for _, _, a, b, f, n in mv:
        c[(a, b, f)] += n
    return c


# --- style A: twin panels, arrows across the gap ------------------------------

def style_twin(bc="obc0", out=None, orient="h", data=None):
    """
    Two panels with the flow drawn BETWEEN them as connection arrows.

    The literal reading of "sectors on the left, attractors on the right, and
    here is where each rule goes".  Costs a lot of horizontal space to the gap,
    which is the price of showing the movement without overplotting either
    plane.
    """
    mv = moves(bc, data)
    horiz = orient == "h"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6)) if horiz else \
        plt.subplots(2, 1, figsize=(4.6, 8.6))
    if horiz:
        fig.subplots_adjust(wspace=0.42)
    else:
        fig.subplots_adjust(hspace=0.30)
    axA, axB = axes

    _style(axA, xlabel=r"$a$  (base of $\#$sectors)",
           ylabel=r"$b$  (base of $D_{\max}$)")
    _style(axB, xlabel=r"$a_{\rm att}$", ylabel=r"$b_{\rm att}$")
    _scatter(axA, _wcc_cells(mv))
    _scatter(axB, _att_cells(mv))
    axA.set_title("sectors (WCC)", fontsize=9)
    axB.set_title("attractors (terminal SCC)", fontsize=9)

    for a, b, aa, bb, f, n in sorted(mv, key=lambda t: t[5]):
        if abs(a - aa) < 1e-9 and abs(b - bb) < 1e-9:
            continue
        con = ConnectionPatch(
            xyA=(a, b), coordsA=axA.transData,
            xyB=(aa, bb), coordsB=axB.transData,
            arrowstyle="-|>", mutation_scale=7,
            linewidth=0.28 + 0.30 * np.sqrt(n), color=ARROW,
            alpha=0.34, zorder=1, shrinkA=2.5, shrinkB=2.5)
        fig.add_artist(con)
    _legend(axA)
    for p in _save(fig, out, bc, "twin_" + orient):
        pass
    return mv


# --- style B: one plane, both positions, arrows inside ------------------------

def style_overlay(bc="obc0", out=None, data=None):
    """
    A single plane.  Each rule appears twice --- hollow where its sectors put
    it, filled where its attractors do --- joined by a light arrow.

    The most compact of the four, and the only one in which the two coordinate
    systems are literally the same axes, which is the honest way to show that
    the move is a displacement in one plane rather than a mapping between two
    different spaces.
    """
    mv = moves(bc, data)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    _style(ax, xlabel=r"base of the count  ($a$ or $a_{\rm att}$)",
           ylabel=r"base of the largest block  ($b$ or $b_{\rm att}$)")

    for a, b, aa, bb, f, n in sorted(mv, key=lambda t: t[5]):
        if abs(a - aa) < 1e-9 and abs(b - bb) < 1e-9:
            continue
        ax.annotate("", xy=(aa, bb), xytext=(a, b),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                    linewidth=0.28 + 0.30 * np.sqrt(n),
                                    alpha=0.42, shrinkA=2.0, shrinkB=2.0,
                                    mutation_scale=7), zorder=1)
    for (x, y, f), n in _wcc_cells(mv).items():          # hollow = sectors
        ax.scatter(x, y, s=14 + 11 * np.sqrt(n), facecolor="none",
                   edgecolor=FAM[f], linewidth=1.0, alpha=0.85, zorder=3)
    _scatter(ax, _att_cells(mv))                         # filled = attractors
    extra = [Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor="none",
                    markeredgecolor="#444", label="sectors (WCC)"),
             Line2D([], [], marker="o", ls="", markersize=6,
                    markerfacecolor="#444", markeredgecolor="white",
                    label="attractors (SCC)")]
    _legend(ax, loc="upper right", extra=extra)
    _save(fig, out, bc, "overlay")
    return mv


# --- style C: slopegraph of the product ---------------------------------------

def style_slope(bc="obc0", out=None, data=None):
    """
    Neither coordinate but the quantity the exclusion curve constrains: the
    product ab.  Sectors on the left axis, attractors on the right, one line per
    rule.

    Throws away the a/b split entirely and keeps only what the hyperbola is
    about, so it is the variant that makes the theorem visible: everything
    starts on or above 2 and almost everything ends below it.
    """
    mv = moves(bc, data)
    fig, ax = plt.subplots(figsize=(3.9, 5.0))
    ax.grid(True, axis="y", color="#ececea", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    seen: Counter = Counter()
    for a, b, aa, bb, f, n in mv:
        seen[(round(a * b, 3), round(aa * bb, 3), f)] += n
    for (p0, p1, f), n in sorted(seen.items(), key=lambda kv: kv[1]):
        ax.plot([0, 1], [p0, p1], color=FAM[f], alpha=0.5,
                linewidth=0.3 + 0.34 * np.sqrt(n), zorder=3, solid_capstyle="round")
        ax.scatter([0, 1], [p0, p1], s=8 + 7 * np.sqrt(n), color=FAM[f],
                   edgecolor="white", linewidth=0.5, zorder=4)
    ax.axhline(2.0, color="#555555", lw=0.9, ls=":", zorder=2)
    ax.annotate(r"$ab=2$", (-0.14, 2.0), fontsize=7.5, color="#555555",
                va="bottom", ha="left")
    ax.set_xlim(-0.16, 1.16)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sectors\n$ab$", "attractors\n$a_{\\rm att}b_{\\rm att}$"],
                       fontsize=8.5)
    ax.set_ylabel("product of the two bases", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5, length=2)
    ax.tick_params(axis="x", length=0)
    _legend(ax, loc="lower left", fontsize=6.8)
    _save(fig, out, bc, "slope")
    return mv


# --- style D: inset -----------------------------------------------------------

def style_inset(bc="obc0", out=None, data=None):
    """
    The sector plane full size, with the attractor plane as a small inset and
    the deficit distribution beneath it.

    Reflects the asymmetry of the two maps rather than pretending they are
    equals: the sector plane is exact for both experiments, the attractor plane
    is monitored-only, so it gets less space and a caveat rather than half the
    figure.
    """
    mv = moves(bc, data)
    d = sectors.load(bc) or sectors.build(bc)
    fam = {p["rule"]: p["family"] for p in d["points"]}
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_axes([0.10, 0.10, 0.62, 0.84])
    _style(ax, xlabel=r"$a$  (base of $\#$sectors)",
           ylabel=r"$b$  (base of $D_{\max}$)")
    _scatter(ax, _wcc_cells(mv))
    ax.set_title("sectors (WCC) --- exact for both experiments", fontsize=9)
    _legend(ax)

    ins = fig.add_axes([0.755, 0.560, 0.225, 0.30])
    _style(ins, ticks=False, curve=False)
    ins.set_aspect("auto")
    ins.plot(np.linspace(*LIM, 200), 2.0 / np.linspace(*LIM, 200),
             color="#555", lw=0.7, ls=":")
    _scatter(ins, _att_cells(mv), base=4.0, per=4.0, alpha=0.85)
    ins.set_title("attractors\n(monitored)", fontsize=7.2, pad=3)

    hist = fig.add_axes([0.755, 0.145, 0.225, 0.29])
    hist.grid(True, color="#ececea", linewidth=0.6)
    hist.set_axisbelow(True)
    for s in ("top", "right"):
        hist.spines[s].set_visible(False)
    for f in ORDER:
        v = [x["deficit"] for x in d["attractor_deficits"]
             if fam.get(x["rule"]) == f]
        if v:
            hist.hist(v, bins=np.linspace(-1.0, 1.05, 30),
                      histtype="stepfilled", color=FAM[f], alpha=0.55)
    hist.axvline(0.0, color="#555", ls=":", lw=0.8)
    hist.set_xlabel(r"deficit $2-a_{\rm att}b_{\rm att}$", fontsize=7)
    hist.tick_params(labelsize=6.2, length=2)
    hist.set_yticks([])
    _save(fig, out, bc, "inset")
    return mv


def _save(fig, out, bc, tag):
    out = out or os.path.join(FIGURES_DIR, f"fig_journal_{tag}_{bc}.pdf")
    for p in (out, out.replace(".pdf", ".png")):
        fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return [out]


STYLES = {"twin_h": lambda bc, d: style_twin(bc, orient="h", data=d),
          "twin_v": lambda bc, d: style_twin(bc, orient="v", data=d),
          "overlay": lambda bc, d: style_overlay(bc, data=d),
          "slope": lambda bc, d: style_slope(bc, data=d),
          "inset": lambda bc, d: style_inset(bc, data=d)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R14 journal-figure candidates")
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = sectors.load(a.bc) or sectors.build(a.bc)
    for name, fn in STYLES.items():
        fn(a.bc, d)


if __name__ == "__main__":
    main()
