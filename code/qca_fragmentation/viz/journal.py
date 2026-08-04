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


def off_axis(bc: str = "obc0", data: Optional[Dict] = None):
    """
    The rules whose move is NOT vertical, i.e. whose sector count and attractor
    count have different growth bases.

    These are not an arbitrary handful.  One attractor per sector forces
    n_scc = n_wcc and hence a_att = a, so a horizontal component exists ONLY for
    a rule with a multi-attractor sector -- R9 sec.7.3's twelve.  Ten of the
    twelve appear here; 235 and 249 are excluded because their ratio is a
    constant 2, which cancels out of a growth base and leaves both at 1.
    """
    d = data or sectors.load(bc) or sectors.build(bc)
    pts = {p["rule"]: p for p in d["points"]}
    out = []
    for x in d["attractor_deficits"]:
        p = pts.get(x["rule"])
        if not p or p["n_wcc"]["base"] is None:
            continue
        a, b = p["n_wcc"]["base"], p["d_max_wcc"]["base"]
        if abs(a - x["a_att"]) > 1e-6:
            out.append({"rule": x["rule"], "family": p["family"],
                        "a": a, "b": b, "aa": x["a_att"], "bb": x["b_att"]})
    return sorted(out, key=lambda r: r["rule"])


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
          "inset": lambda bc, d: style_inset(bc, data=d),
          # the two refined candidates -- see R14
          "overlay2": lambda bc, d: style_overlay2(bc, data=d),
          "stack": lambda bc, d: style_stack(bc, data=d)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R14 journal-figure candidates")
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = sectors.load(a.bc) or sectors.build(a.bc)
    for name, fn in STYLES.items():
        fn(a.bc, d)




# --- refined A: overlay, exceptions only --------------------------------------

OFF = "#d95f0e"


def style_overlay2(bc="obc0", out=None, data=None, label=True):
    """
    Overlay, second pass.

    The first version drew all 230 arrows and the 220 vertical ones swamped the
    ten that are not.  Since verticality is forced -- one attractor per sector
    means n_scc = n_wcc, so a_att = a -- those 220 arrows carry no information
    beyond "this rule moved", which the paired markers already say.  So they are
    reduced to a hairline stem and only the EXCEPTIONS are drawn as arrows: the
    ten rules with a multi-attractor sector, which are the only ones that can
    move sideways at all.
    """
    mv = moves(bc, data)
    ex = off_axis(bc, data)
    exset = {(round(r["a"], 3), round(r["b"], 3),
              round(r["aa"], 3), round(r["bb"], 3)) for r in ex}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _style(ax, xlabel=r"base of the count  ($a$, or $a_{\rm att}$)",
           ylabel=r"base of the largest block  ($b$, or $b_{\rm att}$)")

    # the forced vertical moves: a hairline, no head
    for a, b, aa, bb, f, n in mv:
        if abs(a - aa) < 1e-9 and abs(b - bb) < 1e-9:
            continue
        if (a, b, aa, bb) in exset:
            continue
        ax.plot([a, aa], [b, bb], color="#c3c7cd", lw=0.5, zorder=1,
                solid_capstyle="round")
    for (x, y, f), n in _wcc_cells(mv).items():
        ax.scatter(x, y, s=12 + 10 * np.sqrt(n), facecolor="none",
                   edgecolor=FAM[f], linewidth=0.9, alpha=0.8, zorder=3)
    _scatter(ax, _att_cells(mv), base=12.0, per=10.0)

    # the exceptions
    for r in ex:
        ax.annotate("", xy=(r["aa"], r["bb"]), xytext=(r["a"], r["b"]),
                    arrowprops=dict(arrowstyle="-|>", color=OFF, linewidth=1.5,
                                    alpha=0.95, shrinkA=3, shrinkB=3,
                                    mutation_scale=10), zorder=8)
    seen: Dict[Tuple[float, float], List[int]] = {}
    for r in ex:
        seen.setdefault((round(r["aa"], 3), round(r["bb"], 3)), []).append(r["rule"])
    if label:
        # All three arrow tips land in 1.32 < x < 1.47, so labels placed next to
        # them either collide or attach to the wrong arrow.  The band
        # 1.03 < x < 1.30, 1.0 < y < 1.35 is empty apart from the stem at x = 1,
        # so the labels are stacked there and joined to their tips by leaders.
        anchors = [(1.055, 1.30), (1.055, 1.17), (1.055, 1.045)]
        for (x, y), rs in sorted(seen.items(), key=lambda kv: -kv[0][1]):
            ax_, ay = anchors.pop(0)
            v = sorted(rs)
            txt = "W" + ", ".join(str(k) for k in v)
            if len(v) > 3:
                txt = ("W" + ", ".join(str(k) for k in v[:3]) + ",\n"
                       + ", ".join(str(k) for k in v[3:]))
            ax.annotate(txt, xy=(x, y), xytext=(ax_, ay), ha="left",
                        va="center", fontsize=6.8, color=OFF,
                        fontweight="bold", zorder=10,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=OFF,
                                        alpha=0.75, shrinkA=2, shrinkB=4))
    extra = [Line2D([], [], marker="o", ls="", markersize=5.5,
                    markerfacecolor="none", markeredgecolor="#444",
                    label="sectors (WCC)"),
             Line2D([], [], marker="o", ls="", markersize=5.5,
                    markerfacecolor="#444", markeredgecolor="white",
                    label="attractors (SCC)"),
             Line2D([], [], color=OFF, lw=1.5,
                    label=f"multi-attractor rules ({len(ex)})")]
    _legend(ax, loc="upper right", extra=extra, fontsize=6.9)
    _save(fig, out, bc, "overlay2")
    return ex


# --- refined D: stacked, one x-axis, split y ----------------------------------

def style_stack(bc="obc0", out=None, data=None):
    """
    Two panels stacked on ONE shared x-axis with a split y-axis.

    The x-coordinate means the same thing in both planes -- the growth base of
    the count -- so it is drawn once, at the bottom.  The y-axes are separate
    and the gap between them is left completely empty: no spine, no ticks, no
    titles, so an arrow from the upper panel to the lower one never crosses a
    line or a label.  Because a_att = a for every rule but ten, almost every
    arrow is then exactly vertical, and the ten that lean are the
    multi-attractor rules.
    """
    mv = moves(bc, data)
    ex = off_axis(bc, data)
    exset = {(round(r["a"], 3), round(r["b"], 3),
              round(r["aa"], 3), round(r["bb"], 3)) for r in ex}

    fig, (axU, axL) = plt.subplots(
        2, 1, figsize=(5.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30})

    for ax in (axU, axL):
        ax.grid(True, color="#ececea", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(*LIM)
        ax.set_ylim(0.93, 2.11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axU.spines["bottom"].set_visible(False)          # keep the gap empty
    axU.tick_params(axis="x", which="both", length=0, labelbottom=False)
    axL.spines["top"].set_visible(False)

    a = np.linspace(*LIM, 400)
    for ax in (axU, axL):
        ax.plot(a, 2.0 / a, color="#555555", lw=0.9, ls=":", zorder=2)
    _scatter(axU, _wcc_cells(mv), base=12.0, per=10.0)
    _scatter(axL, _att_cells(mv), base=12.0, per=10.0)

    for a0, b0, aa, bb, f, n in mv:
        if abs(a0 - aa) < 1e-9 and abs(b0 - bb) < 1e-9:
            continue
        isex = (a0, b0, aa, bb) in exset
        con = ConnectionPatch(
            xyA=(a0, b0), coordsA=axU.transData,
            xyB=(aa, bb), coordsB=axL.transData,
            arrowstyle="-|>", mutation_scale=8 if isex else 5,
            linewidth=1.4 if isex else 0.30 + 0.22 * np.sqrt(n),
            color=OFF if isex else ARROW, alpha=0.95 if isex else 0.30,
            zorder=9 if isex else 1, shrinkA=2.5, shrinkB=2.5)
        fig.add_artist(con)

    # broken-axis marks on the left spine, so the split y is explicit
    kw = dict(transform=fig.transFigure, color="#555", lw=0.9,
              clip_on=False, zorder=20)
    x0 = axU.get_position().x0
    yU, yL = axU.get_position().y0, axL.get_position().y1
    for yy in (yU, yL):
        fig.lines.append(plt.Line2D([x0 - 0.012, x0 + 0.012],
                                    [yy - 0.008, yy + 0.008], **kw))

    axU.set_ylabel(r"$b$   (sectors)", fontsize=9)
    axL.set_ylabel(r"$b_{\rm att}$   (attractors)", fontsize=9)
    axL.set_xlabel(r"base of the count:  $a$ above,  $a_{\rm att}$ below",
                   fontsize=9)
    for ax in (axU, axL):
        ax.tick_params(labelsize=7.5, length=2)
    extra = [Line2D([], [], color=OFF, lw=1.5,
                    label=f"multi-attractor rules ({len(ex)})")]
    # the arrows sweep down-right from (1,2); the upper panel's right side is
    # the only region none of them crosses
    _legend(axU, loc="upper right", extra=extra, fontsize=6.6)
    _save(fig, out, bc, "stack")
    return ex


if __name__ == "__main__":
    main()
