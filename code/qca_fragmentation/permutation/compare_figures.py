"""
Figures for R11 (Hadamard vs X gate).

C1  the 16-rule baseline: where each unitary/reversible rule sits under each
    gate, joined by the move.  This is the only fully like-for-like comparison,
    because both families satisfy sectors = attractors on these rules.
C2  all 256 rules at once, in the same plane, with the V-free rules -- which are
    the SAME circuit under both gates -- drawn separately as the control.
C3  R9's headline sets under X: the eight open-system-fragmented rules, the six
    with linear sector counts, and the pinned-frontier family.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from .. import results_io                 # noqa: E402
from ..core import rules                  # noqa: E402
from . import compare                     # noqa: E402

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEXT = "#333333"
C_H = "#1f4e9c"      # Hadamard
C_X = "#c62828"      # X gate
C_SHARED = "#7d8a7d"  # rules with no V: the same circuit either way


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _hyperbola(ax, lo=0.90, hi=2.12):
    a = np.linspace(lo, hi, 400)
    ax.plot(a, 2.0 / a, color="#444444", lw=1.0, ls=":", zorder=2)


def _frame(ax):
    ax.set_xlim(0.90, 2.12)
    ax.set_ylim(0.88, 2.14)
    ax.set_xlabel(r"base $a$ of the sector count")
    ax.set_ylabel(r"base $b$ of the largest sector")


def fig_baseline(bc: str, out: str, d: Optional[Dict] = None):
    d = d or compare.load(bc) or compare.build(bc)
    rows = compare.baseline(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    ax = axes[0]
    _style(ax)
    _hyperbola(ax)
    for r in rows:
        h, x = (r["a_h"], r["b_h"]), (r["a_x"], r["b_x"])
        if abs(h[0] - x[0]) > 1e-6 or abs(h[1] - x[1]) > 1e-6:
            ax.annotate("", xy=x, xytext=h,
                        arrowprops=dict(arrowstyle="-|>", color="#999999",
                                        lw=0.9, alpha=0.75, shrinkA=5,
                                        shrinkB=5), zorder=3)
    for key, col, mk, lab in ((("a_h", "b_h"), C_H, "o", "Hadamard (R9)"),
                              (("a_x", "b_x"), C_X, "s", "$X$ gate (R10)")):
        cells = Counter((round(r[key[0]], 3), round(r[key[1]], 3)) for r in rows)
        for (px, py), n in cells.items():
            ax.scatter(px, py, s=40 + 26 * np.sqrt(n), marker=mk,
                       facecolor=col, edgecolor="white", lw=1.1, zorder=5,
                       label=lab if (px, py) == list(cells)[0] else None)
            if n > 1:
                ax.annotate(f"{n}", (px, py), fontsize=7, ha="center",
                            va="center", color="white", fontweight="bold",
                            zorder=6)
    # label a destination cell only when few enough rules share it; the marker
    # already carries the count, and overlapping labels are worse than none
    dest: Dict = {}
    for r in rows:
        dest.setdefault((round(r["a_x"], 3), round(r["b_x"], 3)), []).append(
            r["rule"])
    for (px, py), rs in dest.items():
        if len(rs) <= 4:
            ax.annotate(", ".join(str(v) for v in sorted(rs)), (px, py),
                        fontsize=7, color=C_X, xytext=(9, -3),
                        textcoords="offset points", zorder=7)
    _frame(ax)
    ax.set_title("C1a  the 16 rules that are unitary under $H$ and\n"
                 "reversible under $X$: same table, different gate",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

    ax = axes[1]
    _style(ax)
    lab = [f"{r['rule']}" for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["a_h"] for r in rows], width=0.4, color=C_H,
           label="$a$, Hadamard")
    ax.bar(xs + 0.2, [r["a_x"] for r in rows], width=0.4, color=C_X,
           label="$a$, $X$ gate")
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=7, rotation=90)
    ax.set_ylim(0.9, 2.15)
    ax.set_ylabel(r"base $a$ of the sector count")
    ax.set_title("C1b  the sector-count base, rule by rule.\n"
                 "$204$ is the only rule the gate does not move", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_all_rules(bc: str, out: str, d: Optional[Dict] = None):
    d = d or compare.load(bc) or compare.build(bc)
    rows = [r for r in d["rows"]
            if None not in (r["a_h"], r["b_h"], r["a_x"], r["b_x"])]
    withV = [r for r in rows if compare.has_V(r["rule"])]
    noV = [r for r in rows if not compare.has_V(r["rule"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    ax = axes[0]
    _style(ax)
    _hyperbola(ax)
    seen = set()
    for r in withV:
        h, x = (round(r["a_h"], 3), round(r["b_h"], 3)), \
               (round(r["a_x"], 3), round(r["b_x"], 3))
        if (h, x) in seen or h == x:
            continue
        seen.add((h, x))
        ax.annotate("", xy=x, xytext=h,
                    arrowprops=dict(arrowstyle="-|>", color=C_X, lw=0.7,
                                    alpha=0.4, shrinkA=3, shrinkB=3), zorder=3)
    # X markers first, Hadamard on top: almost every V rule STARTS at (1, 2),
    # so the origin would otherwise be buried under its own destination
    cells = Counter((round(p["a_x"], 3), round(p["b_x"], 3)) for p in withV)
    for (px, py), n in cells.items():
        ax.scatter(px, py, s=22 + 15 * np.sqrt(n), marker="s", facecolor=C_X,
                   edgecolor="white", lw=0.8, zorder=5, alpha=0.9)
    for pts, col, mk, zo in ((noV, C_SHARED, "D", 6), (withV, C_H, "o", 7)):
        cells = Counter((round(p["a_h"], 3), round(p["b_h"], 3)) for p in pts)
        for (px, py), n in cells.items():
            ax.scatter(px, py, s=16 + 11 * np.sqrt(n), marker=mk,
                       facecolor=col, edgecolor="white", lw=0.8, zorder=zo,
                       alpha=0.95)
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=C_H, markeredgecolor="white",
                     label="Hadamard (rules with a V)"),
          plt.Line2D([], [], marker="s", ls="", markersize=7,
                     markerfacecolor=C_X, markeredgecolor="white",
                     label="$X$ gate (same rules)"),
          plt.Line2D([], [], marker="D", ls="", markersize=6,
                     markerfacecolor=C_SHARED, markeredgecolor="white",
                     label="no V: one point, both gates"),
          plt.Line2D([], [], color="#444444", lw=1.0, ls=":", label=r"$ab=2$")]
    ax.legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)
    _frame(ax)
    ax.set_title("C2a  all $256$ rules, common window $N\\leq16$.  Arrows are "
                 "the gate change;\nthe $81$ V-free rules cannot move",
                 fontsize=9.5)

    ax = axes[1]
    _style(ax)
    ax.plot([0.9, 2.12], [0.9, 2.12], color="#888888", ls="--", lw=0.8)
    cells = Counter((round(r["a_h"], 3), round(r["a_x"], 3),
                     compare.has_V(r["rule"])) for r in rows)
    for (h, x, v), n in cells.items():
        ax.scatter(h, x, s=20 + 15 * np.sqrt(n),
                   facecolor=(C_X if v else C_SHARED), edgecolor="white",
                   lw=0.8, alpha=0.9, zorder=4 + int(v))
    co = d["correlation"].get("has_V_a", {})
    ax.set_xlabel(r"$a$ under the Hadamard")
    ax.set_ylabel(r"$a$ under $X$")
    ax.set_xlim(0.9, 2.12)
    ax.set_ylim(0.9, 2.12)
    pear = co.get("pearson")
    ax.set_title("C2b  sector-count base, gate against gate.  Nothing sits "
                 "below\nthe diagonal: refinement forbids it "
                 f"(Pearson $r={pear:+.2f}$ on the V rules)", fontsize=9.5)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_survivors(bc: str, out: str, d: Optional[Dict] = None):
    d = d or compare.load(bc) or compare.build(bc)
    sv = compare.survivors(d)
    titles = {
        "open_fragmented": "C3a  R9's open-system-fragmented rules",
        "linear_count": "C3b  R9's strictly linear sector counts",
        "frontier": "C3c  R9's pinned-frontier family",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7))
    for ax, (name, rowset) in zip(axes, sv.items()):
        _style(ax)
        _hyperbola(ax)
        rowset = [r for r in rowset
                  if None not in (r["a_h"], r["b_h"], r["a_x"], r["b_x"])]
        seen = set()
        for r in rowset:
            h = (round(r["a_h"], 3), round(r["b_h"], 3))
            x = (round(r["a_x"], 3), round(r["b_x"], 3))
            if h != x and (h, x) not in seen:
                seen.add((h, x))
                ax.annotate("", xy=x, xytext=h,
                            arrowprops=dict(arrowstyle="-|>", color="#999999",
                                            lw=1.0, alpha=0.8, shrinkA=6,
                                            shrinkB=6), zorder=3)
        # destinations first, origins on top and smaller: several rules share a
        # cell in both maps, and the Hadamard point is the one that gets buried
        cells: Dict = {}
        for r in rowset:
            cells.setdefault((round(r["a_x"], 3), round(r["b_x"], 3)),
                             []).append(r["rule"])
        for (px, py), rs in cells.items():
            ax.scatter(px, py, s=44 + 24 * np.sqrt(len(rs)), marker="s",
                       facecolor=C_X, edgecolor="white", lw=1.0, zorder=5)
            ax.annotate(", ".join(str(v) for v in sorted(rs)), (px, py),
                        fontsize=6.6, color=C_X, xytext=(9, -4),
                        textcoords="offset points", ha="left", zorder=7)
        hc: Dict = {}
        for r in rowset:
            hc.setdefault((round(r["a_h"], 3), round(r["b_h"], 3)),
                          []).append(r["rule"])
        for (px, py), rs in hc.items():
            ax.scatter(px, py, s=30 + 16 * np.sqrt(len(rs)), facecolor=C_H,
                       edgecolor="white", lw=1.0, zorder=8)
        _frame(ax)
        ax.set_title(titles[name], fontsize=9.5)
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=C_H, markeredgecolor="white",
                     label="Hadamard"),
          plt.Line2D([], [], marker="s", ls="", markersize=7,
                     markerfacecolor=C_X, markeredgecolor="white",
                     label="$X$ gate"),
          plt.Line2D([], [], color="#444444", lw=1.0, ls=":", label=r"$ab=2$")]
    axes[0].legend(handles=hs, fontsize=8, loc="lower left", framealpha=0.95)
    fig.suptitle("C3  what R9's headline sets do under the other gate "
                 f"({bc}, common window $N\\leq16$).", fontsize=9.5, x=0.01,
                 ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R11 figures")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = compare.build(a.bc) if a.rebuild else (
        compare.load(a.bc) or compare.build(a.bc))
    fig_baseline(a.bc, os.path.join(FIGURES_DIR, f"fig_cmp_baseline_{a.bc}.pdf"), d)
    fig_all_rules(a.bc, os.path.join(FIGURES_DIR, f"fig_cmp_all_{a.bc}.pdf"), d)
    fig_survivors(a.bc, os.path.join(FIGURES_DIR,
                                     f"fig_cmp_survivors_{a.bc}.pdf"), d)


if __name__ == "__main__":
    main()
