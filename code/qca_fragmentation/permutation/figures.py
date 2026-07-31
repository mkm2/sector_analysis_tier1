"""
Figures for R10 (X-gate / permutation circuits).

X1  the two maps side by side.  Their x-axis is IDENTICAL by theorem
    (n_recurrent = n_wcc), so the panels differ only in what goes up: the
    largest sector against the longest cycle.
X2  the drop, per rule: b_wcc -> b_recurrent, which is the same information
    drawn as a difference.
X3  transient structure: cyclic fraction and transient depth against N.
X4  movement WCC -> recurrent: the same rule in the two maps, joined by an
    arrow.  The arrows are vertical, and that is a theorem, not a coincidence.
X5  parent movement, sector plane: the 16 clusters (parent = resets off).
X6  parent movement, cycle plane: the same clusters, same x-coordinates.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from .. import results_io                 # noqa: E402
from ..core import rules                  # noqa: E402
from . import analysis                    # noqa: E402

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEXT = "#333333"
EMPH = {
    "reversible": dict(color="#1f4e9c", z=6, alpha=0.95, dx=0.0),
    "V+reset":    dict(color="#c62828", z=5, alpha=0.85, dx=0.019),
    "V-free":     dict(color="#9bbf9b", z=3, alpha=0.6, dx=-0.019),
}
LABEL = {"reversible": "reversible (I/V only), 16",
         "V+reset": "V + reset, 160", "V-free": "V-free, 80"}


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _hyperbola(ax, lo=0.90, hi=2.12):
    a = np.linspace(lo, hi, 400)
    ax.plot(a, 2.0 / a, color="#444444", lw=1.0, ls=":", zorder=2)


def _cells(pts, ykey):
    c: Counter = Counter()
    for p in pts:
        a, b = p["n_wcc"]["base"], p[ykey]["base"]
        if a is None or b is None:
            continue
        c[(round(a, 3), round(b, 3), p["family"])] += 1
    return c


def _scatter(ax, cells, annotate_above=10):
    labels = []
    for fam in ("V-free", "V+reset", "reversible"):
        st = EMPH[fam]
        for (x, y, f), n in cells.items():
            if f != fam:
                continue
            r = 26 + 15 * np.sqrt(n)
            ax.scatter(x + st["dx"], y, s=r, facecolor=st["color"],
                       edgecolor="white", lw=1.0, alpha=st["alpha"],
                       zorder=st["z"])
            if n >= annotate_above:
                labels.append((x + st["dx"], y, n, st["color"], r))
    for x, y, n, col, r in labels:
        inside = r > 110
        ax.annotate(f"{n}", (x, y), fontsize=6.6, ha="center",
                    va="center" if inside else "bottom",
                    xytext=(0, 0) if inside else (0, 7),
                    textcoords="offset points",
                    color="white" if inside else col,
                    fontweight="bold", zorder=20)


def _label_above(ax, d, ykey, eps=1e-9):
    """
    Name the rules that sit strictly ABOVE $ab=2$, cell by cell.

    Only meaningful in the cycle panel.  There the curve constrains nothing, so
    a point above it is a rule with exponentially many attractors AND an
    exponentially long one; the reader should be able to see which.  Rules
    exactly ON the curve are left unlabelled -- there are ten of them in one
    cell and they are listed in the text instead.
    """
    from . import movement as mvmod
    cells: Dict = {}
    for r in mvmod.above_curve(d, "cycle" if ykey == "d_max_recurrent"
                               else "sector"):
        if r["on_curve"]:
            continue
        st = EMPH[r["family"]]
        cells.setdefault((round(r["a"] + st["dx"], 3), round(r["b"], 3),
                          st["color"]), []).append(r["rule"])
    placed: List = []
    for (x, y, col), rs in sorted(cells.items()):
        # 108/201 and 73/109 sit within 0.015 of each other in both coordinates,
        # so a fixed offset overprints them; nudge the second of a close pair.
        dy = -3
        while any(abs(x - px) < 0.06 and abs((y + dy * 0.0016) - py) < 0.022
                  for px, py in placed):
            dy = -3 - 11 if dy == -3 else dy - 11
        right = x > 1.92          # keep the rightmost cells inside the axes
        if right:
            dy = 9                # ... and clear of the label to their left
        placed.append((x, y + dy * 0.0016))
        ax.annotate(", ".join(str(v) for v in sorted(rs)), (x, y), fontsize=7,
                    color=col, fontweight="bold",
                    xytext=(-9 if right else 9, dy),
                    textcoords="offset points",
                    va="bottom" if right else "baseline",
                    ha="right" if right else "left", zorder=20)


def fig_maps(bc: str, out: str, d: Optional[Dict] = None):
    d = d or analysis.load(bc) or analysis.build(bc)
    pts = d["points"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), sharex=True, sharey=True)
    for ax, key, title in (
            (axes[0], "d_max_wcc",
             "sectors (WCC): base $a$ vs base of the largest SECTOR"),
            (axes[1], "d_max_recurrent",
             "attractors (cycles): same $a$, base of the longest CYCLE")):
        _style(ax)
        _hyperbola(ax)
        _scatter(ax, _cells(pts, key))
        if key == "d_max_recurrent":
            _label_above(ax, d, key)
        ax.set_xlabel(r"base $a$ of $n_{\rm wcc}=n_{\rm recurrent}$")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0.90, 2.12)
        ax.set_ylim(0.88, 2.14)
    axes[0].set_ylabel(r"base $b$")
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=EMPH[k]["color"], markeredgecolor="white",
                     label=LABEL[k]) for k in ("reversible", "V+reset", "V-free")]
    hs.append(plt.Line2D([], [], color="#444444", lw=1.0, ls=":",
                         label=r"$ab=2$ (binds the LEFT panel only)"))
    axes[0].legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.suptitle("X1  X-gate circuits: the two maps share their $x$-axis "
                 "exactly, because a functional graph has one cycle per weak "
                 f"component ({bc}).", fontsize=9, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_drop(bc: str, out: str, d: Optional[Dict] = None):
    d = d or analysis.load(bc) or analysis.build(bc)
    pts = [p for p in d["points"]
           if p["d_max_wcc"]["base"] is not None
           and p["d_max_recurrent"]["base"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    ax = axes[0]
    _style(ax)
    lim = [0.9, 2.12]
    ax.plot(lim, lim, color="#888888", ls="--", lw=0.8, zorder=1)
    cnt: Counter = Counter()
    for p in pts:
        cnt[(round(p["d_max_wcc"]["base"], 3),
             round(p["d_max_recurrent"]["base"], 3), p["family"])] += 1
    for (x, y, f), n in cnt.items():
        st = EMPH[f]
        ax.scatter(x, y, s=26 + 15 * np.sqrt(n), facecolor=st["color"],
                   edgecolor="white", lw=0.9, alpha=st["alpha"], zorder=st["z"])
    ax.set_xlabel(r"base of the largest sector, $b_{\rm wcc}$")
    ax.set_ylabel(r"base of the longest cycle, $b_{\rm rec}$")
    ax.set_title("X2  every rule sits on or below the diagonal:\n"
                 "a cycle cannot be longer than the sector holding it",
                 fontsize=9.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax = axes[1]
    _style(ax)
    for f in ("reversible", "V+reset", "V-free"):
        v = [p["cyclic_fraction_at_Nmax"] for p in d["points"]
             if p["family"] == f]
        if not v:
            continue
        ax.hist(v, bins=np.linspace(0, 1.02, 40), histtype="stepfilled",
                color=EMPH[f]["color"], alpha=0.6, zorder=EMPH[f]["z"],
                label=f"{LABEL[f]}  (median {np.median(v):.3f})")
    ax.set_yscale("log")
    ax.set_xlabel(r"cyclic fraction at $N_{\max}$  "
                  r"(recurrent states $/\,2^N$)")
    ax.set_ylabel("rules")
    ax.set_title("how much of the space is periodic", fontsize=9.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_transient(bc: str, out: str, sel=(22, 28, 90, 150, 156, 201, 204, 232)):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for ax in axes:
        _style(ax)
    for rule in sel:
        s = analysis.load_series(rule, bc, analysis.UNIFORM_N_CAP)
        if len(s["N"]) < 3:
            continue
        lab = f"X{rule}"
        axes[0].plot(s["N"], [1 - t for t in s["transient_fraction"]], "o-",
                     ms=3.5, lw=1.1, label=lab)
        axes[1].plot(s["N"], s["transient_depth"], "o-", ms=3.5, lw=1.1,
                     label=lab)
        axes[2].plot(s["N"], s["d_max_ratio"], "o-", ms=3.5, lw=1.1, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("cyclic fraction")
    axes[1].set_ylabel("transient depth")
    axes[2].set_ylabel(r"$D_{\max}^{\rm wcc}/2^N$")
    for ax, t in zip(axes, ("recurrent states, as a fraction",
                            "longest path to a cycle",
                            "largest sector, as a fraction")):
        ax.set_xlabel("$N$")
        ax.set_title(t, fontsize=9.5)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("X3  transient structure of the permutation circuits "
                 f"({bc}); the reversible rules sit at cyclic fraction 1 and "
                 "depth 0 by construction.",
                 fontsize=9, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_movement(bc: str, out: str, d: Optional[Dict] = None):
    """
    X4.  The move from the sector map to the cycle map, drawn as a movement in
    ONE plane rather than as two panels.  Every arrow is vertical because the
    horizontal coordinate is shared by F1; the reversible rules have no arrow at
    all, since for them the sector IS the cycle.
    """
    from . import movement as mv
    d = d or analysis.load(bc) or analysis.build(bc)
    rows = [m for m in mv.movement(d)
            if m["b_wcc"] is not None and m["b_rec"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    _style(ax)
    _hyperbola(ax)
    seen = set()
    for m in rows:
        st = EMPH[m["family"]]
        x = m["a"] + st["dx"]
        key = (round(x, 3), round(m["b_wcc"], 3), round(m["b_rec"], 3))
        if key in seen:
            continue
        seen.add(key)
        if m["drop"] > 1e-9:
            ax.annotate("", xy=(x, m["b_rec"]), xytext=(x, m["b_wcc"]),
                        arrowprops=dict(arrowstyle="-|>", color=st["color"],
                                        lw=0.9, alpha=0.5, shrinkA=2,
                                        shrinkB=2), zorder=st["z"])
    cells_w = _cells(rows_as_points(rows, "b_wcc"), "y")
    cells_r = _cells(rows_as_points(rows, "b_rec"), "y")
    for cells, face, lw in ((cells_w, None, 1.0), (cells_r, "white", 1.4)):
        for (x, y, f), n in cells.items():
            st = EMPH[f]
            ax.scatter(x + st["dx"], y, s=24 + 14 * np.sqrt(n),
                       facecolor=(face or st["color"]),
                       edgecolor=(st["color"] if face else "white"),
                       lw=lw, alpha=st["alpha"], zorder=st["z"] + 1)
    ax.set_xlabel(r"base $a$ of $n_{\rm wcc}=n_{\rm recurrent}$")
    ax.set_ylabel(r"base $b$")
    ax.set_xlim(0.90, 2.12)
    ax.set_ylim(0.88, 2.14)
    ax.set_title("X4a  filled marker: largest sector.  hollow: longest cycle.\n"
                 "every move is straight down, because $a$ is shared by (F1)",
                 fontsize=9.5)
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=EMPH[k]["color"], markeredgecolor="white",
                     label=LABEL[k]) for k in ("reversible", "V+reset", "V-free")]
    hs.append(plt.Line2D([], [], color="#444444", lw=1.0, ls=":",
                         label=r"$ab=2$ (sector map only)"))
    ax.legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)

    ax = axes[1]
    _style(ax)
    for f in ("reversible", "V+reset", "V-free"):
        v = sorted(m["drop"] for m in rows if m["family"] == f)
        if not v:
            continue
        ax.hist(v, bins=np.linspace(-0.02, 1.05, 44), histtype="stepfilled",
                color=EMPH[f]["color"], alpha=0.62, zorder=EMPH[f]["z"],
                label=f"{LABEL[f]}  (median {np.median(v):.3f})")
    ax.set_yscale("log")
    ax.set_xlabel(r"drop $b_{\rm wcc}-b_{\rm rec}$")
    ax.set_ylabel("rules")
    ax.set_title("X4b  how far the rule falls.  zero drop = the whole sector\n"
                 "is periodic; a drop of 1 = an exponential sector with a "
                 "bounded cycle", fontsize=9.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def rows_as_points(rows, ykey):
    """Adapt movement rows to the (a, y, family) shape _cells expects."""
    return [{"n_wcc": {"base": m["a"]}, "y": {"base": m[ykey]},
             "family": m["family"]} for m in rows]


def fig_parent_clusters(bc: str, out: str, plane: str = "cycle",
                        d: Optional[Dict] = None):
    """
    X5/X6.  One panel per reversible rule: where its descendants sit when the
    resets are switched on.  Star = the parent, dots = its children, arrows =
    the move.  In the cycle plane the parent's height is the ceiling: a cycle of
    a child is (for 177 of 240) literally an orbit of the parent.
    """
    from . import movement as mvmod
    d = d or analysis.load(bc) or analysis.build(bc)
    cl = mvmod.parent_clusters(d, plane)
    ncol, col = 4, ("#c62828" if plane == "cycle" else "#1f6f4c")
    nrow = int(np.ceil(len(cl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.15 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cl):
        _style(ax)
        _hyperbola(ax)
        pxy = c["parent_xy"]
        cells = Counter((round(x, 3), round(y, 3)) for _, (x, y) in c["children"])
        for (x, y), n in cells.items():
            if pxy is None or (abs(x - pxy[0]) < 1e-6 and abs(y - pxy[1]) < 1e-6):
                continue
            ax.annotate("", xy=(x, y), xytext=pxy,
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=0.5 + 0.5 * np.sqrt(n), alpha=0.5,
                                        shrinkA=7, shrinkB=3), zorder=4)
        for (x, y), n in cells.items():
            ax.scatter(x, y, s=22 + 15 * np.sqrt(n), facecolor=col,
                       edgecolor="white", lw=0.8, alpha=0.9, zorder=5)
            if n >= 4:
                ax.annotate(f"{n}", (x, y), fontsize=6.2, ha="center",
                            va="center", color="white", fontweight="bold",
                            zorder=6)
        stay = 0
        if pxy is not None:
            ax.scatter(*pxy, s=210, marker="*", facecolor="#1f4e9c",
                       edgecolor="white", lw=1.0, zorder=7)
            stay = sum(n for (x, y), n in cells.items()
                       if abs(x - pxy[0]) < 1e-6 and abs(y - pxy[1]) < 1e-6)
        ax.set_title(f"X{c['parent']}  {c['tuple']}   "
                     f"({c['n_children']} children, {stay} stay put)",
                     fontsize=8.5)
        ax.set_xlim(0.90, 2.12)
        ax.set_ylim(0.88, 2.14)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cl):]:
        ax.axis("off")
    for ax in axes[:len(cl)]:
        ax.set_xlabel(r"base $a$", fontsize=8)
        ax.set_ylabel(r"base $b_{\rm %s}$" % ("rec" if plane == "cycle" else "wcc"),
                      fontsize=8)
    what = ("the longest CYCLE" if plane == "cycle" else "the largest SECTOR")
    tag = "X6" if plane == "cycle" else "X5"
    hs = [plt.Line2D([], [], marker="*", ls="", markersize=12,
                     markerfacecolor="#1f4e9c", markeredgecolor="white",
                     label="the reversible parent (resets off)"),
          plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=col, markeredgecolor="white",
                     label="its children (resets on)"),
          plt.Line2D([], [], color="#444444", lw=1.0, ls=":", label=r"$ab=2$")]
    fig.legend(handles=hs, fontsize=8, loc="lower right", ncol=3,
               bbox_to_anchor=(0.99, 0.004))
    fig.suptitle(f"{tag}  parent movement in the plane of {what} ({bc}).  All "
                 f"$256$ rules grouped by coherent parent (D, E $\\to$ I); the "
                 f"identity's cluster is the whole V-free family.",
                 fontsize=9.5, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0.025, 1, 0.958))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R10 figures")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = analysis.build(a.bc) if a.rebuild else (
        analysis.load(a.bc) or analysis.build(a.bc))
    fig_maps(a.bc, os.path.join(FIGURES_DIR, f"fig_xgate_maps_{a.bc}.pdf"), d)
    fig_drop(a.bc, os.path.join(FIGURES_DIR, f"fig_xgate_drop_{a.bc}.pdf"), d)
    fig_transient(a.bc, os.path.join(FIGURES_DIR,
                                     f"fig_xgate_transient_{a.bc}.pdf"))
    fig_movement(a.bc, os.path.join(FIGURES_DIR,
                                    f"fig_xgate_movement_{a.bc}.pdf"), d)
    for plane in ("sector", "cycle"):
        fig_parent_clusters(a.bc, os.path.join(
            FIGURES_DIR, f"fig_xgate_parents_{plane}_{a.bc}.pdf"),
            plane=plane, d=d)


if __name__ == "__main__":
    main()
