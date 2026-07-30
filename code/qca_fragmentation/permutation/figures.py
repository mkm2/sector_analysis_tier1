"""
Figures for R10 (X-gate / permutation circuits).

X1  the two maps side by side.  Their x-axis is IDENTICAL by theorem
    (n_recurrent = n_wcc), so the panels differ only in what goes up: the
    largest sector against the longest cycle.
X2  the drop, per rule: b_wcc -> b_recurrent, which is the same information
    drawn as a difference.
X3  transient structure: cyclic fraction and transient depth against N.
X4  validation: the sum rule, the identity n_rec = n_wcc, the finite-N
    hyperbola.
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


if __name__ == "__main__":
    main()
