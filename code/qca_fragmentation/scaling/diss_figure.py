"""
Tier-1c figures for the DISSIPATIVE sweep (PLAN.md T6/T7).

  fig_diss_growth_scatter_{bc}.pdf        plain growth-rate map, no annotation
  fig_diss_growth_scatter_annot_{bc}.pdf  the same points, annotated by
                                          attractor type / coherence / parity
  fig_diss_bases_{bc}.pdf                 histogram of the EXACT growth bases

Both scatters plot the same quantity -- one point per dissipative rule at
(kappa of the attractor count, kappa of the largest attractor) -- so the plain
version can be read without a key and the annotated version answers "which
kinds of rule sit where".  Rules whose series is irregular (no growth law even
after the parity split, see scaling/dissipative.py) carry no rate and are
therefore absent from both; their count is stated in the caption line.

Design follows the same conventions as figure.py: CVD-validated categorical
hues, marker shape as a redundant encoding, recessive grid, direct labels.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from .dissipative import summarize
from .figure import MUTED, TEXT, _style

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

# Attractor structure at the largest computed N -> fixed hue + marker.
CLASS_STYLE = {
    "unique-point":   ("#8a8a86", "o", "unique fixed point"),
    "multistable":    ("#2a78d6", "s", "multistable (point attractors)"),
    "extended":       ("#eda100", "^", "one extended attractor"),
    "multi-extended": ("#e34948", "D", "several extended attractors"),
}
PLAIN = "#2a78d6"

# Rules worth naming on the annotated map: the ground-truth cases, the extremes,
# and the wall-Hadamard core that R5 discusses.
LABEL_RULES = [0, 4, 22, 28, 200, 232, 3, 19, 35, 85, 5, 1]


def _points(rows: List[Dict], bc: str):
    """(kappa_#att, kappa_Dmax, row) for every rule with both rates defined."""
    out = []
    for r in rows:
        if r["bc"] != bc:
            continue
        kn, kd = r["n_recurrent_kappa"], r["d_max_kappa"]
        if kn is None or kd is None:
            continue
        out.append((kn, kd, r))
    return out


_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def _jitter(pts, unit: float = 0.010):
    """Spread coincident points over a filled disc (phyllotaxis).

    Dissipative rules pile up on exact lattice points -- some 90 of them sit at
    kappa = (0, 0) -- so they must be spread to be counted at all.  A ring would
    draw a circle the data does not have, so the offsets fill the disc evenly:
    the j-th of n points goes to radius ~ sqrt(j/n) at the golden angle.
    """
    groups: Dict = {}
    for i, (x, y, _r) in enumerate(pts):
        groups.setdefault((round(x, 4), round(y, 4)), []).append(i)
    dx = np.zeros(len(pts))
    dy = np.zeros(len(pts))
    for idxs in groups.values():
        n = len(idxs)
        if n == 1:
            continue
        rad = unit * np.sqrt(n)
        for j, i in enumerate(idxs):
            rr = rad * np.sqrt((j + 0.5) / n)
            a = (j + 1) * _GOLDEN_ANGLE
            dx[i] = rr * np.cos(a)
            dy[i] = rr * np.sin(a)
    return dx, dy


# Candidate label offsets (points), tried in order of preference.
_CAND = [(6, 4), (6, -11), (-30, 4), (-30, -11),
         (6, 14), (6, -21), (-30, 14), (-30, -21)]


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def _place_labels(ax, fig, pts, dx, dy, want, blocked=()):
    """Greedy direct labels: pick, per point, the offset that collides least.

    Labels are only useful if they can be read, so each candidate offset is
    scored against the boxes already placed, the marker cloud, and any
    `blocked` regions (the legends) -- all in display coordinates.
    """
    fig.canvas.draw()
    xy = np.array([[x + ox, y + oy] for (x, y, _), ox, oy in zip(pts, dx, dy)])
    disp = ax.transData.transform(xy)
    placed = list(blocked)
    for i, (_x, _y, r) in enumerate(pts):
        if r["rule"] not in want:
            continue
        txt = f"W{r['rule']}"
        w, h = 5.2 * len(txt), 11.0
        px, py = disp[i]
        best = None
        for ox, oy in _CAND:
            box = (px + ox, py + oy - 2, px + ox + w, py + oy + h)
            pen = sum(_overlap(box, b) for b in placed) * 0.05
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            pen += 40.0 * int(np.sum(np.hypot(disp[:, 0] - cx,
                                              disp[:, 1] - cy) < 11.0))
            if best is None or pen < best[0]:
                best = (pen, ox, oy, box)
        placed.append(best[3])
        ax.annotate(txt, xy[i], textcoords="offset points",
                    xytext=(best[1], best[2]), fontsize=7.5, color=TEXT,
                    zorder=5)


def _legend_box(fig, leg):
    fig.canvas.draw()
    bb = leg.get_window_extent()
    return (bb.x0 - 4, bb.y0 - 4, bb.x1 + 4, bb.y1 + 4)


# The dissipative rules split cleanly into two families: V-free (classical --
# deterministic on basis states, so an ordinary elementary-CA functional graph)
# and V-carrying (mixed -- genuinely non-classical, the graph need not be
# faithful).  The plain scatter colours by this split, because it is what makes
# the algebraic growth bases (un)surprising: the classical family is textbook
# transfer-matrix combinatorics (R5, "classical baseline").
FAMILY_COLOUR = {"classical": "#1a9e5a", "mixed": "#e34948"}


def _family(r: Dict) -> str:
    return "classical" if "V" not in r["tuple"] else "mixed"


def fig_scatter_plain(rows: List[Dict], bc: str, out: str):
    pts = _points(rows, bc)
    dx, dy = _jitter(pts)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    _style(ax)
    seen = set()
    for (x, y, r), ddx, ddy in zip(pts, dx, dy):
        fam = _family(r)
        lbl = {"classical": "classical (V-free)",
               "mixed": "mixed (V + reset)"}[fam]
        ax.scatter(x + ddx, y + ddy, s=34, color=FAMILY_COLOUR[fam],
                   alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3,
                   label=lbl if lbl not in seen else None)
        seen.add(lbl)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$\kappa$ of the attractor count")
    ax.set_ylabel(r"$\kappa$ of the largest attractor $D_{\max}$")
    ax.set_title(f"Dissipative growth-rate map ({bc}, {len(pts)} rules)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(pts)


def fig_scatter_annotated(rows: List[Dict], bc: str, out: str):
    pts = _points(rows, bc)
    dx, dy = _jitter(pts)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    _style(ax)

    seen = set()
    for (x, y, r), ox, oy in zip(pts, dx, dy):
        color, marker, label = CLASS_STYLE.get(r["attractor_class"],
                                               ("#000", "o", "other"))
        coh = bool(r.get("coherent_attractor"))
        ax.scatter(x + ox, y + oy, s=52 if coh else 38, marker=marker,
                   facecolor="none" if coh else color,
                   edgecolor=color if coh else "white",
                   linewidth=1.4 if coh else 0.5,
                   alpha=0.95 if coh else 0.8, zorder=4 if coh else 3,
                   label=label if label not in seen else None)
        seen.add(label)

    # period-split rules get a light halo (their residue classes mod p obey
    # different laws); p is usually 2 (Neel) but 3 and 4 also occur
    for (x, y, r), ox, oy in zip(pts, dx, dy):
        if int(r["n_recurrent_period"] or 1) > 1 or int(r["d_max_period"] or 1) > 1:
            ax.scatter(x + ox, y + oy, s=118, marker="o", facecolor="none",
                       edgecolor="#4a3aa7", linewidth=0.7, alpha=0.55, zorder=2)

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$\kappa$ of the attractor count")
    ax.set_ylabel(r"$\kappa$ of the largest attractor $D_{\max}$")
    ax.set_title(f"Dissipative growth-rate map, annotated ({bc})",
                 fontsize=11, loc="left")

    # Legends go where the cloud is not: the exponential corner is empty, the
    # kappa ~ 0 corner (upper left) is where the labelled rules live.
    h, l = ax.get_legend_handles_labels()
    order = [l.index(v[2]) for v in CLASS_STYLE.values() if v[2] in l]
    leg = ax.legend([h[i] for i in order], [l[i] for i in order],
                    frameon=False, fontsize=8, loc="upper right",
                    title="attractor structure at $N_{\\max}$",
                    title_fontsize=8)
    leg._legend_box.align = "left"

    extra = [
        plt.Line2D([], [], marker="o", ls="none", mfc="none", mec=TEXT,
                   mew=1.4, ms=7, label="attractor not basis-spanned"),
        plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#4a3aa7",
                   mew=0.7, ms=10, label="even/odd $N$ split"),
    ]
    ax.add_artist(leg)
    leg2 = ax.legend(handles=extra, frameon=False, fontsize=8,
                     loc="center right")

    blocked = [_legend_box(fig, leg), _legend_box(fig, leg2)]
    _place_labels(ax, fig, pts, dx, dy, set(LABEL_RULES), blocked)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(pts)


def fig_bases(rows: List[Dict], bc: str, out: str):
    """The exact algebraic bases, which cluster on a handful of constants."""
    from .fits import name_base
    vals: Dict[float, int] = {}
    for r in rows:
        if r["bc"] != bc:
            continue
        for key in ("n_recurrent", "d_max"):
            b = r[f"{key}_base_exact"]
            if b and r[f"{key}_growth"] == "exponential":
                vals[round(b, 6)] = vals.get(round(b, 6), 0) + 1
    if not vals:
        return 0
    items = sorted(vals.items())
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    ax.bar([f"{b:.4f}" for b, _ in items], [c for _, c in items],
           color=PLAIN, width=0.6)
    for i, (b, c) in enumerate(items):
        nm = name_base(b)
        if nm:
            ax.annotate(nm.replace("$", "").replace("\\varphi", "phi")
                        .replace("\\sqrt", "sqrt").replace("\\psi", "psi")
                        .replace("\\rho", "rho"),
                        (i, c), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7.5, color=TEXT)
    ax.set_xlabel("exact growth base (largest root of the integer recurrence)")
    ax.set_ylabel("# series")
    ax.set_title(f"Exact dissipative growth bases ({bc})", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return sum(vals.values())


def main(argv=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = summarize()
    for bc in ("pbc", "obc0"):
        n = len([r for r in rows if r["bc"] == bc])
        p = fig_scatter_plain(rows, bc,
                              os.path.join(FIGURES_DIR,
                                           f"fig_diss_growth_scatter_{bc}.pdf"))
        fig_scatter_annotated(rows, bc,
                              os.path.join(FIGURES_DIR,
                                           f"fig_diss_growth_scatter_annot_{bc}.pdf"))
        nb = fig_bases(rows, bc,
                       os.path.join(FIGURES_DIR, f"fig_diss_bases_{bc}.pdf"))
        print(f"{bc}: plotted {p}/{n} rules ({n - p} irregular, no growth law); "
              f"{nb} series with an exact base")
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
