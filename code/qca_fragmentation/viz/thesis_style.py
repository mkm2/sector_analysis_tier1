"""
Thesis figure standards (CLAUDE.md "Figure standards"; source of the numbers:
~/Dokumente/Projects/Figure Generation, thesisfig/config.py and style.theme()).

    from qca_fragmentation.viz import thesis_style as ts
    ts.use()                      # thesis mode (422.5 pt text width)
    ts.use("paper")               # single column, 246 pt -- only on request
    fig = ts.figure(aspect=0.6)   # full width, height = aspect * width

Figures are drawn at final print size; never bbox_inches="tight".
"""

from __future__ import annotations

import numpy as np
import matplotlib

TEXTWIDTH_PT = {"thesis": 422.52348, "paper": 246.0}
MAX_HEIGHT_PT = 508.4
PT = 1 / 72.27

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
GREY = "#7F7F7F"
INK = "#222222"

FS = {"annot": 10.0, "tick": 10.0, "legend": 10.0, "label": 10.95, "panel": 10.95}
LW = {"data": 1.0, "axes": 0.6, "grid": 0.4}
MS = 3.5
DOT = 1.05

_mode = "thesis"


def fill(c, f=0.24):
    from matplotlib.colors import to_rgb
    return tuple(1 - f * (1 - np.asarray(to_rgb(c))))


def dark(c, f=0.78):
    from matplotlib.colors import to_rgb
    return tuple(f * np.asarray(to_rgb(c)))


def use(mode: str = "thesis"):
    """Set rcParams for the thesis (default) or, on request, paper mode."""
    global _mode
    assert mode in TEXTWIDTH_PT
    _mode = mode
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
                               r"\usepackage{amsmath}",
        "font.size": FS["annot"],
        "axes.labelsize": FS["label"], "axes.titlesize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "axes.linewidth": LW["axes"], "lines.linewidth": LW["data"],
        "lines.markersize": MS,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": LW["axes"], "ytick.major.width": LW["axes"],
        "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "savefig.dpi": 600,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 2 * PT, "figure.constrained_layout.w_pad": 2 * PT,
        "pdf.fonttype": 42,
    })
    return plt


def width_in(frac: float = 1.0) -> float:
    return frac * TEXTWIDTH_PT[_mode] * PT


def figure(width: float = 1.0, aspect: float = 0.618, **kw):
    import matplotlib.pyplot as plt
    w = width_in(width)
    h = min(aspect * w, MAX_HEIGHT_PT * PT)
    return plt.figure(figsize=(w, h), **kw)


def subplots(nrows=1, ncols=1, width: float = 1.0, aspect: float = 0.618, **kw):
    import matplotlib.pyplot as plt
    w = width_in(width)
    h = min(aspect * w, MAX_HEIGHT_PT * PT)
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def panel_label(ax, text):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=FS["panel"], color=INK)


def save(fig, stem: str):
    """PDF (vector, raster layers at 600 dpi) + 600 dpi PNG, exact canvas size."""
    fig.savefig(stem + ".pdf", dpi=600, metadata={"Title": None})
    fig.savefig(stem + ".png", dpi=600)
