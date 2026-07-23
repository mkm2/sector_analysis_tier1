"""
Unified growth maps over ALL rules -- exponential, polynomial, and ergodic.

A single "growth base" axis is degenerate: every series here is asymptotically
y ~ N^alpha * b^N, and the base b = e^kappa alone cannot separate

  * a linear rule from a constant one       (both have b = 1), nor
  * an ergodic rule from a domain-wall rule  (both have b = 2).

The honest coordinates are the TWO exponents (kappa, alpha) -- the leading rate
and the sub-leading power.  That is the exact parametrisation, and it separates
every regime: exponential rules sit at b>1 with alpha a small correction;
polynomial rules sit at b=1 and climb the alpha-axis by their degree; a constant
rule is the origin (b=1, alpha=0); an ergodic/volume-law rule is (b=2, alpha=0).
It even splits the two ways of reaching b=2: the binomial rules 150/105 carry
alpha=-1/2 (the 1/sqrt(N) of the central binomial), so they sit BELOW the
ergodic point, not on it.

Two figures are produced:

  fig_growth_plane_{bc}   the (base, power) plane, two panels (D_max, #sectors).
                          This is the complete, non-degenerate map.
  fig_growth_baseonly_{bc} the [1,2]x[1,2] base-vs-base map (#sectors base against
                          D_max base), with marker size encoding polynomial degree
                          and ergodic rules as a distinct marker.  Closer to the
                          earlier figure; kept because the base-vs-base view makes
                          the ergodic/domain-wall corner immediately legible.

MODEL SELECTION IS NOT UNIFORMLY TRUSTWORTHY, and the map says so.  For the
unitary rules the growth law is analytic (rule 156's room-packing base 4^{1/5}
and its 150/105 binomial alpha=-1/2 are derived, not fitted; the Fibonacci-type
bases are exact integer recurrences).  For most dissipative rules the class comes
from the null-calibrated BIC of scaling/dissipative.py.  A FILLED marker means
the base is analytically exact (an exact integer recurrence, or one of the
derived unitary constants); an OPEN marker means it is a fit.  So the eye can see
at a glance which points are theorems and which are regressions.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from ..core import rules
from .dissipative import summarize_rule
from .fits import find_integer_recurrence, fit_series
from .summary import load_series

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEXT = "#0b0b0b"
MUTED = "#8a8a86"

# class -> (colour, marker, legend label).  Fixed CVD-safe hues.
# PRIMARY encoding is the rule FAMILY, because that is what makes the algebraic
# bases (un)surprising: a V-free rule is DETERMINISTIC on basis states (both
# Kraus branches of a reset land on the same state), so its transition graph is
# a classical functional graph and its sector count is an ordinary elementary-CA
# transfer-matrix count -- Fibonacci/Lucas, Padovan/Perrin, tribonacci, ... (R5,
# "classical baseline").  The genuinely quantum rules are the MIXED ones, whose
# graph is not faithful.  Colour = family; marker = growth class; fill = exact.
FAMILY_STYLE = {
    "unitary":   ("#2a78d6", "unitary (V only, 16)"),
    "classical": ("#1a9e5a", "classical (V-free, 80)"),
    "mixed":     ("#e34948", "mixed (V + reset, 160)"),
}
CLASS_MARKER = {
    "exponential": "o", "polynomial": "s", "constant": "^",
    "ergodic": "*", "irregular": "X",
}


def rule_family(t) -> str:
    """unitary (I/V only) | classical (V-free, has a reset) | mixed (both)."""
    from ..core.rules import is_unitary
    if is_unitary(t):
        return "unitary"
    return "classical" if "V" not in t else "mixed"

# Named exponential bases, for reference lines on the base axis.  Kept to the
# iconic few; the plastic number rho=1.3247 is dropped because it sits 0.005
# from 4^{1/5} and the two labels overprint.
REF_BASES = [
    (4 ** 0.2, r"$4^{1/5}$"),
    (1.6180339887, r"$\varphi$"),
    (1.7548776662, r"$\psi_{201}$"),
    (2.0, r"$2$"),
]

# Analytic overrides for the unitary rules whose law is DERIVED, not fitted
# (QCA_Circuits.pdf App. B and R2 sec. "binomial rules").  Keyed by rule; each
# entry gives (base, alpha) for the D_max series.  Everything else -- the
# Fibonacci/Lucas/tribonacci bases -- is already exact via find_integer_recurrence
# and needs no override.
_DMAX_ANALYTIC: Dict[int, Tuple[float, float]] = {
    156: (4 ** 0.2, -0.5),   # room-packing saddle point; central-binomial-like
    198: (4 ** 0.2, -0.5),   # reflection partner of 156
    150: (2.0, -0.5),        # central binomial C(N+1, .)
    105: (2.0, -0.5),
}


def _series_descriptor(Ns, ys, analytic: Optional[Tuple[float, float]] = None) -> Optional[Dict]:
    """(class, base, alpha, exact) for one series, unitary-style (no period split).

    class from BIC; base from an exact integer recurrence when one exists,
    else e^kappa; alpha is the polynomial degree (M1) or the sub-leading power
    (M2).  `analytic` overrides base+alpha and marks the point exact.
    """
    f = fit_series(Ns, ys)
    if not f.get("ok"):
        return None
    cls, model = f["growth_class"], f["best_model"]
    if model == "M0":
        base, alpha = 1.0, 0.0
    elif model == "M1":
        base, alpha = 1.0, float(f["params"]["M1"][1])
    else:
        base, alpha = float(f["base"]), float(f["alpha_M2"])
    exact = False
    rec = find_integer_recurrence(list(ys))
    if rec.get("ok") and rec["base"] > 1.0:
        base, exact = rec["base"], True
    if analytic is not None:
        base, alpha, exact = analytic[0], analytic[1], True
    return {"cls": cls, "base": base, "alpha": alpha, "exact": exact}


def _dissipative_descriptor(rule: int, bc: str) -> Dict[str, Optional[Dict]]:
    """Per-series descriptors for a dissipative rule, honouring the calibrated
    period-aware summary in scaling.dissipative (its base and class), and taking
    the polynomial degree from an M1 fit where the class is polynomial."""
    row = summarize_rule(rule, bc, None)
    s = load_series(rule, bc)
    out: Dict[str, Optional[Dict]] = {}
    for key in ("n_recurrent", "d_max"):
        cls = row[f"{key}_growth"]
        if cls == "irregular":
            out[key] = {"cls": "irregular", "base": None, "alpha": None,
                        "exact": False, "kind": row.get(f"{key}_irregular_kind")}
            continue
        exact_base = row[f"{key}_base_exact"]
        base = exact_base if exact_base is not None else (row[f"{key}_base"] or 1.0)
        exact = exact_base is not None
        if cls == "constant":
            alpha = 0.0
        elif cls == "polynomial":
            f = fit_series(s["N"], s[key])
            alpha = float(f["params"]["M1"][1]) if f.get("ok") else 1.0
            base = 1.0
        else:  # exponential: sub-leading power is not identifiable under a
               # period split, so report 0 (points cluster near the base axis).
            alpha = 0.0
        out[key] = {"cls": cls, "base": float(base), "alpha": float(alpha),
                    "exact": exact}
    return out


def rule_points(bc: str) -> List[Dict]:
    """One record per rule: descriptors for #sectors and D_max, plus ergodic flag."""
    pts = []
    for rule in range(256):
        t = rules.wolfram_to_tuple(rule)
        recs = results_io.load_results(rule, bc)
        if not recs:
            continue
        unitary = rules.is_unitary(t)
        ergodic = any(v.get("ergodic_flag") for v in recs.values())
        rec = {"rule": rule, "tuple": "".join(t), "unitary": unitary,
               "ergodic": ergodic, "family": rule_family(t)}
        if ergodic:
            # A single volume-law sector: D_max ~ 2^N, #sectors O(1).
            rec["d_max"] = {"cls": "ergodic", "base": 2.0, "alpha": 0.0,
                            "exact": True}
            rec["n_recurrent"] = {"cls": "constant", "base": 1.0, "alpha": 0.0,
                                  "exact": True}
        elif unitary:
            s = load_series(rule, bc)
            if len(s["N"]) < 3:
                continue
            rec["d_max"] = _series_descriptor(s["N"], s["d_max"],
                                              _DMAX_ANALYTIC.get(rule))
            rec["n_recurrent"] = _series_descriptor(s["N"], s["n_recurrent"])
        else:
            d = _dissipative_descriptor(rule, bc)
            rec["d_max"], rec["n_recurrent"] = d["d_max"], d["n_recurrent"]
        if rec.get("d_max") and rec.get("n_recurrent"):
            pts.append(rec)
    return pts


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


# --------------------------------------------------------------------------
# Figure 1: the (base, power) plane
# --------------------------------------------------------------------------
def _plane_panel(ax, pts, key, title):
    _style(ax)
    for v, name in REF_BASES:
        ax.axvline(v, color=MUTED, lw=0.6, ls=":", alpha=0.6, zorder=1)
        ax.annotate(name, (v, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, 2), ha="center",
                    fontsize=6.5, color=MUTED)
    for r in pts:
        d = r[key]
        if d is None or d["cls"] == "irregular" or d["base"] is None:
            continue
        colour = FAMILY_STYLE[r["family"]][0]
        marker = CLASS_MARKER[d["cls"]]
        # tiny deterministic jitter so coincident rules (reflection/spin-flip
        # partners land on the same exact point) do not fully overprint
        jx = ((r["rule"] * 37) % 7 - 3) * 0.004
        jy = ((r["rule"] * 53) % 7 - 3) * 0.02
        ax.scatter(d["base"] + jx, d["alpha"] + jy, s=46, marker=marker,
                   facecolor=colour if d["exact"] else "none",
                   edgecolor=colour, linewidth=1.0,
                   alpha=0.9 if d["exact"] else 0.7, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(0.94, 2.06)
    ax.set_xlabel(r"growth base $b=e^{\kappa}$")
    ax.set_ylabel(r"sub-leading power $\alpha$  ($y\sim N^{\alpha}b^{N}$)")
    ax.set_title(title, fontsize=10, loc="left")


def _family_class_legend(ax):
    """Two-part legend: family (colour) and growth class (marker)."""
    from matplotlib.lines import Line2D
    fam = [Line2D([0], [0], marker="o", linestyle="none", color=c,
                  markersize=6, label=lbl)
           for c, lbl in FAMILY_STYLE.values()]
    cls = [Line2D([0], [0], marker=m, linestyle="none", color="#555",
                  markerfacecolor="none", markersize=6, label=name)
           for name, m in CLASS_MARKER.items() if name != "irregular"]
    fill = [Line2D([0], [0], marker="o", linestyle="none", color="#555",
                   markerfacecolor="#555", markersize=6, label="exact base"),
            Line2D([0], [0], marker="o", linestyle="none", color="#555",
                   markerfacecolor="none", markersize=6, label="fitted base")]
    leg1 = ax.legend(handles=fam, frameon=False, fontsize=7.5,
                     loc="upper left", title="family")
    leg1.get_title().set_fontsize(7.5)
    ax.add_artist(leg1)
    ax.legend(handles=cls + fill, frameon=False, fontsize=7,
              loc="upper right", ncol=1)


def fig_growth_plane(bc: str, out: str):
    pts = rule_points(bc)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    _plane_panel(axes[0], pts, "d_max", f"$D_{{\\max}}$ ({bc})")
    _plane_panel(axes[1], pts, "n_recurrent", f"$\\#$sectors ({bc})")
    _family_class_legend(axes[0])
    n_irr = sum(1 for r in pts if r["d_max"] and r["d_max"]["cls"] == "irregular")
    fig.suptitle("Growth map: leading rate vs sub-leading power, all rules "
                 f"({len(pts)} shown, {n_irr} irregular $D_{{\\max}}$ omitted). "
                 "Colour = family, marker = growth class, filled = exact base.",
                 fontsize=10.5, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: base-only, base(#sectors) vs base(D_max), enriched
# --------------------------------------------------------------------------
def fig_growth_baseonly(bc: str, out: str):
    pts = rule_points(bc)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    _style(ax)
    for v, name in REF_BASES:
        ax.axvline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
        ax.axhline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax.plot([1, 2], [1, 2], color=MUTED, lw=0.7, ls="--", alpha=0.6, zorder=1)
    seen = set()
    for r in pts:
        dn, dd = r["n_recurrent"], r["d_max"]
        if dn is None or dd is None or dn["cls"] == "irregular" or dd["cls"] == "irregular":
            continue
        bn = dn["base"] if dn["base"] else 1.0
        bd = dd["base"] if dd["base"] else 1.0
        # marker size grows with the polynomial degree of D_max, so the
        # sub-exponential pile at base 1 spreads out by degree
        deg = max(dd["alpha"], dn["alpha"], 0.0)
        size = 26 + 42 * min(deg, 3.0)
        # base-only view: colour = family, size = degree, single marker shape
        # (the growth class is carried by the companion (base, alpha) plane)
        colour = FAMILY_STYLE[r["family"]][0]
        marker = "*" if r["ergodic"] else "o"
        jx = ((r["rule"] * 37) % 7 - 3) * 0.004
        jy = ((r["rule"] * 53) % 7 - 3) * 0.004
        ax.scatter(bn + jx, bd + jy, s=size, marker=marker,
                   facecolor=colour if dd["exact"] else "none",
                   edgecolor=colour, linewidth=1.0,
                   alpha=0.85 if dd["exact"] else 0.7, zorder=3)
    # label the anchor rules, with hand-tuned offsets so coincident partners
    # (150 with the ergodic cluster; 108 on top of 201) do not overprint
    anchor = {204: (-24, -12), 156: (7, 3), 201: (7, 3), 108: (7, -11),
              150: (7, 6), 60: (-20, 6)}
    for r in pts:
        if r["rule"] in anchor and r["d_max"]:
            bn = r["n_recurrent"]["base"] or 1.0
            bd = r["d_max"]["base"] or 1.0
            ox, oy = anchor[r["rule"]]
            ax.annotate(f"W{r['rule']}", (bn, bd), textcoords="offset points",
                        xytext=(ox, oy), fontsize=7, color=TEXT, zorder=5)
    ax.set_xlim(0.96, 2.06)
    ax.set_ylim(0.96, 2.08)
    ax.set_xlabel(r"growth base of $\#$sectors")
    ax.set_ylabel(r"growth base of $D_{\max}$")
    ax.set_title(f"Growth-base map ({bc}): colour = family, size $\\propto$ "
                 "poly. degree", fontsize=10, loc="left")
    from matplotlib.lines import Line2D
    fam = [Line2D([0], [0], marker="o", linestyle="none", color=c, markersize=7,
                  label=lbl) for c, lbl in FAMILY_STYLE.values()]
    ax.legend(handles=fam, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.52, 0.99))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: the corner map -- base-vs-base main panel with alpha margins
# --------------------------------------------------------------------------
# Merges the base-vs-base view and the two (base, alpha) panels into one joint
# layout.  The main panel is base(#sec) x base(D_max); the bottom margin shares
# its x-axis (base #sec) and plots the #sec sub-leading power below it; the left
# margin shares its y-axis (base D_max) and plots the D_max sub-leading power
# beside it.  Reading a rule: its horizontal position is fixed across the main
# and bottom panels, its vertical position across the main and left panels, so
# the full (base, alpha) description of both series is read off one point.
_BASE_XLIM = (0.955, 2.065)
_BASE_YLIM = (0.955, 2.085)
_ANSEC_LIM = (-0.13, 1.18)     # #sectors sub-leading power (0..~1, linear)
_ADMAX_LIM = (-0.72, 2.02)     # D_max sub-leading power (-1/2 binomial .. ~2)


def _corner_points(bc):
    """(bn, an, bd, ad, colour, en, ed) per rule with both bases defined."""
    out = []
    for r in rule_points(bc):
        dn, dd = r["n_recurrent"], r["d_max"]
        if (dn is None or dd is None or dn["cls"] == "irregular"
                or dd["cls"] == "irregular" or dn["base"] is None
                or dd["base"] is None):
            continue
        colour = FAMILY_STYLE[r["family"]][0]
        jx = ((r["rule"] * 37) % 7 - 3) * 0.004
        jy = ((r["rule"] * 53) % 7 - 3) * 0.004
        out.append((dn["base"] + jx, dn["alpha"] + jy, dd["base"] + jy,
                    dd["alpha"] + jx, colour, dn["exact"], dd["exact"]))
    return out


def _scatter(ax, xs, ys, cols, fills, s=34):
    for x, y, c, f in zip(xs, ys, cols, fills):
        ax.scatter(x, y, s=s, marker="o", facecolor=c if f else "none",
                   edgecolor=c, linewidth=0.9, alpha=0.85 if f else 0.65,
                   zorder=3)


def fig_growth_corner(bc: str, out: str):
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    P = _corner_points(bc)
    bn = [p[0] for p in P]; an = [p[1] for p in P]
    bd = [p[2] for p in P]; ad = [p[3] for p in P]
    col = [p[4] for p in P]; en = [p[5] for p in P]; ed = [p[6] for p in P]

    fig = plt.figure(figsize=(9.6, 8.6))
    gs = GridSpec(2, 2, width_ratios=[1.05, 3.0], height_ratios=[3.0, 1.05],
                  wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_leg = fig.add_subplot(gs[1, 0]); ax_leg.axis("off")
    for a in (ax_main, ax_left, ax_bot):
        _style(a)

    # --- main: base(#sec) vs base(D_max) --------------------------------
    for v, _ in REF_BASES:
        ax_main.axvline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
        ax_main.axhline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax_main.plot([1, 2], [1, 2], color=MUTED, lw=0.7, ls="--", alpha=0.55, zorder=1)
    _scatter(ax_main, bn, bd, col, [a and b for a, b in zip(en, ed)])
    ax_main.set_xlim(*_BASE_XLIM); ax_main.set_ylim(*_BASE_YLIM)
    ax_main.tick_params(labelbottom=False, labelleft=False)
    for v, name in REF_BASES:
        ax_main.annotate(name, (v, _BASE_YLIM[1]), textcoords="offset points",
                         xytext=(0, 1), ha="center", va="bottom", fontsize=7,
                         color=MUTED, annotation_clip=False)
        ax_main.annotate(name, (_BASE_XLIM[1], v), textcoords="offset points",
                         xytext=(2, 0), ha="left", va="center", fontsize=7,
                         color=MUTED, annotation_clip=False)

    # --- bottom margin: base(#sec) [shared x] vs alpha(#sec) -------------
    for v, _ in REF_BASES:
        ax_bot.axvline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax_bot.axhline(0, color=MUTED, lw=0.8, zorder=1)
    _scatter(ax_bot, bn, an, col, en)
    ax_bot.set_ylim(*_ANSEC_LIM)
    ax_bot.set_yticks([0, 0.5, 1.0])
    ax_bot.set_xlabel(r"growth base of $\#$sectors")
    ax_bot.set_ylabel(r"$\alpha_{\#\mathrm{sec}}$", labelpad=1)

    # --- left margin: alpha(D_max) vs base(D_max) [shared y] -------------
    for v, _ in REF_BASES:
        ax_left.axhline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax_left.axvline(0, color=MUTED, lw=0.8, zorder=1)
    _scatter(ax_left, ad, bd, col, ed)
    ax_left.set_xlim(*_ADMAX_LIM)
    ax_left.invert_xaxis()          # alpha grows leftward, base spine by main
    ax_left.set_ylabel(r"growth base of $D_{\max}$")
    ax_left.set_xlabel(r"$\alpha_{D_{\max}}$")

    # --- legend in the empty corner (hug the left, clear of the alpha label) --
    short = {"unitary": "unitary (16)", "classical": "classical, V-free (80)",
             "mixed": "mixed, V+reset (160)"}
    fam = [Line2D([0], [0], marker="o", linestyle="none", color=c, markersize=8,
                  label=short[k])
           for k, (c, _) in FAMILY_STYLE.items()]
    fillk = [Line2D([0], [0], marker="o", linestyle="none", color="#555",
                    markerfacecolor="#555", markersize=7, label="exact base"),
             Line2D([0], [0], marker="o", linestyle="none", color="#555",
                    markerfacecolor="none", markersize=7, label="fitted base")]
    leg = ax_leg.legend(handles=fam + fillk, frameon=False, fontsize=8.5,
                        loc="center left", bbox_to_anchor=(-0.02, 0.5),
                        title="family / base")
    leg.get_title().set_fontsize(8.5)

    fig.suptitle(f"Growth map ({bc}): base--base core with $\\alpha$ margins "
                 "(colour = family, filled = exact base)",
                 fontsize=11.5, x=0.5, y=0.995, color=TEXT)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for bc in ("obc0", "pbc"):
        fig_growth_plane(bc, os.path.join(FIGURES_DIR, f"fig_growth_plane_{bc}.pdf"))
        fig_growth_baseonly(bc, os.path.join(FIGURES_DIR, f"fig_growth_baseonly_{bc}.pdf"))
        fig_growth_corner(bc, os.path.join(FIGURES_DIR, f"fig_growth_corner_{bc}.pdf"))
    print(f"wrote growth maps to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
