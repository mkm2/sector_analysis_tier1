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
CLASS_STYLE = {
    "ergodic":     ("#111111", "*", "ergodic (volume law)"),
    "exponential": ("#e34948", "o", "exponential"),
    "polynomial":  ("#2a78d6", "s", "polynomial"),
    "constant":    ("#8a8a86", "^", "constant"),
    "irregular":   ("#eda100", "X", "irregular (no base)"),
}

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
               "ergodic": ergodic}
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
    seen = set()
    for r in pts:
        d = r[key]
        if d is None or d["cls"] == "irregular" or d["base"] is None:
            continue
        colour, marker, label = CLASS_STYLE[d["cls"]]
        # tiny deterministic jitter so coincident rules (reflection/spin-flip
        # partners land on the same exact point) do not fully overprint
        jx = ((r["rule"] * 37) % 7 - 3) * 0.004
        jy = ((r["rule"] * 53) % 7 - 3) * 0.02
        ax.scatter(d["base"] + jx, d["alpha"] + jy, s=46, marker=marker,
                   facecolor=colour if d["exact"] else "none",
                   edgecolor=colour, linewidth=1.0,
                   alpha=0.9 if d["exact"] else 0.75, zorder=3,
                   label=label if label not in seen else None)
        seen.add(label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(0.94, 2.06)
    ax.set_xlabel(r"growth base $b=e^{\kappa}$")
    ax.set_ylabel(r"sub-leading power $\alpha$  ($y\sim N^{\alpha}b^{N}$)")
    ax.set_title(title, fontsize=10, loc="left")


def fig_growth_plane(bc: str, out: str):
    pts = rule_points(bc)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    _plane_panel(axes[0], pts, "d_max", f"$D_{{\\max}}$ ({bc})")
    _plane_panel(axes[1], pts, "n_recurrent", f"$\\#$sectors ({bc})")
    h, l = axes[0].get_legend_handles_labels()
    # a second legend entry pair explaining fill
    from matplotlib.lines import Line2D
    fill_key = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#444",
                       markeredgecolor="#444", label="analytic / exact base"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="#444", label="fitted base")]
    axes[0].legend(h + fill_key, l + [k.get_label() for k in fill_key],
                   frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    n_irr = sum(1 for r in pts if r["d_max"] and r["d_max"]["cls"] == "irregular")
    fig.suptitle("Growth map: leading rate vs sub-leading power, all rules "
                 f"({len(pts)} shown, {n_irr} irregular D_max omitted)",
                 fontsize=11, x=0.01, ha="left", color=TEXT)
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
        cls = "ergodic" if r["ergodic"] else dd["cls"]
        colour, marker, label = CLASS_STYLE[cls]
        jx = ((r["rule"] * 37) % 7 - 3) * 0.004
        jy = ((r["rule"] * 53) % 7 - 3) * 0.004
        ax.scatter(bn + jx, bd + jy, s=size, marker=marker,
                   facecolor=colour if dd["exact"] else "none",
                   edgecolor=colour, linewidth=1.0,
                   alpha=0.85 if dd["exact"] else 0.7, zorder=3,
                   label=label if label not in seen else None)
        seen.add(label)
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
    ax.set_title(f"Growth-base map, marker size $\\propto$ polynomial degree ({bc})",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.52, 0.99))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for bc in ("obc0", "pbc"):
        fig_growth_plane(bc, os.path.join(FIGURES_DIR, f"fig_growth_plane_{bc}.pdf"))
        fig_growth_baseonly(bc, os.path.join(FIGURES_DIR, f"fig_growth_baseonly_{bc}.pdf"))
    print(f"wrote growth maps to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
