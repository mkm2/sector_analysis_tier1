"""
Tier 1e figures F1-F5 (task sec.5).

F1  SECTOR MAP        base(n_wcc) vs base(d_max_wcc), all 256 rules, with the
                      hyperbola b = 2/a drawn as the exclusion boundary.
F2  MONITORED MAP     the same plane for terminal SCCs, with the hyperbola shown
                      as a reference and the allowed-below region shaded -- the
                      contrast is the point.
F3  SECTOR vs ATTRACTOR   att_per_sector, i.e. how much monitored structure
                      lives inside one enclosure.
F4  BASIN STRUCTURE   max_basin/2^N, shared-basin fraction, transient_fraction.
F5  VALIDATION        sum-rule residuals, hyperbola margins, unitary A2 check.

Every caption states which EXPERIMENT its numbers describe: WCC quantities are
exact for both the monitored and the unmonitored dynamics; terminal-SCC
quantities are monitored-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from .. import results_io                 # noqa: E402
from ..core import rules                  # noqa: E402
from . import sectors                     # noqa: E402

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

FAMILY_COLOUR = {"unitary": "#1f77b4", "classical": "#2ca02c",
                 "mixed": "#d62728"}
FAMILY_LABEL = {"unitary": "unitary (16)",
                "classical": "V-free dissipative",
                "mixed": "V + reset"}
MUTED = "#9a9a95"
TEXT = "#333333"


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _hyperbola(ax, lo=1.0, hi=2.05, shade_below=False, label=True):
    a = np.linspace(max(lo, 1.0), hi, 400)
    b = 2.0 / a
    ax.plot(a, b, color="k", lw=1.2, ls="-", zorder=2,
            label=r"$ab=2$ (exclusion)" if label else None)
    if shade_below:
        ax.fill_between(a, 0, b, color="#cccccc", alpha=0.35, zorder=0)


def _jitter(rule, s=0.005):
    return (((rule * 37) % 7 - 3) * s, ((rule * 53) % 7 - 3) * s)


# --- F1 -----------------------------------------------------------------------

def fig_sector_map(bc: str, out: str, data: Optional[Dict] = None):
    d = data or sectors.load(bc) or sectors.build(bc)
    # irregular series have no base and cannot be placed in this plane; they are
    # counted in the title rather than plotted at a fictitious coordinate
    pts = [p for p in d["points"]
           if p["n_wcc"]["base"] is not None
           and p["d_max_wcc"]["base"] is not None]
    n_irr = len(d["points"]) - len(pts)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    _style(ax)
    _hyperbola(ax, shade_below=True)
    for p in pts:
        a, b = p["n_wcc"], p["d_max_wcc"]
        jx, jy = _jitter(p["rule"])
        c = FAMILY_COLOUR[p["family"]]
        filled = a["exact"] and b["exact"]
        ax.scatter(a["base"] + jx, b["base"] + jy, s=38, marker="o",
                   facecolor=c if filled else "none", edgecolor=c,
                   linewidth=1.0, alpha=0.85 if filled else 0.7, zorder=3)
    for r, lab in ((204, "204"), (51, "51"), (150, "150"), (156, "156")):
        p = next((x for x in pts if x["rule"] == r), None)
        if p:
            ax.annotate(lab, (p["n_wcc"]["base"], p["d_max_wcc"]["base"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5,
                        color=TEXT)
    ax.set_xlabel(r"base $a$ of $\#$sectors ($n_{\rm wcc}\sim a^N$)")
    ax.set_ylabel(r"base $b$ of $D_{\max}$ ($\sim b^N$)")
    ax.set_title(f"F1  sector map, {len(pts)} rules ({bc}) — "
                 f"exact for BOTH experiments"
                 + (f"; {n_irr} irregular omitted" if n_irr else ""),
                 fontsize=10)
    hs = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOUR[k],
                     label=FAMILY_LABEL[k]) for k in FAMILY_COLOUR]
    hs.append(plt.Line2D([], [], color="k", lw=1.2, label=r"$ab=2$"))
    hs.append(plt.Line2D([], [], marker="o", ls="", color="k",
                         markerfacecolor="none", label="fitted base"))
    ax.legend(handles=hs, fontsize=7, loc="upper right")

    # marginal: the alpha panels
    ax2 = axes[1]
    _style(ax2)
    for p in pts:
        c = FAMILY_COLOUR[p["family"]]
        ax2.scatter(p["n_wcc"]["alpha"], p["d_max_wcc"]["alpha"], s=26,
                    facecolor=c, edgecolor="none", alpha=0.6, zorder=3)
    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-3.2, 5.2)
    ax2.axhline(-0.5, color=MUTED, ls=":", lw=0.8)
    ax2.annotate(r"$\alpha=-1/2$ (binomial)", (0.02, -0.5), fontsize=7,
                 color=TEXT, textcoords="offset points", xytext=(0, 4))
    ax2.set_xlabel(r"$\alpha_{\#{\rm sec}}$")
    ax2.set_ylabel(r"$\alpha_{D_{\max}}$")
    ax2.set_title("sub-leading powers", fontsize=10)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F2 -----------------------------------------------------------------------

def fig_monitored_map(bc: str, out: str, data: Optional[Dict] = None):
    d = data or sectors.load(bc) or sectors.build(bc)
    defs = d["attractor_deficits"]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _style(ax)
    _hyperbola(ax, shade_below=True)
    fam = {p["rule"]: p["family"] for p in d["points"]}
    for x in defs:
        c = FAMILY_COLOUR.get(fam.get(x["rule"], "mixed"), "#777777")
        jx, jy = _jitter(x["rule"])
        ax.scatter(x["a_att"] + jx, x["b_att"] + jy, s=34, facecolor="none",
                   edgecolor=c, linewidth=0.9, alpha=0.75, zorder=3)
    below = sum(1 for x in defs if x["deficit"] > 0)
    ax.set_xlabel(r"base $a_{\rm att}$ of $\#$terminal SCCs")
    ax.set_ylabel(r"base $b_{\rm att}$ of $D_{\max}^{\rm att}$")
    ax.set_title(f"F2  MONITORED attractor map ({bc})\n"
                 f"{below}/{len(defs)} rules lie in the shaded region — "
                 r"allowed, because terminal SCCs do not partition",
                 fontsize=9.5)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F3 -----------------------------------------------------------------------

def fig_sector_vs_attractor(bc: str, out: str, data: Optional[Dict] = None):
    d = data or sectors.load(bc) or sectors.build(bc)
    pts = [p for p in d["points"] if p["att_per_sector_at_Nmax"]
           and p["n_wcc"]["base"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    _style(ax)
    vals = np.array([p["att_per_sector_at_Nmax"] for p in pts])
    cols = [FAMILY_COLOUR[p["family"]] for p in pts]
    xs = np.array([p["n_wcc"]["base"] for p in pts])
    sc = ax.scatter(xs, vals, c=cols, s=34, alpha=0.75, zorder=3)
    ax.set_yscale("log")
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.8)
    ax.annotate("one attractor per sector", (1.0, 1.0), fontsize=7,
                textcoords="offset points", xytext=(2, 4), color=TEXT)
    ax.set_xlabel(r"base $a$ of $\#$sectors")
    ax.set_ylabel(r"$n_{\rm recurrent}/n_{\rm wcc}$ at $N_{\max}$")
    ax.set_title("F3  monitored attractors per enclosure", fontsize=10)

    ax = axes[1]
    _style(ax)
    tf = [p["transient_fraction_at_Nmax"] for p in pts
          if p["transient_fraction_at_Nmax"] is not None]
    cols2 = [FAMILY_COLOUR[p["family"]] for p in pts
             if p["transient_fraction_at_Nmax"] is not None]
    ax.scatter([p["d_max_ratio_at_Nmax"] for p in pts
                if p["transient_fraction_at_Nmax"] is not None],
               tf, c=cols2, s=34, alpha=0.75, zorder=3)
    ax.set_xlabel(r"$D_{\max}^{\rm wcc}/2^N$ at $N_{\max}$")
    ax.set_ylabel("transient fraction (MONITORED)")
    ax.set_title("how much of the space the attractors miss", fontsize=10)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F4 -----------------------------------------------------------------------

def _basin_series(rule: int, bc: str):
    """max_basin/2^N, shared fraction, transient fraction vs N -- MONITORED."""
    out = {"N": [], "max_basin": [], "shared": [], "tf": []}
    rb = results_io.load_basin_results(rule, bc)
    for N, rec in sorted(results_io.load_results(rule, bc).items()):
        if rec.get("ergodic_flag"):
            continue
        tot = 1 << N
        src = rb.get(N)
        basins = (results_io.basins_from_record(src) if src
                  else (rec.get("sizes_basins") or []))
        if not basins:
            continue
        shared = ((src or rec).get("shared_basin_size") or 0)
        sizes_rec = (results_io.sizes_from_record(rec, "sizes_recurrent") or [])
        out["N"].append(N)
        out["max_basin"].append(max(basins) / tot)
        out["shared"].append(shared / tot)
        out["tf"].append(1 - sum(sizes_rec) / tot)
    return out


def fig_basins(bc: str, out: str, rules_sel=(22, 28, 76, 108, 156, 201, 232)):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax in axes:
        _style(ax)
    for rule in rules_sel:
        s = _basin_series(rule, bc)
        if len(s["N"]) < 2:
            continue
        lab = f"W{rule}"
        axes[0].plot(s["N"], s["max_basin"], "o-", ms=3.5, lw=1.1, label=lab)
        axes[1].plot(s["N"], s["shared"], "o-", ms=3.5, lw=1.1, label=lab)
        axes[2].plot(s["N"], s["tf"], "o-", ms=3.5, lw=1.1, label=lab)
    axes[0].set_ylabel(r"$\max$ basin $/\,2^N$")
    axes[1].set_ylabel(r"shared basin $/\,2^N$")
    axes[2].set_ylabel("transient fraction")
    for ax, t in zip(axes, ("largest basin", "shared basin",
                            "transient fraction")):
        ax.set_xlabel("$N$")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("F4  basin structure — a MONITORED observable "
                 "(Tier-1d T3 certifies the unmonitored basins are zero "
                 "for the V-free rules)", fontsize=9.5, x=0.01, ha="left",
                 color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F5 -----------------------------------------------------------------------

def validation_stats(bc: str, data: Optional[Dict] = None) -> Dict:
    d = data or sectors.load(bc) or sectors.build(bc)
    sums, a2_ok, a2_tot, basin_ok, basin_bad, basin_unknown = [], 0, 0, 0, 0, 0
    for rule in range(256):
        for N, rec in results_io.load_wcc_results(rule, bc).items():
            if rec.get("aborted"):
                continue
            full = results_io.sizes_from_wcc_record(rec)
            sums.append(sum(full) - (1 << N))
            ch = rec.get("checks") or {}
            if ch.get("a2") is True:
                a2_ok += 1
                a2_tot += 1
            elif ch.get("a2") is False:
                a2_tot += 1
            b = rec.get("basin_sum_check")
            if b is True:
                basin_ok += 1
            elif b is False:
                basin_bad += 1
            else:
                basin_unknown += 1
    # irregular rules have no base and therefore no margin
    margins = [c["margin"] for c in d["hyperbola"] if c["margin"] is not None]
    return {"bc": bc, "n_units": len(sums),
            "sum_rule_max_abs_residual": max(abs(s) for s in sums) if sums else 0,
            "a2_ok": a2_ok, "a2_total": a2_tot,
            "basin_ok": basin_ok, "basin_bad": basin_bad,
            "basin_unknown": basin_unknown,
            "hyperbola_margins": margins,
            "hyperbola_violations": len(d["violations"])}


def fig_validation(bc: str, out: str, data: Optional[Dict] = None):
    d = data or sectors.load(bc) or sectors.build(bc)
    st = validation_stats(bc, d)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    _style(ax)
    m = np.array(st["hyperbola_margins"])
    ax.hist(m, bins=40, color="#4c72b0", alpha=0.85)
    ax.axvline(0.0, color="k", lw=1.2)
    ax.set_xlabel(r"$ab-2$")
    ax.set_ylabel("rules")
    ax.set_title(f"F5a  hyperbola margin ({st['hyperbola_violations']} "
                 f"violations of $ab\\geq2$)", fontsize=10)

    ax = axes[1]
    _style(ax)
    labels = ["sum rule\nresidual=0", "A2 unitary\nagreement",
              "basin sum\nrule"]
    vals = [st["n_units"], st["a2_ok"], st["basin_ok"]]
    tots = [st["n_units"], max(st["a2_total"], 1),
            st["basin_ok"] + st["basin_bad"] + st["basin_unknown"]]
    ax.bar(labels, [100 * v / t for v, t in zip(vals, tots)],
           color=["#55a868", "#4c72b0", "#c44e52"], alpha=0.85)
    for i, (v, t) in enumerate(zip(vals, tots)):
        ax.text(i, 100 * v / t + 1.5, f"{v}/{t}", ha="center", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% passing")
    ax.set_title(f"F5b  validation ({st['basin_unknown']} basin checks "
                 f"undefined: ergodic units, or $N$ beyond Tier 1a)",
                 fontsize=9.5)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1e figures F1-F5")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = sectors.build(args.bc) if args.rebuild else (
        sectors.load(args.bc) or sectors.build(args.bc))
    bc = args.bc
    fig_sector_map(bc, os.path.join(FIGURES_DIR, f"fig_sector_map_{bc}.pdf"), d)
    fig_monitored_map(bc, os.path.join(FIGURES_DIR,
                                       f"fig_monitored_map_{bc}.pdf"), d)
    fig_sector_vs_attractor(bc, os.path.join(
        FIGURES_DIR, f"fig_sector_vs_attractor_{bc}.pdf"), d)
    fig_basins(bc, os.path.join(FIGURES_DIR, f"fig_basins_{bc}.pdf"))
    st = fig_validation(bc, os.path.join(FIGURES_DIR,
                                         f"fig_validation_{bc}.pdf"), d)
    with open(os.path.join(sectors.ANALYTICS,
                           f"sector_validation_{bc}.json"), "w") as f:
        json.dump(st, f)
    print(json.dumps({k: v for k, v in st.items()
                      if k != "hyperbola_margins"}, indent=1))


if __name__ == "__main__":
    main()
