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


def _hyperbola(ax, lo=0.96, hi=2.06):
    """The exclusion boundary, as a light dotted line -- a reference, not a
    decoration.  No shading: it competes with the scatter it is meant to frame."""
    a = np.linspace(max(lo, 0.5), hi, 400)
    ax.plot(a, 2.0 / a, color="#444444", lw=1.0, ls=":", zorder=2)


# Quantum rules (those containing a Hadamard V) are the ones of interest; the
# V-free rules are a classical baseline and are drawn back so they frame rather
# than crowd.  "mixed" = V + reset, "unitary" = V only, "classical" = V-free.
_EMPH = {
    "unitary":   dict(color="#1f4e9c", z=6, alpha=0.95, edge=1.3),
    "mixed":     dict(color="#c62828", z=5, alpha=0.85, edge=1.1),
    "classical": dict(color="#9bbf9b", z=3, alpha=0.55, edge=0.7),
}
_FAMLABEL = {"unitary": "unitary (V only), 16",
             "mixed": "V + reset, 160",
             "classical": "V-free baseline, 80"}


def _cells(pts, xkey, ykey):
    """Aggregate coincident points: {(x, y, family): count}.  Essential here --
    73% of the sector map sits in a single cell, so one marker per rule hides
    the entire distribution."""
    from collections import Counter
    c = Counter()
    for x, y, fam in pts:
        if x is None or y is None:
            continue
        c[(round(x, 3), round(y, 3), fam)] += 1
    return c


#: Families sharing a cell are nudged apart so none is hidden under another --
#: without this the 144 V+reset rules at (1,2) sit exactly under 33 V-free ones.
_FAM_DX = {"unitary": 0.0, "mixed": 0.019, "classical": -0.019}


def _scatter_cells(ax, cells, *, base=26.0, per=15.0, annotate_above=12):
    """Marker area grows with the number of rules stacked in the cell.  All
    markers are drawn before any label, so a count is never buried."""
    labels = []
    for fam in ("classical", "mixed", "unitary"):          # draw order = z order
        st = _EMPH[fam]
        dx = _FAM_DX[fam]
        for (x, y, f), n in cells.items():
            if f != fam:
                continue
            r = base + per * np.sqrt(n)
            ax.scatter(x + dx, y, s=r, marker="o", facecolor=st["color"],
                       edgecolor="white", linewidth=st["edge"],
                       alpha=st["alpha"], zorder=st["z"])
            if n >= annotate_above:
                labels.append((x + dx, y, n, st["color"], r))
    for x, y, n, col, r in labels:
        inside = r > 110
        ax.annotate(f"{n}", (x, y), fontsize=6.8, ha="center",
                    va="center" if inside else "bottom",
                    xytext=(0, 0) if inside else (0, 7),
                    textcoords="offset points",
                    color="white" if inside else col,
                    fontweight="bold", zorder=20)


def _panel(ax, cells, title, xlabel, ylabel):
    _style(ax)
    _hyperbola(ax)
    _scatter_cells(ax, cells)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.90, 2.12)
    ax.set_ylim(0.88, 2.14)


# --- F1 + F2: the two maps on identical axes ----------------------------------

def fig_sector_map(bc: str, out: str, data: Optional[Dict] = None):
    """
    Sector map and monitored-attractor map side by side, same axes, so the
    contrast is the figure rather than a claim about two figures.
    """
    d = data or sectors.load(bc) or sectors.build(bc)
    fam = {p["rule"]: p["family"] for p in d["points"]}

    sec = _cells([(p["n_wcc"]["base"], p["d_max_wcc"]["base"], p["family"])
                  for p in d["points"]], None, None)
    att = _cells([(x["a_att"], x["b_att"], fam.get(x["rule"], "mixed"))
                  for x in d["attractor_deficits"]], None, None)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), sharex=True, sharey=True)
    _panel(axes[0], sec,
           f"sectors (WCC) — exact for BOTH experiments",
           r"base $a$ of $\#$sectors", r"base $b$ of $D_{\max}$")
    _panel(axes[1], att,
           "monitored attractors (terminal SCC)",
           r"base $a_{\rm att}$ of $\#$attractors",
           r"base $b_{\rm att}$ of $D_{\max}^{\rm att}$")

    # 51 and 150 sit at exactly the same point, so the labels are staggered
    for r, lab, off in ((204, "204", (-24, 6)), (51, "51", (14, 12)),
                        (150, "150", (14, -14)), (156, "156", (12, 4))):
        p = next((x for x in d["points"] if x["rule"] == r), None)
        if p and p["n_wcc"]["base"] is not None:
            axes[0].annotate(lab, (p["n_wcc"]["base"], p["d_max_wcc"]["base"]),
                             textcoords="offset points", xytext=off,
                             fontsize=7.5, color=TEXT, zorder=21,
                             arrowprops=dict(arrowstyle="-", lw=0.5,
                                             color="#888888"))

    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=_EMPH[k]["color"], markeredgecolor="white",
                     label=_FAMLABEL[k]) for k in ("unitary", "mixed", "classical")]
    hs.append(plt.Line2D([], [], color="#444444", lw=1.0, ls=":",
                         label=r"$ab=2$ (excluded below, left panel only)"))
    axes[0].legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.suptitle("Marker area grows with the number of rules stacked in a cell; "
                 "the count is printed inside crowded cells.  "
                 f"({bc})", fontsize=9, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_monitored_map(bc: str, out: str, data: Optional[Dict] = None):
    """The attractor map alone, for the rules that carry a Hadamard."""
    d = data or sectors.load(bc) or sectors.build(bc)
    fam = {p["rule"]: p["family"] for p in d["points"]}
    quantum = [(x["a_att"], x["b_att"], fam.get(x["rule"], "mixed"))
               for x in d["attractor_deficits"]
               if fam.get(x["rule"]) in ("unitary", "mixed")]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    _panel(ax, _cells(quantum, None, None),
           f"monitored attractor map, quantum rules only ({bc})",
           r"base $a_{\rm att}$ of $\#$attractors",
           r"base $b_{\rm att}$ of $D_{\max}^{\rm att}$")
    below = sum(1 for x in d["attractor_deficits"] if x["deficit"] > 0)
    ax.annotate(f"{below}/{len(d['attractor_deficits'])} rules lie below "
                r"$ab=2$ — allowed:" "\n" r"terminal SCCs do not partition",
                (0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7.5, color=TEXT)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F3: the displacement between the two maps --------------------------------

def fig_sector_vs_attractor(bc: str, out: str, data: Optional[Dict] = None):
    """
    One arrow per rule, from its position in the sector map to its position in
    the attractor map.  This is the difference between the two scatters, drawn
    as a difference rather than left for the eye to compute.
    """
    from collections import Counter
    d = data or sectors.load(bc) or sectors.build(bc)
    pts = {p["rule"]: p for p in d["points"]}
    moves: Counter = Counter()
    for x in d["attractor_deficits"]:
        p = pts.get(x["rule"])
        if not p or p["n_wcc"]["base"] is None:
            continue
        moves[(round(p["n_wcc"]["base"], 3), round(p["d_max_wcc"]["base"], 3),
               round(x["a_att"], 3), round(x["b_att"], 3), p["family"])] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    ax = axes[0]
    _style(ax)
    _hyperbola(ax)
    for (x0, y0, x1, y1, f), n in sorted(moves.items(), key=lambda kv: kv[1]):
        st = _EMPH[f]
        if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
            ax.scatter(x0, y0, s=26 + 13 * np.sqrt(n), facecolor=st["color"],
                       edgecolor="white", lw=0.8, alpha=st["alpha"],
                       zorder=st["z"])
            continue
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=st["color"],
                                    lw=0.6 + 0.9 * np.sqrt(n), alpha=0.7,
                                    shrinkA=1.5, shrinkB=1.5),
                    zorder=st["z"])
    ax.set_xlabel(r"base $a$"), ax.set_ylabel(r"base $b$")
    ax.set_title("sector $\\rightarrow$ monitored attractor, per rule",
                 fontsize=10)
    ax.set_xlim(0.90, 2.12)
    ax.set_ylim(0.88, 2.14)

    # right: how far each family moves
    ax = axes[1]
    _style(ax)
    for f in ("unitary", "mixed", "classical"):
        vals = [x["deficit"] for x in d["attractor_deficits"]
                if pts.get(x["rule"], {}).get("family") == f]
        if not vals:
            continue
        ax.hist(vals, bins=np.linspace(-1.0, 1.05, 42), histtype="stepfilled",
                color=_EMPH[f]["color"], alpha=0.55 if f != "classical" else 0.35,
                label=f"{_FAMLABEL[f]}  (median {np.median(vals):+.2f})",
                zorder=_EMPH[f]["z"])
    ax.axvline(0.0, color="#444444", ls=":", lw=1.0)
    ax.set_xlabel(r"deficit $2-a_{\rm att}b_{\rm att}$")
    ax.set_ylabel("rules")
    ax.set_title("transient dominance, by family", fontsize=10)
    ax.legend(fontsize=7.5)
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
    for plane in ("sector", "attractor"):
        fig_dissipation_clusters(bc, os.path.join(
            FIGURES_DIR, f"fig_dissip_{plane}_{bc}.pdf"), plane=plane, data=d)
    st = fig_validation(bc, os.path.join(FIGURES_DIR,
                                         f"fig_validation_{bc}.pdf"), d)
    with open(os.path.join(sectors.ANALYTICS,
                           f"sector_validation_{bc}.json"), "w") as f:
        json.dump(st, f)
    print(json.dumps({k: v for k, v in st.items()
                      if k != "hyperbola_margins"}, indent=1))


if __name__ == "__main__":
    main()


# --- F6: clustering the dissipative rules by their coherent correspondent -----

def _plane_points(d: Dict, plane: str):
    """{rule: (a, b)} in the requested plane."""
    if plane == "sector":
        return {p["rule"]: (p["n_wcc"]["base"], p["d_max_wcc"]["base"])
                for p in d["points"]
                if p["n_wcc"]["base"] is not None
                and p["d_max_wcc"]["base"] is not None}
    return {x["rule"]: (x["a_att"], x["b_att"]) for x in d["attractor_deficits"]}


def dissipation_clusters(d: Dict, plane: str):
    """
    [(parent, parent_xy, [(child, child_xy), ...]), ...] ordered by cluster size.

    The parent is the coherent correspondent (rules.coherent_part): every reset
    switched off.  Note the parent is unitary, so by A2 its sector and attractor
    coordinates are the same point -- the two planes differ only in where the
    CHILDREN land, which is exactly what these panels show.
    """
    xy = _plane_points(d, plane)
    fam = {p["rule"]: p["family"] for p in d["points"]}
    out = []
    for parent in sorted(rules.UNITARY_RULES):
        allkids = [c for c in rules.dissipative_children(parent)
                   if fam.get(c) == "mixed"]
        kids = [(c, xy[c]) for c in allkids if c in xy]
        if not allkids:
            continue
        # A parent with no coordinate in this plane is kept, with pxy=None: six
        # unitary rules are ergodic at obc0 and Tier 1a never produced an
        # attractor descriptor for them, so the cluster exists but has no
        # origin to draw arrows from.  Dropping it would silently lose 6 of 14.
        out.append((parent, xy.get(parent), kids, len(allkids)))
    return sorted(out, key=lambda t: (-len(t[2]), t[0]))


def fig_dissipation_clusters(bc: str, out: str, plane: str = "attractor",
                             data: Optional[Dict] = None):
    """
    One panel per unitary rule: where its dissipative descendants go when the
    resets are switched on.  Star = the coherent rule, dots = its V+reset
    children, arrows = the move that adding dissipation produces.
    """
    d = data or sectors.load(bc) or sectors.build(bc)
    cl = dissipation_clusters(d, plane)
    ncol = 5
    nrow = int(np.ceil(len(cl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.15 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    lab = ("sectors (WCC)" if plane == "sector"
           else "monitored attractors (terminal SCC)")
    for ax, (parent, pxy, kids, n_all) in zip(axes, cl):
        _style(ax)
        _hyperbola(ax)
        from collections import Counter
        cells = Counter(( round(x, 3), round(y, 3)) for _, (x, y) in kids)
        for (x, y), n in cells.items():
            if pxy is None:
                continue
            if abs(x - pxy[0]) > 1e-6 or abs(y - pxy[1]) > 1e-6:
                ax.annotate("", xy=(x, y), xytext=pxy,
                            arrowprops=dict(arrowstyle="-|>", color="#c62828",
                                            lw=0.5 + 0.55 * np.sqrt(n),
                                            alpha=0.55, shrinkA=6, shrinkB=3),
                            zorder=4)
        for (x, y), n in cells.items():
            ax.scatter(x, y, s=22 + 16 * np.sqrt(n), facecolor="#c62828",
                       edgecolor="white", lw=0.8, alpha=0.9, zorder=5)
            if n >= 4:
                ax.annotate(f"{n}", (x, y), fontsize=6.2, ha="center",
                            va="center", color="white", fontweight="bold",
                            zorder=6)
        if pxy is not None:
            ax.scatter(*pxy, s=200, marker="*", facecolor="#1f4e9c",
                       edgecolor="white", lw=1.0, zorder=7)
            stay = sum(n for (x, y), n in cells.items()
                       if abs(x - pxy[0]) < 1e-6 and abs(y - pxy[1]) < 1e-6)
            note = f"{stay} stay put"
        else:
            ax.annotate("parent ergodic:\nno descriptor", (0.5, 0.93),
                        xycoords="axes fraction", ha="center", va="top",
                        fontsize=6.8, color="#777777")
            note = "no origin"
        tup = "".join(rules.wolfram_to_tuple(parent))
        shown = f"{len(kids)}" if len(kids) == n_all else f"{len(kids)}/{n_all}"
        ax.set_title(f"W{parent}  {tup}   ({shown} children, {note})",
                     fontsize=8.5)
        ax.set_xlim(0.90, 2.12)
        ax.set_ylim(0.88, 2.14)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cl):]:
        ax.axis("off")
    for ax in axes[:len(cl)]:
        ax.set_xlabel(r"base $a$", fontsize=8)
        ax.set_ylabel(r"base $b$", fontsize=8)
    hs = [plt.Line2D([], [], marker="*", ls="", markersize=12,
                     markerfacecolor="#1f4e9c", markeredgecolor="white",
                     label="coherent rule (resets off)"),
          plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor="#c62828", markeredgecolor="white",
                     label="its V+reset children"),
          plt.Line2D([], [], color="#444444", lw=1.0, ls=":", label=r"$ab=2$")]
    fig.legend(handles=hs, fontsize=8, loc="lower right", ncol=3,
               bbox_to_anchor=(0.99, 0.005))
    fig.suptitle(f"F6  where dissipation takes a rule, in the {lab} plane "
                 f"({bc}).  Clusters are the 160 V+reset rules grouped by "
                 f"coherent correspondent (D, E $\\to$ I).",
                 fontsize=9.5, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
