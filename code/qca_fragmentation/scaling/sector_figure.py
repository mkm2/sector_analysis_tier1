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


#: The open-system fragmented rules get a gold halo and a label wherever they
#: appear: they are the physically interesting minority and would otherwise be
#: eight anonymous dots among 256.
HALO = "#e8a33d"


def _highlight(ax, d, xy, *, label=True, ring=210, fam=None):
    """
    Ring and name the open-system fragmented rules.

    `fam` MUST be supplied whenever the markers were drawn by _scatter_cells,
    because that function nudges each family sideways by _FAM_DX to stop the
    three families hiding under one another -- and a ring drawn at the true
    coordinate then misses the dot it is meant to circle.  All eight fragmented
    rules are "mixed" (dx = +0.019), so before this argument existed every ring
    in F1 sat 0.019 to the LEFT of its own marker: the ring at W29's quoted
    (1.2147, 1.8059) was empty and W29's dot was over at 1.234, inside W71's
    ring.  Found 2026-08-04 from the report's own coordinates.
    """
    frag = sectors.open_system_fragmented(d)
    if fam is not None:
        xy = {r: (x + _FAM_DX[fam[r]], y) for r, (x, y) in xy.items()
              if r in fam}
    for r in frag:
        pt = xy.get(r["rule"])
        if pt is None:
            continue
        ax.scatter(*pt, s=ring, marker="o", facecolor="none",
                   edgecolor=HALO, linewidth=2.0, alpha=0.95, zorder=9)
    if label:
        seen = {}
        for r in frag:
            pt = xy.get(r["rule"])
            if pt is None:
                continue
            seen.setdefault((round(pt[0], 3), round(pt[1], 3)), []).append(r["rule"])
        # W29/W71 and W73/W109 sit within 0.015 of each other, so labels are
        # pushed apart vertically in order of y rather than overprinting.
        items = sorted(seen.items(), key=lambda kv: (-kv[0][1], kv[0][0]))
        placed = []
        for (x, y), rs in items:
            dy = -4
            while any(abs(x - px) < 0.09 and abs((y + dy * 0.006)
                                                - (py + pdy * 0.006)) < 0.030
                      for px, py, pdy in placed):
                dy += 13
            placed.append((x, y, dy))
            ax.annotate("W" + "/".join(str(v) for v in sorted(rs)), (x, y),
                        textcoords="offset points", xytext=(12, dy),
                        fontsize=7.0, color="#8a5a12", zorder=10,
                        fontweight="bold",
                        arrowprops=(dict(arrowstyle="-", lw=0.5, color=HALO)
                                    if dy != -4 else None))
    return frag


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
    frag = _highlight(axes[0], d,
                      {p["rule"]: (p["n_wcc"]["base"], p["d_max_wcc"]["base"])
                       for p in d["points"]
                       if p["n_wcc"]["base"] is not None}, fam=fam)
    _highlight(axes[1], d,
               {x["rule"]: (x["a_att"], x["b_att"])
                for x in d["attractor_deficits"]}, label=False, fam=fam)
    for r, lab, off in ((204, "204", (-24, 6)), (51, "51", (14, 12)),
                        (150, "150", (14, -14)), (156, "156", (12, 4))):
        p = next((x for x in d["points"] if x["rule"] == r), None)
        if p and p["n_wcc"]["base"] is not None:
            axes[0].annotate(lab, (p["n_wcc"]["base"] + _FAM_DX[p["family"]],
                                   p["d_max_wcc"]["base"]),
                             textcoords="offset points", xytext=off,
                             fontsize=7.5, color=TEXT, zorder=21,
                             arrowprops=dict(arrowstyle="-", lw=0.5,
                                             color="#888888"))

    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=_EMPH[k]["color"], markeredgecolor="white",
                     label=_FAMLABEL[k]) for k in ("unitary", "mixed", "classical")]
    hs.append(plt.Line2D([], [], color="#444444", lw=1.0, ls=":",
                         label=r"$ab=2$ (excluded below, left panel only)"))
    hs.append(plt.Line2D([], [], marker="o", ls="", markersize=9,
                         markerfacecolor="none", markeredgecolor=HALO,
                         markeredgewidth=2.0,
                         label=f"fragmented open systems ({len(frag)})"))
    axes[0].legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)
    # R16: the sideways nudge is cosmetic, but at the top-left corner it is also
    # misleading -- b = 2/a rises as a falls, so the V-free cell that sits exactly
    # ON the curve at (1, 2) is drawn at a = 0.981, where the curve is 2.0387, and
    # therefore appears under it.  Say so rather than let a reader count it as a
    # violation; the verdict table reports zero.
    fig.suptitle("Marker area grows with the number of rules stacked in a cell; "
                 "the count is printed inside crowded cells.  Families are nudged "
                 r"by $\pm0.019$ in $a$, so the V-free cell at the top left is "
                 "exactly ON the curve, not below it (R16).  "
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
    for which in ("sector", "attractor"):
        fig_corner(bc, os.path.join(FIGURES_DIR,
                                    f"fig_{which}_corner_{bc}.pdf"), which, d)
    fig_recurrent_transient(bc, os.path.join(FIGURES_DIR,
                                             f"fig_rec_transient_{bc}.pdf"))
    # base(n_wcc) lives in the sector map, base(n_scc) in the attractor point;
    # the two have to be married per rule before they can be plotted against
    # each other.
    sec = {p["rule"]: p for p in d["points"]}

    def _pt(r):
        a, b = sec.get(r), sectors.attractor_point(r, bc)
        if a is None or b is None:
            return None
        return {"family": a["family"], "n_wcc": a["n_wcc"],
                "n_recurrent": b["n_recurrent"]}

    rows = wcc_scc_rows(range(256),
                        lambda r: sectors.load_series(r, bc,
                                                      sectors.UNIFORM_N_CAP),
                        _pt)
    for which, t in (("count", "F11"), ("base", "F12")):
        fig_wcc_vs_scc(rows, os.path.join(
            FIGURES_DIR, f"fig_wcc_vs_scc_{which}_{bc}.pdf"), which=which,
            tag=t, gate="Hadamard")
    st = fig_validation(bc, os.path.join(FIGURES_DIR,
                                         f"fig_validation_{bc}.pdf"), d)
    with open(os.path.join(sectors.ANALYTICS,
                           f"sector_validation_{bc}.json"), "w") as f:
        json.dump(st, f)
    print(json.dumps({k: v for k, v in st.items()
                      if k != "hyperbola_margins"}, indent=1))




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
        _highlight(ax, d, {c: v for c, v in kids}, label=False, ring=150)
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



# --- F7/F8: the corner view, as in R2 fig.6 ----------------------------------
#
# R2's growth map folds a base-vs-base main panel together with the two
# sub-leading powers in the margins: the bottom margin shares the main panel's
# x-axis (the base of the COUNT) and plots that series' alpha beneath it; the
# left margin shares the y-axis (the base of D_max) and plots ITS alpha beside
# it.  A rule keeps its horizontal position between main and bottom and its
# vertical position between main and left, so one point carries the full
# (base, alpha) description of both series.  The same layout is worth having for
# the Tier-1e sector map and for the monitored-attractor map, because the base
# plane alone cannot distinguish "one giant sector" from "a linear number of
# them" -- both sit at a = 1 (R9 sec.6.4).

REF_BASES = [(2 ** 0.5, r"$\sqrt{2}$"), (1.32472, r"$\rho$"),
             (1.46557, r"$\psi$"), ((1 + 5 ** 0.5) / 2, r"$\varphi$"),
             (3 ** 0.5, r"$\sqrt{3}$"), (2.0, "$2$")]

_CORNER_BASE_LIM = (0.955, 2.075)
_CORNER_ACOUNT_LIM = (-0.35, 1.35)
_CORNER_ADMAX_LIM = (-0.95, 2.05)


def _corner_rows(bc: str, which: str, data: Optional[Dict] = None):
    """(b_count, a_count, b_size, a_size, family, exact_count, exact_size)."""
    out = []
    if which == "sector":
        d = data or sectors.load(bc) or sectors.build(bc)
        for p in d["points"]:
            dn, dd = p["n_wcc"], p["d_max_wcc"]
            if dn["base"] is None or dd["base"] is None:
                continue
            out.append((dn["base"], dn["alpha"], dd["base"], dd["alpha"],
                        p["family"], dn["exact"], dd["exact"], p["rule"]))
    else:
        for rule in range(256):
            p = sectors.attractor_point(rule, bc)
            if p is None:
                continue
            dn, dd = p["n_recurrent"], p["d_max"]
            if dn["base"] is None or dd["base"] is None:
                continue
            out.append((dn["base"], dn["alpha"], dd["base"], dd["alpha"],
                        p["family"], dn["exact"], dd["exact"], rule))
    return out


def _jit(rule: int, k: int) -> float:
    """A deterministic sub-pixel nudge so coincident rules stay countable."""
    return ((rule * (37 if k == 0 else 53)) % 7 - 3) * 0.004


def _corner_scatter(ax, xs, ys, fams, fills, rs, s=30, colours=None):
    colours = colours or FAMILY_COLOUR
    for x, y, f, fill, r in zip(xs, ys, fams, fills, rs):
        c = colours[f]
        ax.scatter(x, y, s=s, marker="o", facecolor=c if fill else "none",
                   edgecolor=c, linewidth=0.9, alpha=0.85 if fill else 0.6,
                   zorder=3)


def fig_corner(bc: str, out: str, which: str = "sector",
               data: Optional[Dict] = None, rows=None,
               colours: Optional[Dict] = None, labels: Optional[Dict] = None,
               tag: Optional[str] = None, note: Optional[str] = None,
               order: Optional[tuple] = None):
    """
    The corner layout.  R9 calls it with its own two maps; R10 calls it with the
    X-gate rows and its own family palette.  ONE implementation, deliberately:
    R2 and R9 drew this same layout from two separate code paths and silently
    disagreed on every alpha (R9 sec.6.5), which is exactly what a shared
    function prevents.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    rows = _corner_rows(bc, which, data) if rows is None else list(rows)
    colours = colours or FAMILY_COLOUR
    labels = labels or FAMILY_LABEL
    order = order or ("unitary", "classical", "mixed")
    bn = [r[0] + _jit(r[7], 0) for r in rows]
    an = [r[1] + _jit(r[7], 1) for r in rows]
    bd = [r[2] + _jit(r[7], 1) for r in rows]
    ad = [r[3] + _jit(r[7], 0) for r in rows]
    fam = [r[4] for r in rows]
    en = [r[5] for r in rows]
    ed = [r[6] for r in rows]
    rr = [r[7] for r in rows]

    fig = plt.figure(figsize=(9.8, 8.8))
    gs = GridSpec(2, 2, width_ratios=[1.05, 3.0], height_ratios=[3.0, 1.05],
                  wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")
    for a in (ax_main, ax_left, ax_bot):
        _style(a)

    for v, _ in REF_BASES:
        ax_main.axvline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
        ax_main.axhline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    if which == "sector":
        _hyperbola(ax_main, 0.96, 2.07)
    else:
        a = np.linspace(0.96, 2.07, 400)
        ax_main.plot(a, 2.0 / a, color="#444444", lw=1.0, ls=":", zorder=2,
                     alpha=0.45)
    _corner_scatter(ax_main, bn, bd, fam, [x and y for x, y in zip(en, ed)], rr,
                    colours=colours)
    ax_main.set_xlim(*_CORNER_BASE_LIM)
    ax_main.set_ylim(*_CORNER_BASE_LIM)
    ax_main.tick_params(labelbottom=False, labelleft=False)
    for v, name in REF_BASES:
        ax_main.annotate(name, (v, _CORNER_BASE_LIM[1]), xytext=(0, 1),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=7, color=MUTED, annotation_clip=False)
        ax_main.annotate(name, (_CORNER_BASE_LIM[1], v), xytext=(2, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=7, color=MUTED, annotation_clip=False)

    for v, _ in REF_BASES:
        ax_bot.axvline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax_bot.axhline(0, color=MUTED, lw=0.8, zorder=1)
    _corner_scatter(ax_bot, bn, an, fam, en, rr, colours=colours)
    ax_bot.set_ylim(*_CORNER_ACOUNT_LIM)
    lbl = r"n_{\rm wcc}" if which == "sector" else r"n_{\rm rec}"
    ax_bot.set_xlabel(rf"growth base of ${lbl}$")
    ax_bot.set_ylabel(rf"$\alpha_{{{lbl}}}$", labelpad=1)

    for v, _ in REF_BASES:
        ax_left.axhline(v, color=MUTED, lw=0.5, ls=":", alpha=0.5, zorder=1)
    ax_left.axvline(0, color=MUTED, lw=0.8, zorder=1)
    _corner_scatter(ax_left, ad, bd, fam, ed, rr, colours=colours)
    ax_left.set_xlim(*_CORNER_ADMAX_LIM)
    ax_left.invert_xaxis()
    ax_left.set_ylabel(r"growth base of $D_{\max}$")
    ax_left.set_xlabel(r"$\alpha_{D_{\max}}$")

    hs = [Line2D([0], [0], marker="o", ls="none", color=colours[k],
                 markersize=8, label=labels[k])
          for k in order if k in colours]
    hs += [Line2D([0], [0], marker="o", ls="none", color="#555",
                  markerfacecolor="#555", markersize=7, label="exact base"),
           Line2D([0], [0], marker="o", ls="none", color="#555",
                  markerfacecolor="none", markersize=7, label="fitted base")]
    leg = ax_leg.legend(handles=hs, frameon=False, fontsize=8.5,
                        loc="center left", bbox_to_anchor=(-0.02, 0.5),
                        title="family / base")
    leg.get_title().set_fontsize(8.5)

    if tag is None:
        tag = ("F8  sectors (WCC)" if which == "sector"
               else "F9  monitored attractors (terminal SCC)")
    if note is None:
        note = ("exact for both experiments" if which == "sector"
                else "monitored only; the dotted curve is a reference, "
                     "not a bound")
    fig.suptitle(f"{tag} ({bc}): base--base core with $\\alpha$ margins "
                 f"--- {note}", fontsize=11, x=0.5, y=0.995, color=TEXT)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --- F9: terminal SCCs against transient states ------------------------------

def fig_recurrent_transient(bc: str, out: str, rows=None,
                            colours: Optional[Dict] = None,
                            labels: Optional[Dict] = None,
                            order=("unitary", "classical", "mixed"),
                            tag: str = "F10", gate: str = ""):
    """
    Where the mass sits.  The recurrent set is the union of the terminal SCCs;
    everything else is transient.  Left: the growth base of the number of
    terminal SCCs against the growth base of the recurrent MASS |Rec|.  A rule at
    mass-base 2 keeps a finite fraction of the basis recurrent; anything below is
    exponentially transient-dominated, and the unitary rules sit at exactly 2
    with no transient at all.  Right: the recurrent fraction itself at the
    largest N, which is the same statement without a fit.
    """
    if rows is None:
        rows = [sectors.recurrent_mass(r, bc) for r in range(256)]
    rows = [r for r in rows if r and r["mass"]["base"] is not None
            and r["count"]["base"] is not None]
    colours = colours or FAMILY_COLOUR
    labels = labels or FAMILY_LABEL
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    ax = axes[0]
    _style(ax)
    ax.axhline(2.0, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.plot([1, 2], [1, 2], color=MUTED, lw=0.7, ls=":", alpha=0.6, zorder=1)
    from collections import Counter
    cells: Counter = Counter()
    for r in rows:
        cells[(round(r["count"]["base"], 3), round(r["mass"]["base"], 3),
               r["family"])] += 1
    for (x, y, f), n in cells.items():
        ax.scatter(x, y, s=26 + 16 * np.sqrt(n), facecolor=colours[f],
                   edgecolor="white", lw=0.9, alpha=0.9,
                   zorder=5 if f == order[0] else 4)
    ax.annotate("no transient at all ($|{\\rm Rec}|=2^N$)", (1.35, 2.0),
                xytext=(0, 5), textcoords="offset points", fontsize=7.5,
                color=MUTED, ha="center")
    ax.set_xlabel(r"growth base of the number of terminal SCCs")
    ax.set_ylabel(r"growth base of the recurrent mass $|{\rm Rec}|$")
    ax.set_xlim(0.95, 2.08)
    ax.set_ylim(0.95, 2.10)
    ax.set_title(f"{tag}a  {gate}terminal SCCs against transient states:\n"
                 "how much of the basis survives to late times", fontsize=9.5)
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=colours[k], markeredgecolor="white",
                     label=labels[k])
          for k in order if k in colours]
    ax.legend(handles=hs, fontsize=7.5, loc="lower left", framealpha=0.95)

    ax = axes[1]
    _style(ax)
    for f in order:
        v = sorted(r["recurrent_fraction"] for r in rows if r["family"] == f)
        if not v:
            continue
        ax.plot(np.arange(len(v)) / max(len(v) - 1, 1), v, "o", ms=3.5,
                color=colours[f], alpha=0.85,
                label=f"{labels[f]}  (median {np.median(v):.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("rules of the family, sorted")
    ax.set_ylabel(r"recurrent fraction $|{\rm Rec}|/2^N$ at $N_{\max}$")
    ax.set_title(f"{tag}b  the same, unfitted: the share of the basis that "
                 "is\nrecurrent at the largest $N$ computed", fontsize=9.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows




# --- WCC against SCC: the two components, compared directly ------------------
#
# The maps of F1 put the SECTOR plane and the ATTRACTOR plane side by side, and
# F3 draws the move between them.  Neither answers the blunter question: for one
# rule, how does the number of weak components compare with the number of
# terminal SCCs?  Two views, because the two comparisons are different claims.
#
#   count  the raw numbers at N_max.  Every weak component contains at least one
#          terminal SCC, so n_scc >= n_wcc pointwise -- an exact inequality with
#          no fit in it, and the plot is a check of it as much as a display.
#   base   the growth bases.  This is the asymptotic version of the same
#          comparison, and it can sit ON the diagonal while the counts do not:
#          a fixed number of attractors per sector cancels out of the base.

def wcc_scc_rows(rules_iter, series_fn, point_fn, n_cap=None):
    """
    [(rule, family, n_wcc, n_scc, base_wcc, base_scc)] from any Tier-1e-shaped
    source.  series_fn(rule) -> {"N", "n_wcc", "n_recurrent"}; point_fn(rule) ->
    {"family", "n_wcc": {...}, "n_recurrent": {...}} or None.
    """
    out = []
    for rule in rules_iter:
        p = point_fn(rule)
        if p is None:
            continue
        try:
            s = series_fn(rule)
        except Exception:
            continue
        if not s or not s.get("N"):
            continue
        i = len(s["N"]) - 1
        nw, nr = s["n_wcc"][i], s["n_recurrent"][i]
        if not nw or not nr:
            continue
        out.append({"rule": rule, "family": p["family"], "N": s["N"][i],
                    "n_wcc": nw, "n_scc": nr,
                    "base_wcc": p["n_wcc"]["base"],
                    "base_scc": p["n_recurrent"]["base"]})
    return out


def fig_wcc_vs_scc(rows, out, which="count", colours=None, labels=None,
                   order=("unitary", "classical", "mixed"), tag="",
                   gate="Hadamard"):
    from collections import Counter
    colours = colours or FAMILY_COLOUR
    labels = labels or FAMILY_LABEL
    rows = [r for r in rows
            if (r["base_wcc"] is not None and r["base_scc"] is not None)
            or which == "count"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3))

    ax = axes[0]
    _style(ax)
    if which == "count":
        xs = [r["n_wcc"] for r in rows]
        ys = [r["n_scc"] for r in rows]
        lim = (0.8, 1.35 * max(max(xs), max(ys)))
        ax.plot(lim, lim, color="#444444", lw=1.0, ls=":", zorder=2)
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(*lim), ax.set_ylim(*lim)
        ax.set_xlabel(r"weak components $n_{\rm wcc}$ at $N_{\max}$")
        ax.set_ylabel(r"terminal SCCs $n_{\rm scc}$ at $N_{\max}$")
        cells = Counter((r["n_wcc"], r["n_scc"], r["family"]) for r in rows)
    else:
        xs = [r["base_wcc"] for r in rows]
        ys = [r["base_scc"] for r in rows]
        ax.plot([0.95, 2.1], [0.95, 2.1], color="#444444", lw=1.0, ls=":",
                zorder=2)
        ax.set_xlim(0.95, 2.08), ax.set_ylim(0.95, 2.08)
        ax.set_xlabel(r"growth base of $n_{\rm wcc}$")
        ax.set_ylabel(r"growth base of $n_{\rm scc}$")
        cells = Counter((round(r["base_wcc"], 3), round(r["base_scc"], 3),
                         r["family"]) for r in rows)
    for (x, y, f), n in sorted(cells.items(), key=lambda kv: kv[1]):
        ax.scatter(x, y, s=26 + 17 * np.sqrt(n), facecolor=colours[f],
                   edgecolor="white", lw=0.9, alpha=0.9,
                   zorder=6 if f == order[0] else 4)
        if n >= 8:
            ax.annotate(f"{n}", (x, y), fontsize=6.2, ha="center", va="center",
                        color="white", fontweight="bold", zorder=7)
    on = sum(n for (x, y, _), n in cells.items() if abs(x - y) < 1e-9)
    ax.set_title(f"{tag}a  {gate} gate: weak components against terminal SCCs "
                 f"({'raw counts' if which == 'count' else 'growth bases'})\n"
                 f"{on} of {sum(cells.values())} rules sit exactly on the "
                 f"diagonal", fontsize=9.5)
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                     markerfacecolor=colours[k], markeredgecolor="white",
                     label=labels[k]) for k in order if k in colours]
    ax.legend(handles=hs, fontsize=7.5, loc="upper left", framealpha=0.95)

    ax = axes[1]
    _style(ax)
    for f in order:
        if which == "count":
            v = sorted(r["n_scc"] / r["n_wcc"] for r in rows
                       if r["family"] == f)
            lab = r"$n_{\rm scc}/n_{\rm wcc}$"
        else:
            v = sorted(r["base_scc"] - r["base_wcc"] for r in rows
                       if r["family"] == f)
            lab = r"base$(n_{\rm scc})-$base$(n_{\rm wcc})$"
        if not v:
            continue
        ax.plot(np.arange(len(v)) / max(len(v) - 1, 1), v, "o", ms=3.5,
                color=colours[f], alpha=0.85,
                label=f"{labels[f]}  (median {np.median(v):.3g})")
    if which == "count":
        ax.set_yscale("log")
        ax.axhline(1.0, color="#444444", ls=":", lw=1.0)
    else:
        ax.axhline(0.0, color="#444444", ls=":", lw=1.0)
    ax.set_xlabel("rules of the family, sorted")
    ax.set_ylabel(lab)
    ax.set_title(f"{tag}b  the same as a per-rule ratio, each family sorted:\n"
                 "how many attractors a sector carries", fontsize=9.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for p_ in (out, out.replace(".pdf", ".png")):
        fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows


if __name__ == "__main__":
    main()
