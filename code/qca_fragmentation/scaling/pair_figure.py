"""
Tier-1d figures (task Sec.5): the monitored-vs-unmonitored certificate picture.

  fig_pair_sandwich_{bc}.pdf     per coherent-core rule, the growth-base sandwich
                                 [b_{K_M} .. b_{|P_rec|}] with the exact
                                 within-sector dim-Fix base marked inside it.
  fig_certificate_coverage_{bc}  fraction of dissipative units certified exact
                                 for the unmonitored channel, per N.
  fig_coherence_parity_{bc}      # rules with protected coherence per N, split by
                                 N-parity -- resolves the census non-monotonicity.

Design follows figure.py / diss_figure.py: CVD-safe hues, redundant marker
shapes, recessive grid, direct labels.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from .figure import MUTED, TEXT, _style
from .pair_scaling import (TABLE_RULES, certificate_coverage, parity_resolution,
                           summarize)

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

LOWER = "#8a8a86"     # K_M lower bound
EXACT = "#2a78d6"     # within-sector dim Fix (exact, N<=6)
UPPER = "#e34948"     # |P_rec| upper bound


def fig_pair_sandwich(rows: List[Dict], bc: str, out: str) -> int:
    sel = [r for r in rows if r["bc"] == bc and r["rule"] in TABLE_RULES]
    sel = [r for r in sel if r["km_base"] is not None
           or r["fix_upper_base"] is not None]
    sel.sort(key=lambda r: (r["fix_upper_base"] or r["km_base"] or 0))
    if not sel:
        return 0
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _style(ax)
    ys = list(range(len(sel)))
    for y, r in zip(ys, sel):
        lo, ex, up = r["km_base"], r["cesaro_base"], r["fix_upper_base"]
        if lo is not None and up is not None:
            ax.plot([lo, up], [y, y], color=MUTED, lw=1.4, zorder=1)
        if lo is not None:
            ax.scatter(lo, y, s=48, marker="|", color=LOWER, zorder=3,
                       label="$b_{K_M}$ (lower)" if y == ys[0] else None)
        if up is not None:
            ax.scatter(up, y, s=48, marker="|", color=UPPER, zorder=3,
                       label="$b_{|P_{rec}|}$ (upper)" if y == ys[0] else None)
        if ex is not None:
            ax.scatter(ex, y, s=42, marker="D", color=EXACT,
                       edgecolor="white", linewidth=0.5, zorder=4,
                       label="$b_{\\mathrm{Fix}}$ (exact, $N\\leq 6$)"
                       if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"W{r['rule']} {r['tuple']}" for r in sel], fontsize=8)
    ax.set_xlabel("growth base")
    ax.set_title(f"Coherent-core growth-base sandwich ({bc})",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(sel)


def fig_certificate_coverage(bc: str, out: str) -> int:
    cov = certificate_coverage(bc)
    if not cov:
        return 0
    Ns = sorted(int(n) for n in cov)
    frac = [cov[str(n)]["certified"] / max(cov[str(n)]["total"], 1) for n in Ns]
    tot = [cov[str(n)]["total"] for n in Ns]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _style(ax)
    ax.plot(Ns, frac, "-o", color=EXACT, lw=1.6, ms=5, zorder=3)
    for n, f, tt in zip(Ns, frac, tot):
        ax.annotate(f"{int(round(f*tt))}/{tt}", (n, f),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7.5, color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("chain length $N$")
    ax.set_ylabel("fraction of dissipative units certified exact")
    ax.set_title(f"Unmonitored-channel certificate coverage ({bc})",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(Ns)


def fig_coherence_parity(bc: str, out: str) -> int:
    pr = parity_resolution(bc)
    even = {int(n): c for n, c in pr["exact_even"].items()}
    odd = {int(n): c for n, c in pr["exact_odd"].items()}
    if not even and not odd:
        return 0
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _style(ax)
    if even:
        xs = sorted(even)
        ax.plot(xs, [even[n] for n in xs], "-s", color=EXACT, lw=1.6, ms=6,
                label="even $N$ (exact)", zorder=3)
    if odd:
        xs = sorted(odd)
        ax.plot(xs, [odd[n] for n in xs], "-^", color=UPPER, lw=1.6, ms=6,
                label="odd $N$ (exact)", zorder=3)
    # the even-only Tier-1b dense census, for comparison
    dense = {4: 84, 6: 60, 8: 72}
    ax.plot(sorted(dense), [dense[n] for n in sorted(dense)], ":o",
            color=LOWER, lw=1.2, ms=5, mfc="none",
            label="Tier-1b dense census (even only)", zorder=2)
    ax.set_xlabel("chain length $N$")
    ax.set_ylabel("# rules with within-sector coherence")
    ax.set_title(f"Coherence census resolved by $N$-parity ({bc})",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(even) + len(odd)


def _pair_status_by_rule(bc: str) -> Dict[int, Dict]:
    """certified / coherent status at the largest computed N, per rule."""
    out = {}
    for f in (os.listdir(results_io.PAIR_RESULTS_DIR)
              if os.path.isdir(results_io.PAIR_RESULTS_DIR) else []):
        if not f.endswith(f"_{bc}.jsonl"):
            continue
        recs = results_io.load_pair_results(int(f.split("_")[0]), bc)
        if not recs:
            continue
        r = recs[max(recs)]
        out[r["rule"]] = {
            "certified": r.get("certified"),
            "coherent": (r.get("pair_offdiag") or 0) > 0,
            "bounded_only": r.get("bounded_only"),
        }
    return out


def fig_growth_certified(bc: str, out: str) -> int:
    """The dissipative growth-rate plane, coloured by the unmonitored-channel
    certificate: filled = certified exact, hollow ring = protected coherence."""
    from .dissipative import summarize as diss_summarize
    from .diss_figure import _points, _jitter
    rows = diss_summarize((bc,))
    pts = _points(rows, bc)
    status = _pair_status_by_rule(bc)
    if not pts or not status:
        return 0
    dx, dy = _jitter(pts)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _style(ax)
    seen = set()
    for (x, y, r), ddx, ddy in zip(pts, dx, dy):
        st = status.get(r["rule"])
        if st is None:
            continue
        if st["certified"] is True:
            col, lbl, face, ring = EXACT, "certified exact", True, False
        elif st["coherent"]:
            col, lbl, face, ring = UPPER, "protected coherence", False, True
        else:
            col, lbl, face, ring = LOWER, "uncertified (interval)", True, False
        ax.scatter(x + ddx, y + ddy, s=40 if not ring else 54, marker="o",
                   facecolor="none" if not face else col,
                   edgecolor=col, linewidth=1.4 if ring else 0.5,
                   alpha=0.85, zorder=4 if ring else 3,
                   label=lbl if lbl not in seen else None)
        seen.add(lbl)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$\kappa$ of the attractor count")
    ax.set_ylabel(r"$\kappa$ of the largest attractor $D_{\max}$")
    ax.set_title(f"Dissipative growth plane by channel certificate ({bc})",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(pts)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="pbc")
    args = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = summarize((args.bc,))
    a = fig_pair_sandwich(rows, args.bc,
                          os.path.join(FIGURES_DIR, f"fig_pair_sandwich_{args.bc}.pdf"))
    b = fig_certificate_coverage(args.bc,
                                 os.path.join(FIGURES_DIR,
                                              f"fig_certificate_coverage_{args.bc}.pdf"))
    c = fig_coherence_parity(args.bc,
                             os.path.join(FIGURES_DIR,
                                          f"fig_coherence_parity_{args.bc}.pdf"))
    d = fig_growth_certified(args.bc,
                             os.path.join(FIGURES_DIR,
                                          f"fig_growth_certified_{args.bc}.pdf"))
    print(f"{args.bc}: sandwich {a} rules, coverage {b} sizes, parity {c} "
          f"points, growth-certified {d} rules")
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
