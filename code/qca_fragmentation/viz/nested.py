"""
Nested fragmentation: can a rule fragment twice?

THE QUESTION.  A rule fragments once when the number of sectors (weak
components) grows exponentially with N.  It fragments a second time, INSIDE a
sector, when one weak component contains exponentially many terminal SCCs.  Only
the thirteen rules of R9 sec.7.3 can do the second thing at all --- every other
rule has exactly one attractor per sector.  So: is there a rule that does both?

The answer is PARTLY.  Eight rules fragment exponentially inside a sector while
their sector count stays constant or linear -- that half is settled, by exact
integer recurrences.  The fully nested case (exponential BOTH ways) has exactly
one candidate, rule 29, and at the sizes reachable here a quadratic and a slow
exponential fit its in-sector count equally well, so it is left open rather than
decided.  Rule 71, its near-twin, is exactly linear inside a sector and is
therefore definitely not nested.

WHAT IS MEASURED, per rule and per N:
  n_wcc      number of sectors
  n_att      number of terminal SCCs in total
  max_att    the most attractors any single sector contains
  big_att    attractors inside the LARGEST sector
  dist       attractors per sector, descending -- the prettiest of the four
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from . import spy

ANALYTICS = os.path.join(results_io.REPO_ROOT, "analytics")
STORE = os.path.join(ANALYTICS, "nested_fragmentation_{bc}.json")

#: R9 sec.7.3's twelve, plus rule 37 which is an exception only at N = 2 (mod 3).
MULTI_RULES = (203, 217, 219, 36, 44, 100, 104, 233, 29, 71, 235, 249, 37)

#: The three exact recurrences the total attractor counts obey.  All are
#: "Narayana-like": a linear recurrence with two unit coefficients and no others.
FAMILIES = {
    (1, 0, 1): ("supergolden $\\psi$", 1.4655712318767682,
                "a_n = a_{n-1} + a_{n-3}"),
    (1, 0, 0, 1): ("$x^4=x^3+1$", 1.3802775690976143,
                   "a_n = a_{n-1} + a_{n-4}"),
    (0, 1, 1): ("plastic $\\rho$", 1.3247179572447454,
                "a_n = a_{n-2} + a_{n-3}"),
}


def census(rule: int, Ns: Sequence[int], bc: str = "obc0") -> Dict:
    """The four series, plus the per-sector distribution at the largest N."""
    nw, natt, mx, big, dist = [], [], [], [], None
    for N in Ns:
        sb, term = spy.scc_blocks(rule, N, bc)
        wb = spy.wcc_blocks(rule, N, bc)          # largest sector first
        own = spy.block_of(wb, 1 << N)
        per = Counter(own[b[0]] for b, t in zip(sb, term) if t)
        nw.append(len(wb))
        natt.append(sum(term))
        mx.append(max(per.values()) if per else 0)
        big.append(per.get(0, 0))
        dist = sorted(per.values(), reverse=True)
    return {"rule": rule, "bc": bc, "N": list(Ns), "n_wcc": nw,
            "n_att": natt, "max_att": mx, "big_att": big, "dist": dist}


def build(Ns=range(6, 17), bc: str = "obc0", rules_in=MULTI_RULES) -> Dict:
    out = {"bc": bc, "N": list(Ns),
           "rows": [census(r, Ns, bc) for r in rules_in]}
    os.makedirs(ANALYTICS, exist_ok=True)
    with open(STORE.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = STORE.format(bc=bc)
    return json.load(open(p)) if os.path.exists(p) else None


# --- classification ----------------------------------------------------------

def growth(Ns: Sequence[int], ys: Sequence[int], rule: int = 0,
           bc: str = "obc0", key: str = "y") -> Dict:
    """
    Growth class of one series, using the project's parity-aware descriptor.

    Several of these series oscillate with N (rule 29's sector count runs
    3, 6, 5, 9, 7, 14, ...), so a naive ratio test misreads them in both
    directions -- which is exactly why the shared descriptor is used here rather
    than a fresh heuristic.
    """
    from ..scaling import sectors as S
    d = S.series_descriptor(rule, bc, key, list(Ns), list(ys))
    if d is None:
        return {"cls": "unknown", "base": None}
    return {"cls": d["cls"], "base": d["base"], "exact": d["exact"],
            "alpha": d["alpha"]}


def classify(row: Dict) -> Dict:
    """(sector growth, in-sector growth) for one rule, and the joint verdict."""
    Ns = row["N"]
    a = growth(Ns, row["n_wcc"], row["rule"], row["bc"], "n_wcc")
    b = growth(Ns, row["max_att"], row["rule"], row["bc"], "max_att")
    t = growth(Ns, row["n_att"], row["rule"], row["bc"], "n_att")
    both = a["cls"] == "exponential" and b["cls"] == "exponential"
    return {"rule": row["rule"], "sectors": a, "in_sector": b, "total": t,
            "doubly_exponential_fragmentation": both}


def exclusion_check(bc: str = "obc0", data: Optional[Dict] = None) -> Dict:
    """
    Which rules fragment exponentially in the sector count, which inside a
    sector, and whether any does both.

    Rules outside MULTI_RULES have one attractor per sector, so their in-sector
    count is identically 1 and cannot grow; the check therefore only has to
    inspect the thirteen, which it does explicitly.

    NOTE.  `violations` is what the shared descriptor says, and for rule 29 that
    verdict is not trustworthy on its own: its in-sector series oscillates with
    parity and is superlinear but short, and on the even branch (N = 6..20)
    a quadratic and an exponential of base 1.12 fit it to within 6% with
    essentially equal BIC.  R13 treats rule 29 as OPEN.  See
    `rule29_model_comparison`.
    """
    d = data or load(bc) or build(bc=bc)
    rows = [classify(r) for r in d["rows"]]
    bad = [r["rule"] for r in rows if r["doubly_exponential_fragmentation"]]
    exp_sectors = [r["rule"] for r in rows if r["sectors"]["cls"] == "exponential"]
    exp_inside = [r["rule"] for r in rows
                  if r["in_sector"]["cls"] == "exponential"]
    return {"bc": bc, "rows": rows, "violations": bad,
            "exponential_sectors": exp_sectors,
            "exponential_in_sector": exp_inside,
            "disjoint": not (set(exp_sectors) & set(exp_inside))}


def narayana(n: int) -> List[int]:
    """1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, ... -- a(k) = a(k-1) + a(k-3)."""
    a = [1, 1, 1]
    while len(a) < n:
        a.append(a[-1] + a[-3])
    return a[:n]


# --- figures -----------------------------------------------------------------

FAM = {"psi": "#c62828", "x4": "#e08a1e", "rho": "#1f4e9c", "flat": "#8a8f98"}


def _fig_dir():
    import os
    return os.path.join(results_io.REPO_ROOT, "figures")


def fig_plane(bc="obc0", out=None, data=None):
    """
    The two fragmentation exponents against each other: how fast the number of
    sectors grows, against how fast the attractor count inside ONE sector grows.

    The upper-right quadrant is nested fragmentation -- exponential in both.  It
    is empty except for one undecided candidate, which is the report's result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = data or load(bc) or build(bc=bc)
    rows = [classify(r) for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.grid(True, color="#ececea", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvspan(1.02, 1.35, ymin=0, ymax=1, color="#f6f2e6", zorder=0)
    ax.axhspan(1.02, 1.62, xmin=0, xmax=1, color="#f6f2e6", zorder=0)
    ax.axvline(1.02, color="#bbb", lw=0.8, ls="--", zorder=1)
    ax.axhline(1.02, color="#bbb", lw=0.8, ls="--", zorder=1)
    for r in rows:
        x = r["sectors"]["base"] or 1.0
        y = r["in_sector"]["base"] or 1.0
        exp_x = r["sectors"]["cls"] == "exponential"
        exp_y = r["in_sector"]["cls"] == "exponential"
        c = "#c62828" if (exp_x and exp_y) else ("#1f4e9c" if exp_x else
                                                 ("#e08a1e" if exp_y else "#8a8f98"))
        ax.scatter(x, y, s=64, facecolor=c, edgecolor="white", lw=1.0, zorder=5)
        ax.annotate(f"{r['rule']}", (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, color=c, zorder=6)
    ax.set_xlabel("growth base of the number of sectors", fontsize=9)
    ax.set_ylabel("growth base of the attractors inside ONE sector", fontsize=9)
    ax.set_xlim(0.97, 1.35)
    ax.set_ylim(0.97, 1.62)
    ax.annotate("nested fragmentation\n(exponential both ways)", (1.28, 1.55),
                fontsize=8, ha="right", color="#8a5a12")
    ax.annotate("all fragmentation\ninside the sectors", (0.99, 1.55),
                fontsize=8, color="#8a5a12")
    ax.annotate("all fragmentation\nin the sector count", (1.28, 1.00),
                fontsize=8, ha="right", color="#3a4a6a")
    ax.set_title("Where the thirteen multi-attractor rules sit", fontsize=10)
    return _save(fig, out, bc, "nested_plane")


def fig_staircase(bc="obc0", out=None, N=16, rules_in=(100, 44, 36, 203, 29)):
    """
    Attractors per sector, sorted, for one N.  Rules 44 and 100 give consecutive
    Narayana numbers; 36 gives one giant sector and singletons; 29 gives a broad
    shallow spread.  Same total for the first three.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.grid(True, color="#ececea", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cols = ["#c62828", "#e08a1e", "#2e7d32", "#1f4e9c", "#7b3fa0"]
    for i, r in enumerate(rules_in):
        row = census(r, [N], bc)
        y = row["dist"]
        ax.step(range(1, len(y) + 1), y, where="mid", lw=1.6,
                color=cols[i % len(cols)],
                label=f"W{r}  ({sum(y)} attractors in {len(y)} sectors)")
        ax.scatter(range(1, len(y) + 1), y, s=14, color=cols[i % len(cols)],
                   zorder=5)
    nar = [v for v in narayana(24) if v >= 3][:12]
    ax.plot(range(1, len(nar) + 1), sorted(nar, reverse=True), ls=":", lw=1.2,
            color="#444", label="Narayana numbers (reference)")
    ax.set_yscale("log")
    ax.set_xlabel("sector, ordered by how many attractors it holds", fontsize=9)
    ax.set_ylabel("attractors in that sector", fontsize=9)
    ax.set_title(f"How the attractors distribute across sectors ($N={N}$)",
                 fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.95)
    return _save(fig, out, bc, f"nested_staircase_N{N}")


def fig_rule29(bc="obc0", out=None):
    """
    The undecided case.  Rule 29's in-sector attractor count on the even-N
    branch, with the two models that fit it equally well and the N at which they
    would separate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = np.array([6, 8, 10, 12, 14, 16, 18, 20])
    y = np.array([3, 4, 5, 6, 7, 9, 12, 16], dtype=float)
    c2 = np.polyfit(Ns, y, 2)
    c1 = np.polyfit(Ns, np.log(y), 1)
    xs = np.linspace(6, 30, 200)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.grid(True, color="#ececea", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot(xs, np.polyval(c2, xs), lw=1.4, color="#1f4e9c",
            label="quadratic in $N$")
    ax.plot(xs, np.exp(np.polyval(c1, xs)), lw=1.4, color="#c62828",
            label=f"exponential, base ${np.exp(c1[0]):.4f}^N$")
    ax.scatter(Ns, y, s=42, color="#111", zorder=5, label="computed")
    ax.axvspan(20, 30, color="#f6f2e6", zorder=0)
    ax.annotate("computed", (13, 17), fontsize=8, color="#444")
    ax.annotate("the two models only separate here", (21, 6), fontsize=8,
                color="#8a5a12")
    ax.set_xlabel("$N$", fontsize=9)
    ax.set_ylabel("attractors in the largest sector (even $N$)", fontsize=9)
    ax.set_title("W29: superlinear, but quadratic and exponential are tied",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out, bc, "nested_rule29")


def _save(fig, out, bc, tag):
    import matplotlib.pyplot as plt
    out = out or os.path.join(_fig_dir(), f"fig_{tag}_{bc}.pdf")
    for p in (out, out.replace(".pdf", ".png")):
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R13 nested fragmentation")
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    d = build(bc=a.bc) if a.rebuild else (load(a.bc) or build(bc=a.bc))
    fig_plane(a.bc, data=d)
    fig_staircase(a.bc)
    fig_rule29(a.bc)


if __name__ == "__main__":
    main()


#: Rule 29's attractors-in-the-largest-sector on the even-N branch, N = 6..20.
#: Computed with a lean union-find + Tarjan pass; N = 20 is 2^20 states.
RULE29_EVEN_N = (6, 8, 10, 12, 14, 16, 18, 20)
RULE29_EVEN_Y = (3, 4, 5, 6, 7, 9, 12, 16)


def rule29_model_comparison() -> Dict:
    """
    The open case, quantified.

    Rule 29 is the only rule whose sector count is exponential AND whose
    in-sector attractor count is superlinear, so it is the only candidate for
    nested fragmentation.  On the even branch the two competing models are:

        quadratic in N     second differences settle at 1
        exponential        base ~1.12 per site

    They fit equally well (BIC differs by 1.6, which is nothing) and they only
    separate by more than one attractor beyond N = 26 -- about 6.7e7 states.
    So the honest verdict is OPEN, not "nested".
    """
    N = np.asarray(RULE29_EVEN_N, dtype=float)
    y = np.asarray(RULE29_EVEN_Y, dtype=float)
    c2 = np.polyfit(N, y, 2)
    c1 = np.polyfit(N, np.log(y), 1)
    r2 = y - np.polyval(c2, N)
    r1 = y - np.exp(np.polyval(c1, N))

    def bic(res, k):
        n = len(res)
        return n * np.log(max((res ** 2).sum() / n, 1e-12)) + k * np.log(n)

    return {"N": list(RULE29_EVEN_N), "y": list(RULE29_EVEN_Y),
            "second_differences": [int(v) for v in np.diff(y, 2)],
            "quadratic": {"rss": float((r2 ** 2).sum()), "bic": float(bic(r2, 3)),
                          "at_22": float(np.polyval(c2, 22)),
                          "at_26": float(np.polyval(c2, 26))},
            "exponential": {"rss": float((r1 ** 2).sum()), "bic": float(bic(r1, 2)),
                            "base": float(np.exp(c1[0])),
                            "at_22": float(np.exp(np.polyval(c1, 22))),
                            "at_26": float(np.exp(np.polyval(c1, 26)))},
            "verdict": "open"}
