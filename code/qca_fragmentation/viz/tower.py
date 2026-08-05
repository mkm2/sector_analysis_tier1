"""
The tower in R9 F1's right panel: V+reset rules with a BOUNDED number of
attractors whose largest one still grows exponentially.

In the monitored-attractor plane those rules pile up in a column at
a_att = 1 with b_att running from 1 up to nearly 2 -- a tower parallel to the
b-axis.  This module locates them, tabulates the growth bases (which land on a
handful of named algebraic constants), and tests the natural explanation:

    HYPOTHESIS.  The dissipative channels never fire on the largest attractor,
    so the dynamics there is the underlying unitary and the attractor is really
    a sector of the coherent parent.

It is true for 12 of the 68, and false for the rest -- see `idle_census`.

WHAT "FIRES" MEANS.  A reset D writes 0 and E writes 1.  Either does WORK only
when the target bit is not already at that value; otherwise it acts as the
identity on that state.  `firing_sites` runs the brick-wall sweep by hand,
following every Hadamard branch, and returns the set of sites at which some
branch had a reset do work.  It is validated against the engine's successor map
in the tests -- the hand sweep must reproduce `wcc.make_succ` exactly, or the
firing flags mean nothing.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from ..core.cycle import even_sites, odd_sites, trigger_symbol
from ..graph import scc as scc_mod
from ..graph import wcc
from ..scaling import sectors

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")

#: a rule is "in the tower" if it is V+reset, its attractor COUNT is flat, and
#: its largest attractor grows exponentially
A_FLAT = 0.02
B_GROW = 1.02


def tower_rules(bc: str = "obc0", data: Optional[Dict] = None):
    """[(rule, a_att, b_att)] for the tower, sorted by rule."""
    d = data or sectors.load(bc) or sectors.build(bc)
    pts = {p["rule"]: p for p in d["points"]}
    out = []
    for x in d["attractor_deficits"]:
        p = pts.get(x["rule"])
        if not p or p["family"] != "mixed":
            continue
        if abs(x["a_att"] - 1.0) < A_FLAT and x["b_att"] > B_GROW:
            out.append((x["rule"], x["a_att"], x["b_att"]))
    return sorted(out)


# --- does a reset do work? ----------------------------------------------------

def sweep_labeled(x: int, N: int, t, bc: str = "obc0"):
    """
    [(successor, frozenset of sites where a reset did work)] over all branches.

    The project's brick wall: the even sublattice ascending, then the odd one,
    each site reading the CURRENT state.  A Hadamard doubles the branch list; a
    reset writes its value and records the site iff the bit had to change.
    """
    branches: List[Tuple[int, FrozenSet[int]]] = [(x, frozenset())]
    for group in (even_sites(N), odd_sites(N)):
        for i in group:
            nxt: List[Tuple[int, FrozenSet[int]]] = []
            for y, fired in branches:
                s = trigger_symbol(y, i, N, t, bc)
                bit = (y >> i) & 1
                if s == "I":
                    nxt.append((y, fired))
                elif s == "V":
                    nxt.append((y & ~(1 << i), fired))
                    nxt.append((y | (1 << i), fired))
                elif s == "D":
                    nxt.append((y & ~(1 << i),
                                fired | {i} if bit == 1 else fired))
                elif s == "E":
                    nxt.append((y | (1 << i),
                                fired | {i} if bit == 0 else fired))
                else:
                    raise ValueError(f"unknown symbol {s!r}")
            branches = nxt
    return branches


def firing_sites(x: int, N: int, t, bc: str = "obc0") -> Set[int]:
    out: Set[int] = set()
    for _, f in sweep_labeled(x, N, t, bc):
        out |= f
    return out


def largest_attractor(rule: int, N: int, bc: str = "obc0") -> List[int]:
    t = rules_mod.wolfram_to_tuple(rule)
    rec = scc_mod.recurrent_classes(rule, N, bc, t)
    return max(rec, key=len) if rec else []


def idle_census(rule: int, N: int, bc: str = "obc0") -> Dict:
    """
    Everything the hypothesis needs, for one rule at one N.

    `fires` counts states of the largest attractor on which some reset does
    work; `sites` is where.  `agrees_with_parent` asks the same question the
    other way round -- does the child's successor map coincide with the coherent
    parent's on that attractor?  The two must agree, and the tests check they do.
    """
    t = rules_mod.wolfram_to_tuple(rule)
    A = largest_attractor(rule, N, bc)
    sites: Set[int] = set()
    fires = 0
    for x in A:
        f = firing_sites(x, N, t, bc)
        if f:
            fires += 1
            sites |= f
    parent = rules_mod.coherent_parent(rule)
    sc = wcc.make_succ(rule, N, bc)
    sp = wcc.make_succ(parent, N, bc)
    agree = all(sorted(sc(x)) == sorted(sp(x)) for x in A)
    pres = wcc.weak_components(parent, N, bc)
    return {"rule": rule, "tuple": "".join(t), "N": N, "bc": bc,
            "size": len(A), "fires": fires, "idle": fires == 0,
            "n_firing_sites": len(sites), "firing_sites": sorted(sites),
            "parent": parent, "parent_d_max": pres.d_max_wcc,
            "agrees_with_parent": bool(agree),
            "is_parent_sector": len(A) == pres.d_max_wcc and bool(agree)}


def census(bc: str = "obc0", N: int = 10, data: Optional[Dict] = None):
    return [idle_census(r, N, bc) for r, _, _ in tower_rules(bc, data)]


def base_distribution(bc: str = "obc0", data: Optional[Dict] = None):
    """{rounded b_att: {'rules': [...], 'n_exact': k, 'name': str}}"""
    from ..scaling.fits import name_base
    out: Dict[float, Dict] = {}
    for rule, _a, b in tower_rules(bc, data):
        key = round(b, 6)
        e = out.setdefault(key, {"rules": [], "n_exact": 0,
                                 "name": name_base(key)})
        e["rules"].append(rule)
        p = sectors.attractor_point(rule, bc)
        if p and p["d_max"]["exact"]:
            e["n_exact"] += 1
    return dict(sorted(out.items()))


# --- figure -------------------------------------------------------------------

def fig_tower(bc: str = "obc0", out: Optional[str] = None, N: int = 10,
              rows: Optional[List[Dict]] = None, data: Optional[Dict] = None):
    """
    Left: the tower itself, b_att against rule count, with the named constants
    marked and the idle rules picked out.  Right: how many sites see a firing
    reset, against N -- flat at zero for the idle rules, linear for the rest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tw = tower_rules(bc, data)
    rows = rows or census(bc, N, data)
    idle = {r["rule"] for r in rows if r["idle"]}
    dist = base_distribution(bc, data)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))

    ax = axes[0]
    ax.grid(True, color="#ececea", lw=0.6, axis="x")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ys = sorted(dist)
    for i, v in enumerate(ys):
        rs = dist[v]["rules"]
        n_idle = sum(1 for r in rs if r in idle)
        ax.barh(i, len(rs) - n_idle, color="#c62828", height=0.62,
                edgecolor="white", lw=0.6)
        if n_idle:
            ax.barh(i, n_idle, left=len(rs) - n_idle, color="#e8a33d",
                    height=0.62, edgecolor="white", lw=0.6)
        nm = dist[v]["name"] or ""
        ax.annotate(f"  {len(rs)}" + (f"   ({n_idle} idle)" if n_idle else ""),
                    (len(rs), i), va="center", fontsize=7.5, color="#333")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{v:.5f}   {dist[v]['name'] or ''}" for v in ys],
                       fontsize=7.8)
    ax.set_xlabel("rules", fontsize=9)
    ax.set_title(f"the tower: {len(tw)} V+reset rules with a flat attractor\n"
                 "count and an exponentially growing largest attractor",
                 fontsize=9.5)
    ax.set_xlim(0, max(len(v["rules"]) for v in dist.values()) * 1.28)

    ax = axes[1]
    ax.grid(True, color="#ececea", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    Ns = [8, 10, 12]
    for r in rows:
        ys2 = [idle_census(r["rule"], n, bc)["n_firing_sites"] for n in Ns]
        c = "#e8a33d" if r["rule"] in idle else "#c62828"
        ax.plot(Ns, ys2, "-o", ms=3, lw=0.9, color=c,
                alpha=0.95 if r["rule"] in idle else 0.28)
    ax.plot(Ns, Ns, ls=":", color="#444", lw=1.0)
    ax.annotate("all $N$ sites", (12, 12), fontsize=7.5, color="#444",
                ha="right", va="bottom")
    ax.set_xlabel("$N$", fontsize=9)
    ax.set_ylabel("sites at which a reset does work\non the largest attractor",
                  fontsize=9)
    ax.set_xticks(Ns)
    ax.set_title("the hypothesis, tested: idle (gold) vs firing (red)",
                 fontsize=9.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#e8a33d", marker="o", ms=4,
               label=f"no reset ever fires ({len(idle)})"),
        Line2D([], [], color="#c62828", marker="o", ms=4, alpha=0.5,
               label=f"resets fire ({len(rows) - len(idle)})")],
        fontsize=7.5, loc="upper left")

    fig.tight_layout()
    out = out or os.path.join(FIGURES_DIR, f"fig_tower_{bc}.pdf")
    for p in (out, out.replace(".pdf", ".png")):
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows


def write_tables(bc: str = "obc0", N: int = 10, rows: Optional[List] = None):
    rows = rows or census(bc, N)
    dist = base_distribution(bc)
    tex = os.path.join(results_io.REPO_ROOT, "reports", "tex")
    idle = [r for r in rows if r["idle"]]

    p1 = os.path.join(tex, f"tab_r15_bases_{bc}.tex")
    with open(p1, "w") as f:
        f.write("\\begin{tabular}{rrrl}\n\\hline\n")
        f.write("$b_\\att$ & rules & exact & constant \\\\\n\\hline\n")
        for v, e in dist.items():
            f.write(f"${v:.6f}$ & ${len(e['rules'])}$ & ${e['n_exact']}$ & "
                    f"{e['name'] or '---'} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    p2 = os.path.join(tex, f"tab_r15_idle_{bc}.tex")
    with open(p2, "w") as f:
        f.write("\\begin{tabular}{rlrrrl}\n\\hline\n")
        f.write("rule & tuple & $b_\\att$ & $|A|$ & parent & parent "
                "$D_{\\max}$ \\\\\n\\hline\n")
        bt = {r: b for r, _, b in tower_rules(bc)}
        for r in sorted(idle, key=lambda r: r["rule"]):
            f.write(f"${r['rule']}$ & \\rt{{{r['tuple']}}} & "
                    f"${bt[r['rule']]:.5f}$ & ${r['size']}$ & "
                    f"$W{r['parent']}$ & ${r['parent_d_max']}$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return [p1, p2]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R15: the attractor tower")
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--N", type=int, default=10)
    a = ap.parse_args(argv)
    rows = census(a.bc, a.N)
    n_idle = sum(1 for r in rows if r["idle"])
    print(f"tower rules: {len(rows)};  no reset ever fires: {n_idle}")
    fig_tower(a.bc, N=a.N, rows=rows)
    for p in write_tables(a.bc, a.N, rows):
        print("wrote", p)


if __name__ == "__main__":
    main()


# --- the mechanism for the rules whose resets DO fire -------------------------
#
# The largest attractor turns out to be a SUBSHIFT OF FINITE TYPE over the
# two-site brick-wall unit cell: whether a state belongs to it is decided by
# which pairs of ADJACENT 2-site blocks occur, nothing longer.  The growth base
# is then sqrt(Perron root of the 4x4 block-transition matrix) -- the square
# root because one block is two sites.
#
# That is why the resets firing does not spoil the algebraic constant.  A reset
# is a LOCAL map, so the set it leaves invariant is cut out by local
# constraints, and a locally-constrained set of words is counted by a transfer
# matrix whatever the map does.  Unitarity is sufficient for an algebraic base,
# not necessary; locality is what is doing the work.

def blocks_of(x: int, N: int) -> Tuple[int, ...]:
    """The state as N/2 two-site blocks, low bits first."""
    return tuple((x >> (2 * i)) & 3 for i in range(N // 2))


def learn_grammar(A: Sequence[int], N: int):
    """(first blocks, allowed adjacent pairs, last blocks) observed in A."""
    S: Set[int] = set()
    T: Set[Tuple[int, int]] = set()
    E: Set[int] = set()
    for x in A:
        b = blocks_of(x, N)
        S.add(b[0])
        E.add(b[-1])
        T |= set(zip(b, b[1:]))
    return S, T, E


def sft_closure(S, T, E, N: int) -> Set[int]:
    """Every state the grammar allows.  A is always a SUBSET of this."""
    out: Set[int] = set()
    m = N // 2

    def walk(seq):
        if len(seq) == m:
            if seq[-1] in E:
                out.add(sum(b << (2 * i) for i, b in enumerate(seq)))
            return
        for nb in range(4):
            if (seq[-1], nb) in T:
                walk(seq + (nb,))

    for b0 in S:
        walk((b0,))
    return out


def block_perron(T) -> float:
    M = np.zeros((4, 4))
    for a, b in T:
        M[a, b] = 1.0
    return float(max(e.real for e in np.linalg.eigvals(M)))


def sft_check(rule: int, N: int, bc: str = "obc0") -> Dict:
    """
    Is the largest attractor an SFT over 2-site blocks, and what does the
    transfer matrix predict for the growth base?

    `exact` is the whole claim: the grammar learned from A regenerates A and
    nothing else.  When it fails the closure is strictly larger, so
    sqrt(perron) is only an UPPER BOUND on the true base.
    """
    A = set(largest_attractor(rule, N, bc))
    S, T, E = learn_grammar(A, N)
    R = sft_closure(S, T, E, N)
    pr = block_perron(T)
    return {"rule": rule, "N": N, "size": len(A), "closure": len(R),
            "exact": R == A, "perron": pr, "base": pr ** 0.5,
            "n_transitions": len(T), "transitions": sorted(T)}
