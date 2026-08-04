"""
Spy plots of the one-step transition matrix, in the computational basis and in
a WCC-sorted basis.

WHAT IS PLOTTED.  M[y, x] = 1 whenever the one-step map sends basis state x to
basis state y with nonzero amplitude, i.e. the sparsity pattern of the circuit
unitary U (equivalently of the doubly stochastic |U|^2, which has the same
support).  No amplitudes: a spy plot is about support, and the whole point here
is that the support alone already carries the fragmentation.

WHY IT IS WORTH A FIGURE.  Every sector statement in R2 and R9 is a number --- a
count, a growth base, an exponent.  The block structure those numbers describe is
a property of a matrix, and permuting the basis so the blocks are contiguous
turns the claim into something you can look at.  The two panels are the SAME
matrix; only the ordering of the basis differs.

Rule 156 (\\rt{IIVI}) is the natural subject: it is unitary, so |U|^2 is doubly
stochastic and weak, strong and forward-closure partitions all coincide (R9's
assertion A2) --- the blocks are unambiguous.  Its sector count at obc0 is
Fibonacci (55, 89, 144, 233, 377 for N = 8..12), which is where R2's golden ratio
comes from.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from ..graph import wcc

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")


def transition_pattern(rule: int, N: int, bc: str = "obc0"):
    """(rows, cols) of the nonzeros of M, with M[y, x] = 1 iff x -> y."""
    succ = wcc.make_succ(rule, N, bc)
    rows: List[int] = []
    cols: List[int] = []
    for x in range(1 << N):
        for y in succ(x):
            rows.append(y)
            cols.append(x)
    return np.asarray(rows), np.asarray(cols)


def wcc_blocks(rule: int, N: int, bc: str = "obc0"):
    """
    Sectors as node lists, largest first -- the same partition R9 counts.

    Weak connectivity, so the undirected closure of the transition graph; for a
    unitary rule this coincides with the strong one (A2).
    """
    succ = wcc.make_succ(rule, N, bc)
    n = 1 << N
    adj: Dict[int, set] = {v: set() for v in range(n)}
    for x in range(n):
        for y in succ(x):
            adj[x].add(y)
            adj[y].add(x)
    seen = [False] * n
    out: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
        out.append(sorted(comp))
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def sorted_permutation(blocks: List[List[int]]):
    """
    perm[new_index] = old_state, plus the block boundaries.

    Blocks are laid out largest first, states ascending inside each block, so the
    picture is a staircase of shrinking diagonal blocks and the tail of frozen
    singletons is the thin diagonal at the bottom right.
    """
    perm = [v for b in blocks for v in b]
    starts, k = [], 0
    for b in blocks:
        starts.append(k)
        k += len(b)
    return np.asarray(perm), starts


def block_of(blocks: List[List[int]], n: int) -> np.ndarray:
    owner = np.empty(n, dtype=int)
    for i, b in enumerate(blocks):
        for v in b:
            owner[v] = i
    return owner


def check_block_diagonal(rule: int, N: int, bc: str = "obc0") -> Dict:
    """
    The claim the sorted panel makes: EVERY nonzero lies inside a diagonal block.

    This is what "the sectors block-diagonalise the dynamics" means concretely,
    and it is cheap to verify rather than assert -- so the figure ships with its
    own check.
    """
    rows, cols = transition_pattern(rule, N, bc)
    blocks = wcc_blocks(rule, N, bc)
    owner = block_of(blocks, 1 << N)
    off = int(np.count_nonzero(owner[rows] != owner[cols]))
    sizes = [len(b) for b in blocks]
    return {"rule": rule, "N": N, "bc": bc, "dim": 1 << N,
            "nnz": int(len(rows)), "n_blocks": len(blocks),
            "d_max": max(sizes), "n_frozen": sum(1 for s in sizes if s == 1),
            "off_block_nonzeros": off,
            "density": len(rows) / float((1 << N) ** 2),
            "block_fill": len(rows) / float(sum(s * s for s in sizes))}


# --- the figure ---------------------------------------------------------------

_DOT = "#16223a"
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
            "#8c564b", "#e377c2"]


def _style(ax, n):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)          # matrix convention: row 0 at the top
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_spy(rule: int, N: int, bc: str = "obc0", out: Optional[str] = None,
            dot: float = 0.45):
    """
    Two panels of the same matrix: computational-basis order, and WCC order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = 1 << N
    rows, cols = transition_pattern(rule, N, bc)
    blocks = wcc_blocks(rule, N, bc)
    perm, starts = sorted_permutation(blocks)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)
    owner = block_of(blocks, n)
    stat = check_block_diagonal(rule, N, bc)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))

    ax = axes[0]
    _style(ax, n)
    ax.scatter(cols, rows, s=dot, c=_DOT, marker="s", linewidths=0)
    # NOT "structureless": the computational basis shows strong banding and a
    # self-similar block-of-blocks pattern, because the integer order groups
    # states by their high bits and the rule is local.  What it does not show is
    # the SECTOR structure -- the bands cross between sectors freely.
    ax.set_title("computational basis: banded and self-similar,\n"
                 "but not block-diagonal", fontsize=10)
    ax.set_xlabel("column = state $x$"), ax.set_ylabel("row = successor $y$")

    ax = axes[1]
    _style(ax, n)
    # shade every block first, so the dots sit on top of their own square
    for i, b in enumerate(blocks):
        s, L = starts[i], len(b)
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), L, L,
                               facecolor=_PALETTE[i % len(_PALETTE)],
                               alpha=0.30, edgecolor="none", zorder=1))
    ax.scatter(pos[cols], pos[rows],
               s=dot, c=[_PALETTE[i % len(_PALETTE)] for i in owner[cols]],
               marker="s", linewidths=0, zorder=3)
    ax.set_title(f"sorted by weak component: {stat['n_blocks']} blocks,\n"
                 f"largest {stat['d_max']}, every nonzero inside one",
                 fontsize=10)
    ax.set_xlabel("column = state, sectors contiguous and largest first")
    ax.set_ylabel("row = successor")

    tup = "".join(rules_mod.wolfram_to_tuple(rule))
    fig.suptitle(
        f"W{rule} ({tup}), "
        f"{bc}, $N={N}$: transition-matrix support, "
        f"{stat['nnz']} nonzeros in a ${n}\\times{n}$ matrix "
        f"({100 * stat['density']:.2f}% dense).  "
        f"Off-block nonzeros: {stat['off_block_nonzeros']}.",
        fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out or os.path.join(FIGURES_DIR, f"fig_spy_{rule}_N{N}_{bc}.pdf")
    for p in (out, out.replace(".pdf", ".png")):
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return stat


def fig_spy_zoom(rule: int, N: int, bc: str = "obc0", span: int = 140,
                 out: Optional[str] = None, dot: float = 6.0):
    """
    The top-left corner of the sorted matrix, where the blocks are big enough to
    read individually.  At N = 10 the largest sector is 20 states, which is four
    pixels wide in the full picture and unreadable there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = 1 << N
    rows, cols = transition_pattern(rule, N, bc)
    blocks = wcc_blocks(rule, N, bc)
    perm, starts = sorted_permutation(blocks)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)
    owner = block_of(blocks, n)

    r2, c2 = pos[rows], pos[cols]
    keep = (r2 < span) & (c2 < span)

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    _style(ax, span)
    ax.set_ylim(span - 0.5, -5.0)      # headroom: the first block's size label
                                       # sits above the matrix and was clipped
    for i, b in enumerate(blocks):
        s, L = starts[i], len(b)
        if s >= span:
            break
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), L, L,
                               facecolor=_PALETTE[i % len(_PALETTE)],
                               alpha=0.25, edgecolor=_PALETTE[i % len(_PALETTE)],
                               linewidth=0.8, zorder=1))
        if L >= 8:
            ax.annotate(f"{L}", (s + L / 2 - 0.5, s - 1.5), ha="center",
                        va="bottom", fontsize=6.5,
                        color=_PALETTE[i % len(_PALETTE)], zorder=5)
    ax.scatter(c2[keep], r2[keep], s=dot,
               c=[_PALETTE[i % len(_PALETTE)] for i in owner[cols[keep]]],
               marker="s", linewidths=0, zorder=3)
    ax.set_title(f"W{rule} {bc} $N={N}$: first {span} rows/columns of the "
                 "sorted matrix\n(block sizes annotated)", fontsize=10)
    ax.set_xlabel("column"), ax.set_ylabel("row")
    fig.tight_layout()
    out = out or os.path.join(FIGURES_DIR,
                              f"fig_spy_zoom_{rule}_N{N}_{bc}.pdf")
    for p in (out, out.replace(".pdf", ".png")):
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def block_size_table(rule: int, Ns=(8, 9, 10, 11, 12), bc: str = "obc0",
                     out: Optional[str] = None) -> str:
    """Dimensions, block counts and fill, for the report."""
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex",
                              f"tab_r12_spy_{bc}.tex")
    rows = [check_block_diagonal(rule, N, bc) for N in Ns]
    with open(out, "w") as f:
        f.write("\\begin{tabular}{rrrrrrr}\n\\hline\n")
        f.write("$N$ & $\\dim$ & nnz & density & blocks & $D_{\\max}$ & "
                "off-block \\\\\n\\hline\n")
        for r in rows:
            f.write(f"${r['N']}$ & ${r['dim']}$ & ${r['nnz']}$ & "
                    f"${100 * r['density']:.2f}\\%$ & ${r['n_blocks']}$ & "
                    f"${r['d_max']}$ & ${r['off_block_nonzeros']}$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="R12 spy plots")
    ap.add_argument("--rule", type=int, default=156)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args(argv)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_spy(a.rule, a.N, a.bc)
    fig_spy(a.rule, 8, a.bc)
    fig_spy_zoom(a.rule, a.N, a.bc)
    print("table:", block_size_table(a.rule, bc=a.bc))


if __name__ == "__main__":
    main()
