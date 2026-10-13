"""
Export the spy-plot data of one unitary rule for the thesis figure project.

    python -m qca_fragmentation.viz.export_spy --rule 150 --N 11 \\
        --out "~/Dokumente/Projects/Figure Generation/data/rule150_hadamard_spy_l11"

Writes  <out>/data.npz  with the support of the one-cycle matrix (the same
pattern spy.transition_pattern draws) and the sector (weak-component) ordering
used for the sorted panel.  The meta.yaml next to it is written by hand: the
figure project reads only that and the arrays, never this code.

Everything is exact (support of the exact Z[1/sqrt2] engine, no tolerances).
Before writing, the export checks that every nonzero lies inside one sector and,
for rule 150 at obc0, that the sectors are the domain-wall shells with
|S_w| = C(N+1, w) for even w (R8).
"""

from __future__ import annotations

import argparse
import os
from math import comb

import numpy as np

from . import spy


def domain_walls(x: int, N: int) -> int:
    """Domain walls of x on the obc0 chain 0 x_0 .. x_{N-1} 0 (N+1 bonds)."""
    padded = x << 1                     # bit 0 = left vacuum, bit N+1 = right vacuum
    return bin(padded ^ (padded >> 1)).count("1")


def export(rule: int, N: int, out: str, bc: str = "obc0") -> dict:
    n = 1 << N
    rows, cols = spy.transition_pattern(rule, N, bc)
    blocks = spy.wcc_blocks(rule, N, bc)            # largest first, states ascending
    order, starts = spy.sorted_permutation(blocks)
    position = np.empty(n, dtype=np.int32)
    position[order] = np.arange(n)
    sector = spy.block_of(blocks, n)

    off = int(np.count_nonzero(sector[rows] != sector[cols]))
    assert off == 0, f"{off} nonzeros outside their sector"
    w = []
    for b in blocks:
        ws = {domain_walls(x, N) for x in b}
        assert len(ws) == 1, f"sector mixes domain-wall numbers {ws}"
        w.append(ws.pop())
    if rule == 150 and bc == "obc0":
        for b, wb in zip(blocks, w):
            assert wb % 2 == 0 and len(b) == comb(N + 1, wb), (wb, len(b))

    os.makedirs(out, exist_ok=True)
    arrays = dict(
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        order=np.asarray(order, dtype=np.int32),
        position=position,
        sector=sector.astype(np.int32),
        sector_size=np.asarray([len(b) for b in blocks], dtype=np.int32),
        sector_start=np.asarray(starts, dtype=np.int32),
        sector_w=np.asarray(w, dtype=np.int32),
    )
    np.savez_compressed(os.path.join(out, "data.npz"), **arrays)
    summary = {"rule": rule, "N": N, "bc": bc, "dim": n, "nnz": int(len(rows)),
               "sector_size": arrays["sector_size"].tolist(),
               "sector_w": w}
    print(summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="export spy-plot data")
    ap.add_argument("--rule", type=int, default=150)
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    export(a.rule, a.N, os.path.expanduser(a.out), a.bc)


if __name__ == "__main__":
    main()
