"""
Export the sector (transition) graph of one rule for the thesis figure project.

    python -m qca_fragmentation.viz.export_graph --rule 150 --N 11 \\
        --out "~/Dokumente/Projects/Figure Generation/data/rule150_hadamard_graph_l11"

Writes  <out>/data.npz : the graph of dot_sectors.sector_graph (one node per
basis state, one undirected edge per transition x -> y with <y|U|x> != 0,
self-loops dropped) and a precomputed layout, so the renderer computes nothing.

Layout: each sector separately by stress majorization (dot_sectors.stress_layout,
deterministic: classical-MDS start, fixed iteration count), then centred and
scaled to unit radius.  Placing the sectors relative to each other is left to
the renderer.  Sector indices follow spy.wcc_blocks / export_spy (0 = largest,
ties by smallest state index), so colours match the spy plot by construction.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import dot_sectors, spy
from .export_spy import domain_walls


def export(rule: int, N: int, out: str, bc: str = "obc0") -> dict:
    n = 1 << N
    edges, sectors, adj = dot_sectors.sector_graph(rule, N, bc)
    sectors.sort(key=lambda s: (-len(s), s[0]))
    assert sectors == spy.wcc_blocks(rule, N, bc), "sector order differs from the spy plot"

    sector = np.empty(n, dtype=np.int32)
    for k, s in enumerate(sectors):
        sector[s] = k
    u = np.asarray([e[0] for e in edges], dtype=np.int32)
    v = np.asarray([e[1] for e in edges], dtype=np.int32)
    assert np.all(u < v) and np.all(sector[u] == sector[v]), "edge leaves its sector"

    x = np.zeros(n)
    y = np.zeros(n)
    for s in sectors:
        P = np.asarray(dot_sectors.stress_layout(s, adj, seed=rule))
        if len(s) > 1:
            P = P - P.mean(axis=0)
            P = P / np.max(np.hypot(P[:, 0], P[:, 1]))
        x[s], y[s] = P[:, 0], P[:, 1]

    w = []
    for s in sectors:
        ws = {domain_walls(z, N) for z in s}
        assert len(ws) == 1
        w.append(ws.pop())

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "data.npz"),
        node_x=x, node_y=y, node_sector=sector,
        edge_u=u, edge_v=v,
        sector_size=np.asarray([len(s) for s in sectors], dtype=np.int32),
        sector_w=np.asarray(w, dtype=np.int32),
    )
    summary = {"rule": rule, "N": N, "bc": bc, "n_nodes": n, "n_edges": len(edges),
               "sector_size": [len(s) for s in sectors], "sector_w": w}
    print(summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="export sector-graph data")
    ap.add_argument("--rule", type=int, default=150)
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    export(a.rule, a.N, os.path.expanduser(a.out), a.bc)


if __name__ == "__main__":
    main()
