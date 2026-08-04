"""
Graphviz sector pictures, ported from HSF/visualization_V3.jl and _V4.jl.

WHY A PORT AND NOT THE JULIA.  The Julia scripts drive HSF's
`build_unitary_sparse`, which is unitary-only: `binary_coefficients` rejects
anything outside 0-15, and the model has no reset in it at all.  R9's survivors
are all V+reset rules (73, 109, 28, 70, 157, 199, 71, 29 exponential; 44, 100,
104, 110, 124, 20, 148, 158, 188, 6, 134, 214, 230, 233 polynomial), so the
Julia path cannot build a single one of them.  The conventions do match, which
is what makes a port faithful:

  * HSF's boundary gates read the missing neighbour as |0>  -> our `obc0`;
  * HSF applies sites 1,3,5,... (1-based) and then 2,4,6,...  -> our
    even-sublattice-then-odd brick wall;
  * HSF's Krylov sectors are connected components of |U| -> our WCC sectors,
    which is the same object once the dissipative graph is symmetrised.

WHAT IS DRAWN.  One node per basis state (node id = the state integer), one
undirected edge per transition, self-loops dropped exactly as the Julia does.
Positions are written absolutely as pos="x,y!", so the file is meant for

    neato -n2 -Tpng graph.dot -o graph.png

which honours the coordinates instead of recomputing a layout.

TWO LAYOUTS, both kept because they answer different questions:

  V3  "core-periphery": the largest sector is blown up at the origin, the next
      num_central-1 sit on a ring of radius 130, the rest are scattered.  Good
      when one sector dominates -- the polynomial survivors put half the basis
      in a single component.
  V4  "archipelago": every sector including the largest goes through the same
      rejection sampling, with the three biggest drawn larger.  Good when many
      sectors are comparable, which is the exponential survivors.

The Julia calls `rand()` unseeded; here the placement is seeded per rule so a
rerun reproduces the same picture.
"""

from __future__ import annotations

import colorsys
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import results_io
from ..graph import wcc

PLOTS_DIR = os.path.join(results_io.REPO_ROOT, "figures", "sector_graphs")

#: R9 sec.7: the eight V+reset rules whose sector count stays exponential.
EXPONENTIAL_SURVIVORS = (73, 109, 28, 70, 157, 199, 71, 29)
#: R9 tab_r9_polysurv: the fourteen that drop to a polynomial sector count.
POLYNOMIAL_SURVIVORS = (44, 100, 104, 110, 124, 20, 148, 158, 188, 6, 134,
                        214, 230, 233)
#: R9 sec.7.3: the rules whose sectors carry MORE THAN ONE terminal SCC, i.e.
#: the rules for which counting sectors is not counting attractors.  Six of them
#: (29, 71, 44, 100, 104, 233) are also survivors, so the two groups overlap and
#: the union is 28 rules, not 34.
#:
#: THE TWELVE ARE MEASURED AT N_max = 16.  That is not the same as the rules that
#: ever have the property: rule 37 (EDDV) has one sector at every N and two
#: attractors exactly when N = 2 (mod 3), so it is an exception at N = 8, 11, 14
#: and invisible at N_max = 16 = 1 (mod 3).  It is listed separately rather than
#: folded in, because the twelve are what sec.7.3's figures actually plot.
EXCEPTIONS = (203, 217, 219, 36, 44, 100, 104, 233, 29, 71, 235, 249)
#: multi-attractor at some N but not at N_max; see R9 sec.8.2.
OSCILLATING_EXCEPTIONS = (37,)


# --- the graph ---------------------------------------------------------------

def sector_graph(rule: int, N: int, bc: str = "obc0"):
    """
    (edges, sectors) for one unit.

    edges: sorted list of undirected (u, v), u < v, self-loops dropped -- the
    same filter the Julia applies when it writes the file.
    sectors: node lists, largest first, exactly the WCC partition of R9.
    """
    succ = wcc.make_succ(rule, N, bc)
    n = 1 << N
    adj: Dict[int, set] = {v: set() for v in range(n)}
    for x in range(n):
        for y in succ(x):
            if x != y:
                adj[x].add(y)
                adj[y].add(x)
    seen = [False] * n
    sectors: List[List[int]] = []
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
        sectors.append(sorted(comp))
    sectors.sort(key=len, reverse=True)
    edges = sorted({(min(u, v), max(u, v)) for u in adj for v in adj[u]})
    return edges, sectors, adj


# --- layout ------------------------------------------------------------------

def _bfs_distances(nodes: Sequence[int], adj: Dict[int, set]) -> np.ndarray:
    """All-pairs graph distance inside one sector."""
    idx = {v: i for i, v in enumerate(nodes)}
    m = len(nodes)
    D = np.full((m, m), np.inf)
    for i, s in enumerate(nodes):
        D[i, i] = 0.0
        q = deque([s])
        while q:
            v = q.popleft()
            for w in adj[v]:
                j = idx.get(w)
                if j is not None and not np.isfinite(D[i, j]):
                    D[i, j] = D[i, idx[v]] + 1.0
                    q.append(w)
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = (finite.max() if finite.size else 1.0) + 1.0
    return D


def stress_layout(nodes: Sequence[int], adj: Dict[int, set],
                  iters: int = 120, seed: int = 0) -> List[Tuple[float, float]]:
    """
    SMACOF stress majorization, the algorithm behind NetworkLayout's Stress().

    Minimises sum_{i<j} w_ij (||x_i - x_j|| - d_ij)^2 with w_ij = 1/d_ij^2, the
    usual weighting, starting from the classical-MDS solution so the result is
    deterministic rather than dependent on a random start.
    """
    m = len(nodes)
    if m == 1:
        return [(0.0, 0.0)]
    if m == 2:
        return [(-0.5, 0.0), (0.5, 0.0)]
    D = _bfs_distances(nodes, adj)
    # classical MDS start: double-centre -D^2/2 and take the top two eigenvectors
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    X = vecs[:, order] * np.sqrt(np.maximum(vals[order], 1e-12))
    if not np.all(np.isfinite(X)):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(m, 2))

    with np.errstate(divide="ignore"):
        W = 1.0 / (D ** 2)              # D_ii = 0 -> inf, zeroed on the next line
    np.fill_diagonal(W, 0.0)
    wsum = W.sum(axis=1)
    wsum[wsum == 0] = 1.0
    for _ in range(iters):
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        # two nodes can land on top of each other, and 1/0 there turns the whole
        # update into NaN; the floor keeps the step finite and lets them separate
        np.maximum(dist, 1e-9, out=dist)
        np.fill_diagonal(dist, 1.0)
        ratio = W * D / dist
        np.fill_diagonal(ratio, 0.0)
        Xn = (ratio[:, :, None] * diff).sum(axis=1) + (W @ X)
        X = Xn / wsum[:, None]
        if not np.all(np.isfinite(X)):
            raise FloatingPointError("stress layout diverged")
    X = X - X.mean(axis=0)
    return [(float(p[0]), float(p[1])) for p in X]


def _palette(n: int) -> List[str]:
    """
    A stand-in for Colors.jl `distinguishable_colors` seeded with white: evenly
    spaced hues at high saturation, so no entry comes out near-white.  Not the
    identical sequence -- Colors.jl solves a Lab-space packing -- but it serves
    the same purpose, which is that adjacent sectors never share a colour.
    """
    out = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        s = 0.72 + 0.18 * ((i % 3) / 2.0)
        v = 0.72 + 0.22 * ((i % 2))
        r, g, b = colorsys.hsv_to_rgb(h, s, min(v, 0.95))
        out.append("#%02X%02X%02X" % (round(r * 255), round(g * 255),
                                      round(b * 255)))
    return out


def _place(sorted_sectors, layouts, variant: str, num_central: int, seed: int):
    """
    Cluster centres and per-cluster scale, following the Julia line for line.
    Returns [(cx, cy, scale)] aligned with sorted_sectors.
    """
    rng = random.Random(seed)
    placed: List[Tuple[float, float, float]] = []
    out: List[Tuple[float, float, float]] = []
    scalefactor = 8.0
    theta = 0.0
    core_radius = 130.0

    for i, (sector, local) in enumerate(zip(sorted_sectors, layouts), start=1):
        max_dist = max(math.hypot(p[0], p[1]) for p in local) if local else 0.0

        if variant == "V3":
            if i == 1:
                scale = 6.0 * scalefactor
                cx = cy = 0.0
                placed.append((0.0, 0.0,
                               max(max_dist * scale + 10.0, 10.0)))
                out.append((cx, cy, scale))
                continue
            if i <= num_central:
                scale = 4.0 * scalefactor
                angle_step = 2 * math.pi / max(num_central - 1, 1)
                cx = core_radius * math.cos(theta)
                cy = core_radius * math.sin(theta)
                theta += angle_step
                placed.append((cx, cy, max(max_dist * scale + 10.0, 10.0)))
                out.append((cx, cy, scale))
                continue
            scale = 2.0 * scalefactor
        else:                                   # V4
            if i == 1:
                scale = 12.0 * scalefactor
            elif i <= 3:
                scale = 10.0 * scalefactor
            else:
                scale = 4.0 * scalefactor

        bounding = max(max_dist * scale + 10.0, 10.0)
        if not math.isfinite(bounding):
            raise FloatingPointError(f"cluster {i} has a non-finite extent")
        outer = 300.0
        attempts = 0
        # The Julia declares max_attempts = 1000 and then never reads it, so a
        # cluster that cannot be placed spins forever.  The arena grows every 50
        # attempts, so a real bound cannot cost a good placement -- it only stops
        # the loop being unbounded.
        while attempts < 200_000:
            rt = rng.random() * 2 * math.pi
            rr = math.sqrt(rng.random()) * outer
            tx, ty = rr * math.cos(rt), rr * math.sin(rt)
            if all(math.hypot(tx - px, ty - py) >= bounding + pr
                   for px, py, pr in placed):
                placed.append((tx, ty, bounding))
                out.append((tx, ty, scale))
                break
            attempts += 1
            if attempts % 50 == 0:
                outer += 10.0
        else:
            raise RuntimeError(f"could not place cluster {i} of {len(local)} "
                               "nodes after 200000 attempts")
    return out


# --- the .dot file -----------------------------------------------------------

def write_dot(rule: int, N: int, bc: str = "obc0", variant: str = "V4",
              num_central: int = 7, out_dir: Optional[str] = None,
              seed: Optional[int] = None) -> str:
    """Emit the Graphviz file.  Render it with `neato -n2 -Tpng`."""
    out_dir = out_dir or PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    edges, sectors, adj = sector_graph(rule, N, bc)
    layouts = [stress_layout(s, adj, seed=rule) for s in sectors]
    places = _place(sectors, layouts, variant, num_central,
                    seed=rule if seed is None else seed)
    pal = _palette(num_central)

    pos: Dict[int, Tuple[float, float]] = {}
    col: Dict[int, str] = {}
    wid: Dict[int, float] = {}
    ealpha: Dict[int, float] = {}
    central_alpha = 0.2 if variant == "V3" else 0.1
    for i, (sector, local, (cx, cy, scale)) in enumerate(
            zip(sectors, layouts, places), start=1):
        if i <= num_central:
            c, w, ea = pal[i - 1], 0.1, central_alpha
        else:
            c, w, ea = "#555555FF", 0.05, 0.4
        for node, p in zip(sector, local):
            pos[node] = (cx + p[0] * scale, cy + p[1] * scale)
            col[node], wid[node], ealpha[node] = c, w, ea

    name = f"Scattered_CorePeriphery_N{N}_{rule}_{variant}_{bc}.dot"
    path = os.path.join(out_dir, name)
    with open(path, "w") as io:
        io.write("graph G {\n")
        io.write("  layout=neato;\n")
        io.write('  bgcolor="white";\n')
        if variant == "V4":
            io.write('  outputorder="edgesfirst";\n')
        io.write("  node [shape=point, style=filled];\n")
        for v in range(1 << N):
            x, y = pos[v]
            io.write(f'  {v} [pos="{x},{y}!", width={wid[v]}, '
                     f'color="{col[v]}"];\n')
        for u, v in edges:
            a = format(round(ealpha[u] * 255), "02x")
            io.write(f'  {u} -- {v} [color="#aaaaaa{a}", penwidth=0.5];\n')
        io.write("}\n")
    return path


def write_both(rule: int, N: int, bc: str = "obc0", num_central: int = 7,
               out_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Both layouts for one rule, sharing one set of per-sector stress layouts.

    V3 and V4 differ only in where the clusters are PLACED; the intra-sector
    geometry is the same computation, and it is the expensive one.
    """
    out_dir = out_dir or PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    edges, sectors, adj = sector_graph(rule, N, bc)
    layouts = [stress_layout(s, adj, seed=rule) for s in sectors]
    pal = _palette(num_central)
    paths = {}
    for variant in ("V3", "V4"):
        places = _place(sectors, layouts, variant, num_central, seed=rule)
        pos, col, wid, ea = {}, {}, {}, {}
        central_alpha = 0.2 if variant == "V3" else 0.1
        for i, (sector, local, (cx, cy, sc)) in enumerate(
                zip(sectors, layouts, places), start=1):
            if i <= num_central:
                c, w, a = pal[i - 1], 0.1, central_alpha
            else:
                c, w, a = "#555555FF", 0.05, 0.4
            for node, p in zip(sector, local):
                pos[node] = (cx + p[0] * sc, cy + p[1] * sc)
                col[node], wid[node], ea[node] = c, w, a
        name = f"Scattered_CorePeriphery_N{N}_{rule}_{variant}_{bc}.dot"
        path = os.path.join(out_dir, name)
        with open(path, "w") as io:
            io.write("graph G {\n  layout=neato;\n  bgcolor=\"white\";\n")
            if variant == "V4":
                io.write('  outputorder="edgesfirst";\n')
            io.write("  node [shape=point, style=filled];\n")
            for v in range(1 << N):
                x, y = pos[v]
                io.write(f'  {v} [pos="{x},{y}!", width={wid[v]}, '
                         f'color="{col[v]}"];\n')
            for u, v in edges:
                a = format(round(ea[u] * 255), "02x")
                io.write(f'  {u} -- {v} [color="#aaaaaa{a}", penwidth=0.5];\n')
            io.write("}\n")
        paths[variant] = path
    return paths


# --- WCC and SCC together ----------------------------------------------------
#
# The WCC picture and the SCC picture answer different questions and R9 keeps
# them strictly apart: a weak component is a SECTOR (exact for the monitored and
# the unmonitored dynamics alike), a terminal SCC is a MONITORED ATTRACTOR.
# Three files per rule:
#
#   _wcc    the whole basis, positioned and coloured by weak component.
#   _scc    the RECURRENT SUBGRAPH only, positioned and coloured by terminal
#           SCC.  Transient states are not drawn: they belong to no attractor,
#           and for a dissipative rule they are almost the entire basis, so
#           drawing them would bury the object of interest.
#   _joint  the whole basis in the SAME positions as _wcc, coloured by which
#           attractor each state drains to -- recurrent states solid, transient
#           states faded.  A sector holding one attractor is one colour; a
#           sector holding several is visibly striped, which is R9 F11's twelve
#           multi-attractor rules made concrete.  Six of the 22 survivors are
#           among those twelve: 29, 71 (exponential) and 44, 100, 104, 233.

SHARED_COLOUR = "#000000"          # drains to more than one attractor


def decompositions(rule: int, N: int, bc: str = "obc0"):
    """(edges, wcc sectors, terminal SCCs, basin map, adjacency) for one unit."""
    from ..core import rules as rules_mod
    from ..graph import scc as scc_mod
    edges, sectors, adj = sector_graph(rule, N, bc)
    t = rules_mod.wolfram_to_tuple(rule)
    rec = scc_mod.recurrent_classes(rule, N, bc, t)
    rec.sort(key=len, reverse=True)

    # which attractor does each state reach?  Walk the graph FORWARDS from every
    # state is O(2^N * depth); instead walk BACKWARDS from each attractor once.
    succ = wcc.make_succ(rule, N, bc)
    pred: Dict[int, List[int]] = {v: [] for v in range(1 << N)}
    for x in range(1 << N):
        for y in succ(x):
            pred[y].append(x)
    basin: Dict[int, Optional[int]] = {}
    for k, cls in enumerate(rec):
        q = deque(cls)
        for v in cls:
            basin.setdefault(v, k)
        seen_here = set(cls)
        while q:
            v = q.popleft()
            for u in pred[v]:
                if u in seen_here:
                    continue
                seen_here.add(u)
                if u in basin and basin[u] != k:
                    basin[u] = -1                     # shared basin
                else:
                    basin.setdefault(u, k)
                q.append(u)
    return edges, sectors, rec, basin, adj


def _emit(path, nodes, pos, col, wid, edges, ealpha, variant):
    with open(path, "w") as io:
        io.write("graph G {\n  layout=neato;\n  bgcolor=\"white\";\n")
        if variant == "V4":
            io.write('  outputorder="edgesfirst";\n')
        io.write("  node [shape=point, style=filled];\n")
        for v in nodes:
            x, y = pos[v]
            io.write(f'  {v} [pos="{x},{y}!", width={wid[v]}, '
                     f'color="{col[v]}"];\n')
        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            a = format(round(ealpha[u] * 255), "02x")
            io.write(f'  {u} -- {v} [color="#aaaaaa{a}", penwidth=0.5];\n')
        io.write("}\n")
    return path


def write_dot_set(rule: int, N: int, bc: str = "obc0", variant: str = "V4",
                  num_central: int = 7,
                  out_dir: Optional[str] = None) -> Dict[str, str]:
    """The three diagrams for one rule.  Render each with `neato -n2 -Tpng`."""
    out_dir = out_dir or PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    edges, sectors, rec, basin, adj = decompositions(rule, N, bc)
    pal = _palette(max(num_central, len(rec)))
    stem = os.path.join(out_dir, f"N{N}_{rule}_{variant}_{bc}")
    paths = {}

    # --- 1. the sector picture ---
    lay = [stress_layout(s, adj, seed=rule) for s in sectors]
    place = _place(sectors, lay, variant, num_central, seed=rule)
    pos, col, wid, ea = {}, {}, {}, {}
    for i, (sec, loc, (cx, cy, sc)) in enumerate(zip(sectors, lay, place), 1):
        c, w, a = ((pal[i - 1], 0.1, 0.2 if variant == "V3" else 0.1)
                   if i <= num_central else ("#555555FF", 0.05, 0.4))
        for node, p in zip(sec, loc):
            pos[node] = (cx + p[0] * sc, cy + p[1] * sc)
            col[node], wid[node], ea[node] = c, w, a
    paths["wcc"] = _emit(f"{stem}_wcc.dot", range(1 << N), pos, col, wid,
                         edges, ea, variant)

    # --- 3. the joint picture: same geometry, attractor colouring ---
    jcol, jwid = {}, {}
    recset = {v: k for k, cls in enumerate(rec) for v in cls}
    for v in range(1 << N):
        k = basin.get(v)
        if k is None:
            base = "#cccccc"                       # reaches no attractor
        elif k == -1:
            base = SHARED_COLOUR
        else:
            base = pal[k % len(pal)]
        if v in recset:
            jcol[v], jwid[v] = base + "FF", 0.11   # recurrent: solid, bigger
        else:
            jcol[v], jwid[v] = base + "55", 0.05   # transient: faded
    paths["joint"] = _emit(f"{stem}_joint.dot", range(1 << N), pos, jcol, jwid,
                           edges, ea, variant)

    # --- 2. the attractor picture: recurrent subgraph alone ---
    if rec:
        rlay = [stress_layout(c, adj, seed=rule) for c in rec]
        rplace = _place(rec, rlay, variant, num_central, seed=rule + 1)
        rpos, rcol, rwid, rea = {}, {}, {}, {}
        for i, (cls, loc, (cx, cy, sc)) in enumerate(zip(rec, rlay, rplace), 1):
            c = pal[(i - 1) % len(pal)]
            for node, p in zip(cls, loc):
                rpos[node] = (cx + p[0] * sc, cy + p[1] * sc)
                rcol[node], rwid[node], rea[node] = c, 0.12, 0.5
        rnodes = sorted(rpos)
        paths["scc"] = _emit(f"{stem}_scc.dot", rnodes, rpos, rcol, rwid,
                             edges, rea, variant)
    return paths


def group_table(groups, N: int = 11, bc: str = "obc0") -> List[Dict]:
    """
    What the pictures show, as numbers: sectors, attractors, and how the two
    relate, for each R9 survivor.  `att_per_sector` > 1 is the case R9 F11
    isolates -- a weak component holding more than one terminal SCC, which is
    what makes the joint diagram striped rather than solid.
    """
    from collections import Counter
    rows = []
    for kind, group in groups:
        for rule in group:
            _, sectors, rec, basin, _ = decompositions(rule, N, bc)
            owner = {v: i for i, s in enumerate(sectors) for v in s}
            per = Counter(owner[c[0]] for c in rec)
            recurrent = sum(len(c) for c in rec)
            rows.append({
                "rule": rule, "kind": kind, "n_wcc": len(sectors),
                "d_max": len(sectors[0]), "n_scc": len(rec),
                "d_max_scc": max((len(c) for c in rec), default=0),
                "recurrent": recurrent,
                "recurrent_frac": recurrent / (1 << N),
                "max_att_per_sector": max(per.values(), default=0),
                "multi_sectors": sum(1 for v in per.values() if v > 1),
                "shared": sum(1 for v in basin.values() if v == -1),
            })
    return rows


def survivor_table(N: int = 11, bc: str = "obc0") -> List[Dict]:
    return group_table((("exponential", EXPONENTIAL_SURVIVORS),
                        ("polynomial", POLYNOMIAL_SURVIVORS)), N, bc)


def exception_table(N: int = 11, bc: str = "obc0") -> List[Dict]:
    """The twelve of sec.7.3, tagged by whether the rule is also a survivor."""
    surv = set(EXPONENTIAL_SURVIVORS) | set(POLYNOMIAL_SURVIVORS)
    rows = group_table((("exception", EXCEPTIONS + OSCILLATING_EXCEPTIONS),),
                       N, bc)
    for r in rows:
        if r["rule"] in OSCILLATING_EXCEPTIONS:
            r["kind"] = "osc"                     # exception at this N only
        else:
            r["kind"] = "both" if r["rule"] in surv else "exc"
    return rows


def write_table(N: int = 11, bc: str = "obc0", out: Optional[str] = None,
                rows=None, stem: str = "graphs") -> str:
    rows = survivor_table(N, bc) if rows is None else rows
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex",
                              f"tab_r9_{stem}_{bc}.tex")
    with open(out, "w") as f:
        f.write("\\begin{tabular}{rlrrrrrr}\n\\hline\n")
        f.write("rule & class & $n_{\\rm wcc}$ & $D_{\\max}$ & $n_{\\rm scc}$ "
                "& $|{\\rm Rec}|$ & max att/sector & shared \\\\\n\\hline\n")
        for r in rows:
            f.write(f"${r['rule']}$ & {r['kind'][:4]} & ${r['n_wcc']}$ & "
                    f"${r['d_max']}$ & ${r['n_scc']}$ & ${r['recurrent']}$ & "
                    f"${r['max_att_per_sector']}$ & ${r['shared']}$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return out


def main(argv=None):
    import argparse
    import subprocess
    import time
    ap = argparse.ArgumentParser(
        description="R9 survivor sector graphs (render with neato -n2 -Tpng)")
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--variant", default="V4", choices=["V3", "V4"])
    ap.add_argument("--rules", default="survivors")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    if a.rules == "survivors":
        todo = ([("exp", r) for r in EXPONENTIAL_SURVIVORS]
                + [("poly", r) for r in POLYNOMIAL_SURVIVORS])
    elif a.rules == "exceptions":
        todo = ([("exc", r) for r in EXCEPTIONS]
                + [("osc", r) for r in OSCILLATING_EXCEPTIONS])
    elif a.rules == "all":
        seen, todo = set(), []
        for tag, group in (("exp", EXPONENTIAL_SURVIVORS),
                           ("poly", POLYNOMIAL_SURVIVORS),
                           ("exc", EXCEPTIONS),
                           ("osc", OSCILLATING_EXCEPTIONS)):
            for r in group:
                if r not in seen:            # six rules are in both groups
                    seen.add(r)
                    todo.append((tag, r))
    else:
        todo = [("custom", int(x)) for x in a.rules.split(",")]
    for kind, rule in todo:
        t0 = time.time()
        paths = write_dot_set(rule, a.N, a.bc, a.variant)
        if not a.no_render:
            for p in paths.values():
                subprocess.run(["neato", "-n2", "-Tpng", p, "-o",
                                p.replace(".dot", ".png")], check=True)
        print(f"W{rule:<4} {kind:<5} {time.time() - t0:6.1f}s  "
              + ", ".join(sorted(paths)), flush=True)


if __name__ == "__main__":
    main()
