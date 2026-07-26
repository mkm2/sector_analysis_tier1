"""
Rule 150 = (I, V, V, I): the exactly solved member of the unitary family.

Rule 150 fires the Hadamard at site i exactly when its two neighbours differ,
t[01] = t[10] = V and t[00] = t[11] = I, i.e.

    control(i) = x_{i-1} XOR x_{i+1}         (the classical ECA-90 image)

Combined with the flip reduction of graph/flip_graph.py the whole sector
structure collapses to a statement about DOMAIN WALLS.  Write the bond variables

    obc0:  b_j = x_j XOR x_{j+1},  j = 0..N   with x_0 = x_{N+1} = 0   (N+1 bonds)
    pbc :  b_j = x_j XOR x_{j+1},  j = 0..N-1 (indices mod N)          (N   bonds)

Then:

  * obc0:  x |-> b is a BIJECTION from {0,1}^N onto the even-weight strings of
    length N+1 (recover x by prefix XOR), and
  * pbc:   x |-> b is 2-to-1 onto the even-weight strings of length N (the two
    preimages are x and its global spin flip).

Flipping x_i toggles exactly the two bonds b_{i-1}, b_i, and it is allowed
(symbol V) precisely when b_{i-1} != b_i.  So an elementary move is

    (b_{i-1}, b_i) = (1,0) <-> (0,1) :   a HARD-CORE HOP of one wall

which conserves the wall number w = sum_j b_j exactly and, since hard-core
hopping on a segment (resp. a ring) is connected on each fixed-occupancy shell,
connects every pair of configurations with the same w.  Hence

  obc0:  sectors = { w = 0, 2, 4, ... },   |S_w| = C(N+1, w),
         #sectors = floor((N+1)/2) + 1,    D_max = max_{w even} C(N+1, w);

  pbc :  for 0 < w < N the wall shell lifts to ONE sector of size 2 C(N, w);
         w = 0 (and w = N when N is even) are the frozen shells -- no move is
         available -- so each splits into TWO singletons, giving
         #sectors = floor(N/2) + 1 + 2*[N even] + 1*[N odd] ... see
         sector_sizes_pbc for the exact enumeration.

The conserved charge is the Ising domain-wall number, a strictly 2-local
diagonal operator

    Q = sum_j (1 - Z_j Z_{j+1}) / 2     (obc0: j = 0..N, Z_0 = Z_{N+1} = +1),

so rule 150 conserves the open Ising energy with boundary fields.  Because the
number of sectors is LINEAR in N and each sector is a full Q-eigenspace, rule
150 is *symmetry-resolved, not fragmented*: there is no Krylov structure beyond
the U(1) charge.  It is the natural null model against which the genuinely
fragmented rules (204's 2^N frozen singletons, 51/201's wall families) should be
compared.

CLI
    python -m qca_fragmentation.c150 --frontier 21-32 --bc obc0
    python -m qca_fragmentation.c150 --verify
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import comb
from typing import Dict, List, Optional

from .core import rules
from .graph import flip_graph
from .results_io import REPO_ROOT

RULE = 150
TUPLE = ("I", "V", "V", "I")

ANALYTICS_DIR = os.path.join(REPO_ROOT, "analytics")
FRONTIER_PATH = os.path.join(ANALYTICS_DIR, "c150_frontier.jsonl")


# --- wall (bond) variables ---------------------------------------------------

def wall_string(x: int, N: int, bc: str) -> int:
    """Bond bitmask b (bit j = b_j).  obc0: N+1 bonds; pbc: N bonds."""
    if bc == "obc0":
        out = 0
        prev = 0                       # x_0 = 0
        for j in range(N + 1):
            cur = (x >> j) & 1 if j < N else 0      # x_{j+1}, with x_{N+1} = 0
            if prev ^ cur:
                out |= 1 << j
            prev = cur
        return out
    if bc == "pbc":
        out = 0
        for j in range(N):
            if ((x >> j) & 1) ^ ((x >> ((j + 1) % N)) & 1):
                out |= 1 << j
        return out
    raise ValueError(f"unknown boundary convention {bc!r}")


def wall_number(x: int, N: int, bc: str) -> int:
    return bin(wall_string(x, N, bc)).count("1")


def spin_from_walls(b: int, N: int) -> int:
    """obc0 inverse of wall_string: x_j = XOR of b_0..b_{j-1} (prefix XOR)."""
    x = 0
    cur = 0
    for j in range(N):
        cur ^= (b >> j) & 1            # x_{j+1} = x_j XOR b_j, sites 1-indexed
        if cur:
            x |= 1 << j
    return x


def states_with_walls(N: int, w: int, bc: str = "obc0") -> List[int]:
    """All basis states of the wall-number-w sector, ascending (obc0 only)."""
    if bc != "obc0":
        raise NotImplementedError("enumeration is implemented for obc0")
    out = []
    for x in range(1 << N):
        if wall_number(x, N, "obc0") == w:
            out.append(x)
    return out


# --- closed forms ------------------------------------------------------------

def sector_sizes_obc0(N: int) -> List[int]:
    """Exact sector-size multiset, descending: C(N+1, w) for every even w."""
    return sorted((comb(N + 1, w) for w in range(0, N + 2, 2)), reverse=True)


def sector_sizes_pbc(N: int) -> List[int]:
    """Exact sector-size multiset, descending, for the ring."""
    out: List[int] = []
    for w in range(0, N + 1, 2):
        if w == 0 or w == N:           # frozen shell: two singleton sectors
            out += [1, 1]
        else:
            out.append(2 * comb(N, w))
    return sorted(out, reverse=True)


def sector_sizes(N: int, bc: str) -> List[int]:
    return sector_sizes_obc0(N) if bc == "obc0" else sector_sizes_pbc(N)


def n_sectors_obc0(N: int) -> int:
    return (N + 1) // 2 + 1


def d_max_obc0(N: int) -> int:
    return max(comb(N + 1, w) for w in range(0, N + 2, 2))


def d_max_argmax_obc0(N: int) -> int:
    return max(range(0, N + 2, 2), key=lambda w: comb(N + 1, w))


def d_max_ratio_obc0(N: int) -> float:
    """D_max / 2^N.  Asymptotically sqrt(2/(pi(N+1)))*2 -> base exactly 2 with a
    N^{-1/2} prefactor, which is why a pure-exponential fit reports 2.02."""
    return d_max_obc0(N) / float(1 << N)


# --- frontier runner ---------------------------------------------------------

def load_frontier() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not os.path.exists(FRONTIER_PATH):
        return out
    with open(FRONTIER_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[f"{rec['N']}_{rec['bc']}"] = rec
    return out


def _append(rec: dict) -> None:
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    with open(FRONTIER_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def frontier_unit(N: int, bc: str, *, force: bool = False,
                  chunk: int = 1 << 26) -> dict:
    """
    One flip-graph frontier unit: exact sector decomposition at length N,
    checked against the closed form.  Checkpointed to analytics/.
    """
    have = load_frontier()
    key = f"{N}_{bc}"
    if not force and key in have:
        print(f"skip  C150 N{N} {bc} (cached)")
        return have[key]
    t0 = time.time()
    sizes = flip_graph.flip_components_np(N, TUPLE, bc, chunk=chunk)
    dt = time.time() - t0
    pred = sector_sizes(N, bc)
    rec = {
        "rule": RULE, "bc": bc, "N": N,
        "n_sectors": len(sizes),
        "d_max": sizes[0],
        "n_sectors_closed_form": len(pred),
        "d_max_closed_form": pred[0],
        "closed_form_exact": sizes == pred,
        "d_max_over_2N": sizes[0] / float(1 << N),
        "sizes": sizes,
        "runtime": dt,
        "method": "flip-graph label propagation (uint32, in-place)",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _append(rec)
    print(f"done  C150 N{N} {bc}: n_sec={len(sizes)} Dmax={sizes[0]} "
          f"closed_form={'OK' if rec['closed_form_exact'] else 'MISMATCH'} "
          f"[{dt:.1f}s]", flush=True)
    return rec


def verify_against_engine(max_N: int = 14) -> dict:
    """
    Cross-check the closed form against the streamed engine on every stored
    Tier-1a unit, and against the flip graph up to max_N.
    """
    from . import results_io
    from .graph import scc

    bad = []
    checked = 0
    for bc in ("obc0", "pbc"):
        for N, rec in sorted(results_io.load_results(RULE, bc).items()):
            sizes = results_io.sizes_from_record(rec)
            if not sizes:
                continue
            checked += 1
            if sizes != sector_sizes(N, bc):
                bad.append(("stored", bc, N))
    for bc in ("obc0", "pbc"):
        for N in range(3, max_N + 1):
            eng = scc.analyze(RULE, N, bc, TUPLE, detect_ergodic=False)
            checked += 1
            if eng.sizes_recurrent != sector_sizes(N, bc):
                bad.append(("engine", bc, N))
    # charge conservation at the support level (=> [U, Q] = 0 exactly, Q diagonal)
    from .core.cycle import succ
    viol = 0
    for bc in ("obc0", "pbc"):
        for N in range(3, 13):
            for x in range(1 << N):
                qx = wall_number(x, N, bc)
                for y in succ(x, N, TUPLE, bc):
                    if wall_number(y, N, bc) != qx:
                        viol += 1
    return {"units_checked": checked, "mismatches": bad,
            "charge_violations": viol}


def main(argv=None):
    ap = argparse.ArgumentParser(description="rule 150: exact sector structure")
    ap.add_argument("--frontier", type=str, default=None,
                    help="single N or 'lo-hi' for the flip-graph frontier")
    ap.add_argument("--bc", choices=["obc0", "pbc"], default="obc0")
    ap.add_argument("--chunk-log2", type=int, default=26)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        out = verify_against_engine()
        print(json.dumps(out, indent=2))
        assert not out["mismatches"] and out["charge_violations"] == 0
        print("VERIFIED")
    if args.frontier:
        if "-" in args.frontier:
            lo, hi = args.frontier.split("-")
            Ns = list(range(int(lo), int(hi) + 1))
        else:
            Ns = [int(args.frontier)]
        for N in Ns:
            frontier_unit(N, args.bc, force=args.force,
                          chunk=1 << args.chunk_log2)


if __name__ == "__main__":
    main()
