"""
Sweep the 256 X-gate rules.

    python -m qca_fragmentation.permutation.sweep --bc obc0 --n-min 6 --n-max 20

Same operational contract as the other sweeps: append-only JSONL, one record per
(rule, N, bc), cached units never recomputed, safe to interrupt.  N is the outer
loop so a run stopped early still leaves uniform coverage.

There is no ergodic early exit and no cost tier.  The whole analysis is a single
O(2^N) pass over a functional graph, so every rule costs the same and the only
limit is memory.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List

from .. import results_io
from ..core import rules
from . import functional


def run_unit(rule: int, N: int, bc: str, *, force: bool = False,
             quiet: bool = False) -> dict:
    if not force and results_io.has_xgate_unit(rule, bc, N):
        return results_io.load_xgate_results(rule, bc)[N]
    res = functional.analyze(rule, N, bc)
    rec = results_io.record_from_xgate(res)
    results_io.append_xgate_result(rec)
    if not quiet:
        print(f"done  X{rule} N{N} {bc}: n_wcc={rec['n_wcc']} "
              f"Dmax={rec['d_max_wcc']} n_rec={rec['n_recurrent']} "
              f"tf={rec['transient_fraction']:.3f} [{res['runtime']:.1f}s]",
              flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="X-gate (permutation) QCA sweep")
    ap.add_argument("--bc", choices=["pbc", "obc0"], default="obc0")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    rs = ([int(x) for x in args.rules.split(",")] if args.rules
          else list(range(256)))
    t0 = time.time()
    for N in range(args.n_min, args.n_max + 1):
        tn = time.time()
        for i, rule in enumerate(rs, 1):
            run_unit(rule, N, args.bc, force=args.force, quiet=True)
            if not args.quiet and i % 64 == 0:
                print(f"    N={N}: {i}/{len(rs)} [{time.time()-tn:.0f}s]",
                      flush=True)
        print(f"  == N={N} {args.bc} complete: {len(rs)} rules "
              f"[{time.time()-tn:.0f}s] ==", flush=True)
    print(f"\nsweep finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
