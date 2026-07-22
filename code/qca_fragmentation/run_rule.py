"""
CLI to analyze a single QCA rule over a range of chain lengths (Tier 1a).

    python -m qca_fragmentation.run_rule --rule 150 --N 8-16 --bc obc0
    python -m qca_fragmentation.run_rule --rule 201 --N 12 --bc both --force
    python -m qca_fragmentation.run_rule --rule 54 --N 6-13 --no-ergodic-abort --force

Writes one JSONL record per (rule, N, bc) to results/{rule}_{bc}.jsonl and an
audit entry to checkpoints/manifest.jsonl.  Idempotent: a unit already present
for the current engine_version is skipped unless --force.
"""

from __future__ import annotations

import argparse
import time

from .core import rules
from .graph import scc
from . import results_io


def parse_N(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def parse_bc(spec: str):
    return ["pbc", "obc0"] if spec == "both" else [spec]


def run_unit(rule: int, N: int, bc: str, *, f_erg: float, node_budget,
             force: bool, quiet: bool = False, detect_ergodic: bool = True) -> dict:
    if not force and results_io.has_unit(rule, bc, N):
        if not quiet:
            print(f"skip  W{rule} N{N} {bc} (cached)")
        return results_io.load_results(rule, bc)[N]
    t = rules.wolfram_to_tuple(rule)
    t0 = time.time()
    res = scc.analyze(rule, N, bc, t, f_erg=f_erg, detect_ergodic=detect_ergodic,
                      node_budget=node_budget)
    dt = time.time() - t0
    rec = results_io.record_from_graph_result(res, dt)
    if not detect_ergodic and not res.ergodic:
        # The abort was switched off only to LEARN the sector structure of an
        # ergodic-looking rule, not to reclassify it: keep the flag the sweep
        # uses to stop enlarging N, but now with the full decomposition beside
        # it, so "ergodic" can be read as "1 sector of size 2^N" or exposed as
        # "one huge sector plus a handful of small ones" at larger N.
        big = res.sizes_recurrent[0]
        if big > f_erg * (1 << N):
            rec["ergodic_flag"] = True
            rec["ergodic_bound"] = big
    results_io.append_result(rec)
    results_io.append_manifest(rule, bc, N, dt, res.ergodic)
    if not quiet:
        if res.ergodic:
            print(f"done  W{rule} N{N} {bc}: ERGODIC bound>{res.ergodic_bound} "
                  f"[{dt:.1f}s]")
        else:
            print(f"done  W{rule} N{N} {bc}: n_rec={res.n_recurrent} "
                  f"Dmax={res.sizes_recurrent[0]} depth={res.transient_depth} "
                  f"[{dt:.1f}s]")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1a single-rule analysis")
    ap.add_argument("--rule", type=int, required=True, help="Wolfram number 0..255")
    ap.add_argument("--N", type=str, required=True, help="single N or 'lo-hi'")
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="both")
    ap.add_argument("--f-erg", type=float, default=0.5)
    ap.add_argument("--node-budget", type=int, default=None,
                    help="abort a unit after exploring this many nodes")
    ap.add_argument("--tiers", default="1a")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-ergodic-abort", action="store_true",
                    help="do the FULL decomposition even when one component "
                         "exceeds f_erg*2^N, so an ergodic-looking rule still "
                         "reports how many sectors it really has")
    args = ap.parse_args(argv)

    for bc in parse_bc(args.bc):
        for N in parse_N(args.N):
            run_unit(args.rule, N, bc, f_erg=args.f_erg,
                     node_budget=args.node_budget, force=args.force,
                     detect_ergodic=not args.no_ergodic_abort)


if __name__ == "__main__":
    main()
