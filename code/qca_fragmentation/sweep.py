"""
Checkpointed, idempotent Tier-1a sweep driver (PLAN.md sec.4, sec.6).

Designed for unattended overnight runs:
  - one JSONL record per (rule, N, bc); a completed unit is never recomputed;
  - safe to kill and restart at any time (append-only results + manifest);
  - per-unit node budget so a single (near-)ergodic rule cannot eat the night;
  - once a (rule, bc) is ergodic / budget-capped at some N, larger N are skipped
    (the fragmentation cannot reappear at larger N for these rules).

Usage:
  python -m qca_fragmentation.sweep --which unitary --n-min 6 --n-max 20
  python -m qca_fragmentation.sweep --rules 150,201,156 --bc obc0 --n-max 22
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List

from .core import rules
from . import results_io
from .run_rule import run_unit


def sweep_rule(rule: int, bcs: Iterable[str], n_min: int, n_max: int, *,
               f_erg: float, node_budget, force: bool,
               stop_on_ergodic: bool = True, quiet: bool = False):
    for bc in bcs:
        for N in range(n_min, n_max + 1):
            rec = run_unit(rule, N, bc, f_erg=f_erg, node_budget=node_budget,
                           force=force, quiet=quiet)
            if stop_on_ergodic and rec.get("ergodic_flag"):
                if not quiet:
                    print(f"  -> W{rule} {bc} ergodic at N={N}; skipping larger N")
                break


def rule_set(which: str, explicit: List[int]):
    if explicit:
        return explicit
    if which == "unitary":
        # 10 reflection representatives of the 16 unitary rules
        return rules.unitary_reflection_reps()
    if which == "unitary-all":
        return rules.UNITARY_RULES
    if which == "all":
        # all 256 rules reduced to reflection representatives (a WORK grouping
        # only; the dissipative reduction is not a size symmetry -- see
        # rules.reflect_tuple).
        return rules.reflection_pairs(range(256))
    if which == "all256":
        # every rule 0..255 (reflection is NOT a sector-size symmetry for
        # dissipative rules under the even-first convention, so do not reduce).
        return list(range(256))
    if which == "dissipative":
        # only the rules that contain a D or E channel (208 of 256).
        return [r for r in range(256)
                if not rules.is_unitary(rules.wolfram_to_tuple(r))]
    raise ValueError(which)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1a checkpointed sweep")
    ap.add_argument("--which",
                    choices=["unitary", "unitary-all", "all", "all256",
                             "dissipative"],
                    default="unitary")
    ap.add_argument("--rules", type=str, default="",
                    help="comma-separated Wolfram numbers (overrides --which)")
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="both")
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=20)
    # 0.9 (not the context default 0.5): a sector must span >90% of the space to
    # be called "effectively ergodic at this N".  0.5 misclassifies fragmented
    # rules whose central-binomial dominant sector transiently exceeds half the
    # space at small N (e.g. rule 150 at N=6: Dmax=34/64).  See run notes.
    ap.add_argument("--f-erg", type=float, default=0.9)
    ap.add_argument("--node-budget", type=int, default=5_000_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-stop-on-ergodic", action="store_true")
    args = ap.parse_args(argv)

    explicit = [int(x) for x in args.rules.split(",") if x.strip()] if args.rules else []
    ruleset = rule_set(args.which, explicit)
    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]

    t0 = time.time()
    print(f"sweep: {len(ruleset)} rules x {len(bcs)} bc, N={args.n_min}..{args.n_max}, "
          f"node_budget={args.node_budget}, engine={results_io.ENGINE_VERSION}",
          flush=True)
    for i, rule in enumerate(ruleset):
        print(f"[{time.time()-t0:7.1f}s] ({i+1}/{len(ruleset)}) rule W{rule} "
              f"{rules.wolfram_to_tuple(rule)}", flush=True)
        sweep_rule(rule, bcs, args.n_min, args.n_max,
                   f_erg=args.f_erg, node_budget=args.node_budget,
                   force=args.force,
                   stop_on_ergodic=not args.no_stop_on_ergodic)
    print(f"[{time.time()-t0:7.1f}s] sweep complete.", flush=True)


if __name__ == "__main__":
    main()
