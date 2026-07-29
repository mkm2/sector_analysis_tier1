"""
Tier 1e sweep driver: the WCC (sector) decomposition over the rule space.

    python -m qca_fragmentation.wcc_sweep --which all --bc obc0 --n-min 6 --n-max 18
    python -m qca_fragmentation.wcc_sweep --rules 150,156 --bc obc0 --n-max 20

Same operational contract as the Tier-1a sweep (PLAN.md sec.4, sec.6):
append-only JSONL, one record per (rule, N, bc), a completed unit is never
recomputed, safe to kill and restart at any point.  Results go to their own
store (results_tier1e/) so the Tier-1a/1b/1d records stay untouched.

Two differences from the Tier-1a sweep, both deliberate:

  * the pass is always run to completion, so the A1 sum rule holds and the
    hyperbola anchors are exact (see graph/wcc.weak_components);
  * `ergodic` is therefore a classification, and it is used only to stop
    ENLARGING N -- the exponential saving -- not to truncate a scan.

The Tarjan node_budget is not inherited: union-find has no DFS stack.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional

from . import results_io
from .core import rules
from .graph import wcc


def run_wcc_unit(rule: int, N: int, bc: str, *, f_erg: float = 0.5,
                 force: bool = False, quiet: bool = False,
                 strict: bool = True) -> dict:
    """One (rule, N, bc) unit, cached and checkpointed."""
    if not force and results_io.has_wcc_unit(rule, bc, N):
        if not quiet:
            print(f"skip  W{rule} N{N} {bc} (cached)")
        return results_io.load_wcc_results(rule, bc)[N]
    t = rules.wolfram_to_tuple(rule)
    res = wcc.weak_components(rule, N, bc, t, f_erg=f_erg)
    scc_rec = results_io.load_results(rule, bc).get(N)
    checks = {}
    if scc_rec:
        try:
            checks = wcc.check_against_scc(res, scc_rec)
        except AssertionError as exc:
            if strict:
                raise
            checks = {"failed": str(exc)}
    rec = results_io.record_from_wcc_result(res, scc_rec, checks)
    results_io.append_wcc_result(rec)
    results_io.append_manifest(rule, bc, N, res.runtime, res.ergodic,
                               extra={"tier": "1e",
                                      "tier1e_version": results_io.TIER1E_VERSION,
                                      "n_wcc": res.n_wcc,
                                      "d_max_wcc": res.d_max_wcc})
    if not quiet:
        flag = " ERGODIC" if res.ergodic else ""
        print(f"done  W{rule} N{N} {bc}: n_wcc={res.n_wcc} "
              f"Dmax={res.d_max_wcc} frozen={res.n_frozen}{flag} "
              f"[{res.runtime:.1f}s]")
    return rec


#: Do not let a rule be written off as ergodic on the strength of a tiny chain.
#: At N=6 there are only 64 states and almost anything connects up past half the
#: space; a rule can look ergodic there and fragment later.  We therefore
#: require the ergodic verdict to hold at TWO CONSECUTIVE N, and never before
#: this N, before skipping larger sizes.  This costs a handful of cheap units
#: and protects the sector map from a small-N artefact.
STOP_MIN_N = 10
STOP_STREAK = 2


def sweep_rule(rule: int, bcs: Iterable[str], n_min: int, n_max: int, *,
               f_erg: float = 0.5, force: bool = False,
               stop_on_ergodic: bool = True, quiet: bool = False,
               strict: bool = True, stop_min_n: int = STOP_MIN_N,
               stop_streak: int = STOP_STREAK):
    for bc in bcs:
        streak = 0
        for N in range(n_min, n_max + 1):
            rec = run_wcc_unit(rule, N, bc, f_erg=f_erg, force=force,
                               quiet=quiet, strict=strict)
            streak = streak + 1 if rec.get("ergodic_flag") else 0
            if (stop_on_ergodic and N >= stop_min_n
                    and streak >= stop_streak):
                if not quiet:
                    print(f"  -> W{rule} {bc} ergodic at N={N} "
                          f"({streak} in a row); skipping larger N")
                break


def rule_set(which: str, explicit: Optional[List[int]]) -> List[int]:
    if explicit:
        return explicit
    if which == "unitary":
        return list(rules.UNITARY_RULES)
    if which == "vfree":
        return [r for r in range(256)
                if "V" not in rules.wolfram_to_tuple(r)]
    if which == "dissipative":
        return [r for r in range(256)
                if not rules.is_unitary(rules.wolfram_to_tuple(r))]
    if which == "all":
        return list(range(256))
    raise ValueError(f"unknown rule set {which!r}")


def sweep_n_major(rs: List[int], bcs: Iterable[str], n_min: int, n_max: int, *,
                  f_erg: float = 0.5, force: bool = False, quiet: bool = False,
                  strict: bool = True):
    """
    Sweep N in the OUTER loop: every rule at N, then every rule at N+1, ...

    Rule-major order is wrong for an unattended run with a wall-clock budget:
    stopping half way leaves the first rules complete and the rest with no data
    at all, and the sector map needs a descriptor for every rule.  N-major order
    means that whenever the run is stopped, coverage is UNIFORM -- every rule has
    the same N range and the fits are comparable across the whole rule space.
    Each N is also a natural checkpoint boundary.
    """
    for bc in bcs:
        for N in range(n_min, n_max + 1):
            t0 = time.time()
            for i, rule in enumerate(rs, 1):
                run_wcc_unit(rule, N, bc, f_erg=f_erg, force=force,
                             quiet=True, strict=strict)
                if not quiet and i % 32 == 0:
                    print(f"    N={N} {bc}: {i}/{len(rs)} rules "
                          f"[{time.time() - t0:.0f}s]", flush=True)
            print(f"  == N={N} {bc} complete: {len(rs)} rules "
                  f"[{time.time() - t0:.0f}s] ==", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1e WCC / sector sweep")
    ap.add_argument("--which", default="all",
                    choices=["all", "unitary", "dissipative", "vfree"])
    ap.add_argument("--rules", default=None,
                    help="explicit comma-separated Wolfram numbers")
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="obc0")
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=18)
    ap.add_argument("--f-erg", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-stop-on-ergodic", action="store_true",
                    help="carry ergodic-flagged rules to the same N as the "
                         "rest, so every rule has a comparable series")
    ap.add_argument("--order", choices=["rule", "n"], default="n",
                    help="'n' (default) sweeps N in the outer loop, so a run "
                         "stopped early still has uniform coverage")
    ap.add_argument("--lax", action="store_true",
                    help="record a failed A2/A3/A4 check instead of raising")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    explicit = ([int(x) for x in args.rules.split(",")] if args.rules else None)
    rs = rule_set(args.which, explicit)
    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]
    t0 = time.time()
    if args.order == "n":
        sweep_n_major(rs, bcs, args.n_min, args.n_max, f_erg=args.f_erg,
                      force=args.force, quiet=args.quiet,
                      strict=not args.lax)
    else:
        for i, rule in enumerate(rs, 1):
            if not args.quiet:
                print(f"[{i}/{len(rs)}] W{rule}", flush=True)
            sweep_rule(rule, bcs, args.n_min, args.n_max, f_erg=args.f_erg,
                       force=args.force,
                       stop_on_ergodic=not args.no_stop_on_ergodic,
                       quiet=args.quiet, strict=not args.lax)
    print(f"\nsweep finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
