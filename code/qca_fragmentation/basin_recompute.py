"""
Independent recomputation of the basin decomposition, where the Tier-1a archive
cannot answer the sum rule.

WHY.  Tier 1a stores sizes_basins capped at 2048 entries and never stored a
basin histogram (unlike sizes_recurrent, which has size_hist).  For a unit whose
basin list hit that cap the multiset is genuinely lost, so

    sum(sizes_basins) + shared_basin_size == 2^N

is not checkable from the archive -- 76 obc0 units are affected, all of them
truncated ones.  Asserting the check away would be wrong (it hides a real gap);
recording None forever would leave the Tier-1e validation panel with a hole.  So
we simply redo those units from scratch and store the answer ALONGSIDE, in
results_basins/, this time with the histogram.

The recomputation is independent in the sense that matters: it re-runs the
directed analysis from the rule definition rather than trusting any stored
number, and it cross-checks the fields Tier 1a did keep intact (n_scc,
n_recurrent, shared_basin_size, transient_depth) as `agrees_with_tier1a`.

Cost is modest -- the recorded Tier-1a runtimes for the 76 obc0 units total
about 0.9 h -- and the pass is append-only, cached and resumable like every
other sweep here.

    python -m qca_fragmentation.basin_recompute --bc obc0            # the gaps
    python -m qca_fragmentation.basin_recompute --bc obc0 --all-units
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import List, Optional, Tuple

from . import results_io
from .core import rules
from .graph import scc

_CAP = results_io._SIZES_CAP


def needs_recompute(rule: int, bc: str, N: int, rec: dict) -> bool:
    """True iff the archive cannot answer the basin sum rule for this unit."""
    if rec.get("ergodic_flag") or rec.get("sizes_basins") is None:
        return False
    if results_io.has_basin_unit(rule, bc, N):
        return False
    basins = rec.get("sizes_basins")
    return len(basins) >= _CAP


def gaps(bc: str, rule_range=range(256)) -> List[Tuple[int, int]]:
    out = []
    for rule in rule_range:
        for N, rec in sorted(results_io.load_results(rule, bc).items()):
            if needs_recompute(rule, bc, N, rec):
                out.append((rule, N))
    return out


def recompute_unit(rule: int, N: int, bc: str, *, force: bool = False,
                   quiet: bool = False) -> Optional[dict]:
    if not force and results_io.has_basin_unit(rule, bc, N):
        if not quiet:
            print(f"skip  W{rule} N{N} {bc} (cached)")
        return results_io.load_basin_results(rule, bc)[N]
    t = rules.wolfram_to_tuple(rule)
    t0 = time.time()
    res = scc.analyze(rule, N, bc, t, detect_ergodic=False)
    dt = time.time() - t0
    if res.ergodic:
        return None
    total = 1 << N
    basins = list(res.sizes_basins)
    shared = res.shared_basin_size or 0
    ok = (sum(basins) + shared) == total
    old = results_io.load_results(rule, bc).get(N, {})
    agrees = {
        k: (old.get(k) == v)
        for k, v in (("n_scc", res.n_scc), ("n_recurrent", res.n_recurrent),
                     ("shared_basin_size", shared),
                     ("transient_depth", res.transient_depth),
                     ("n_transient_scc", res.n_transient_scc))
        if old.get(k) is not None
    }
    sizes_rec = list(res.sizes_recurrent)
    rec = {
        "rule": rule, "bc": bc, "N": N,
        "n_scc": res.n_scc, "n_recurrent": res.n_recurrent,
        "sizes_basins": basins[:_CAP],
        "basin_hist": {str(s): c
                       for s, c in sorted(Counter(basins).items(),
                                          reverse=True)},
        "basin_truncated": len(basins) > _CAP,
        "shared_basin_size": shared,
        "sizes_recurrent_hist": {str(s): c
                                 for s, c in sorted(Counter(sizes_rec).items(),
                                                    reverse=True)},
        "transient_depth": res.transient_depth,
        "n_transient_scc": res.n_transient_scc,
        "basin_sum_check": bool(ok),
        "transient_fraction": 1.0 - sum(sizes_rec) / total,
        "agrees_with_tier1a": agrees,
        "runtime": dt,
        "basins_version": results_io.BASIN_VERSION,
    }
    results_io.append_basin_result(rec)
    results_io.append_manifest(rule, bc, N, dt, False,
                               extra={"tier": "1e-basins",
                                      "basin_sum_check": bool(ok)})
    if not quiet:
        bad = [k for k, v in agrees.items() if not v]
        print(f"done  W{rule} N{N} {bc}: basins={len(basins)} shared={shared} "
              f"sum_ok={ok} agrees={'yes' if not bad else 'NO:' + ','.join(bad)}"
              f" [{dt:.1f}s]")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recompute basins where the Tier-1a archive is truncated")
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="obc0")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--all-units", action="store_true",
                    help="redo every non-ergodic unit, not just the gaps")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]
    rr = ([int(x) for x in args.rules.split(",")] if args.rules
          else list(range(256)))
    t0 = time.time()
    n_ok = n_bad = 0
    for bc in bcs:
        if args.all_units:
            units = [(r, N) for r in rr
                     for N, rec in sorted(results_io.load_results(r, bc).items())
                     if not rec.get("ergodic_flag")
                     and rec.get("sizes_basins") is not None]
        else:
            units = gaps(bc, rr)
        if args.n_max is not None:
            units = [(r, N) for r, N in units if N <= args.n_max]
        print(f"{bc}: {len(units)} units to recompute")
        for i, (rule, N) in enumerate(units, 1):
            if not args.quiet:
                print(f"[{i}/{len(units)}]", end=" ", flush=True)
            rec = recompute_unit(rule, N, bc, force=args.force,
                                 quiet=args.quiet)
            if rec is not None:
                n_ok += bool(rec["basin_sum_check"])
                n_bad += (not rec["basin_sum_check"])
    print(f"\nbasin sum rule: {n_ok} hold, {n_bad} FAIL "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
