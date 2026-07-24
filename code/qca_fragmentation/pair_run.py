"""
Tier 1d single-unit runner: pair-graph certificate + weak charges for one
(rule, N, bc), written to the separate results_tier1d store (Tier 1a untouched).

    python -m qca_fragmentation.pair_run --rule 22 --N 4-10 --bc pbc

Idempotent: a unit already present for the current tier1d_version is skipped
unless --force.  cesaro_rank (the exact channel fixed-point dimension) is filled
only for N <= --cesaro-max-N, where the dense 4^N superoperator is affordable.
"""

from __future__ import annotations

import argparse
import time

from . import results_io
from .core import rules
from .graph.pair_graph import analyze_pair, wcc_labels
from .graph.scc import recurrent_classes
from .quantum.weak_charges import analyze_charges


def within_sector_cesaro(N: int, t, bc: str) -> int:
    """Exact WITHIN-sector fixed-point dimension: sum over WCCs of the geometric
    multiplicity of eigenvalue 1 of the WCC-restricted superoperator.

    This is the scope the diagonal P_rec upper-bounds -- it EXCLUDES cross-sector
    coherences (e.g. the |S|^2 frozen-frozen block of a V-free rule, which the
    full cesaro_rank would count), so that K_M <= cesaro_within <= fix_upper.
    """
    from .quantum.peripheral import restricted_superoperator, geometric_multiplicity
    wcc = wcc_labels(N, t, bc)
    groups = {}
    for x in range(1 << N):
        groups.setdefault(wcc[x], []).append(x)
    total = 0
    for S in groups.values():
        if len(S) == 1:
            total += 1                     # a frozen sector: one fixed point
        else:
            Phi = restricted_superoperator(sorted(S), N, t, bc)
            total += geometric_multiplicity(Phi, 1.0)
    return total


def run_pair_unit(rule: int, N: int, bc: str, *, force: bool = False,
                  cesaro_max_N: int = 6, charges: bool = True, r: int = 3,
                  node_budget: int = 6_000_000, charge_offdiag_cap: int = 6000,
                  quiet: bool = False) -> dict:
    if not force and results_io.has_pair_unit(rule, bc, N):
        if not quiet:
            print(f"skip  W{rule} N{N} {bc} (cached)")
        return results_io.load_pair_results(rule, bc)[N]

    t = rules.wolfram_to_tuple(rule)
    t0 = time.time()
    unitary = rules.is_unitary(t)

    # K_M = number of terminal SCCs (Tier 1a lower bound).
    km = 0 if unitary else len(recurrent_classes(rule, N, bc, t))

    res = analyze_pair(rule, N, bc, t, cross_sector=False, node_budget=node_budget,
                       return_support=True)

    cz = None
    if not unitary and N <= cesaro_max_N:
        cz = within_sector_cesaro(N, t, bc)

    # Weak charges: skip near-ergodic units whose coherence support is huge (the
    # Fraction grading enumeration is O(|offdiag| * deg) and the grading is only
    # crisp for fragmented rules anyway); reuse the pair result to avoid a recompute.
    ch = None
    if (charges and not unitary and not res.bounded_only
            and res.pair_offdiag <= charge_offdiag_cap):
        ch = analyze_charges(rule, N, bc, t, r=r, node_budget=node_budget, pair=res)

    rec = {
        "rule": rule, "bc": bc, "N": N, "km": km,
        "n_recurrent_states": res.n_recurrent_states,
        "cesaro_rank": cz,
        "pair_rec_size": res.pair_rec_size,
        "pair_offdiag": res.pair_offdiag,
        "pair_diag_extra": res.pair_diag_extra,
        "fix_upper": res.fix_upper,
        "certified": res.certified,
        "bounded_only": res.bounded_only,
        "n_pair_nodes": res.n_pair_nodes,
        "n_strong": ch.n_strong if ch else None,
        "n_weak": ch.n_weak if ch else None,
        "weak_grades_coherence": ch.weak_grades_coherence if ch else None,
        "d_values_on_coherence": ch.d_values_on_coherence if ch else None,
        "runtime": round(time.time() - t0, 4),
        "tier1d_version": results_io.TIER1D_VERSION,
    }
    results_io.append_pair_result(rec)
    if not quiet:
        if res.bounded_only:
            print(f"done  W{rule} N{N} {bc}: BOUNDED (nodes>{res.n_pair_nodes}) "
                  f"[{rec['runtime']}s]")
        else:
            print(f"done  W{rule} N{N} {bc}: K_M={km} cesaro={cz} "
                  f"|P_rec|={res.pair_rec_size} offdiag={res.pair_offdiag} "
                  f"cert={res.certified} weak={rec['n_weak']} [{rec['runtime']}s]")
    return rec


def parse_N(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1d pair-graph single-rule run")
    ap.add_argument("--rule", type=int, required=True)
    ap.add_argument("--N", type=str, required=True)
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="pbc")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-charges", action="store_true")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--cesaro-max-N", type=int, default=6)
    ap.add_argument("--node-budget", type=int, default=6_000_000)
    args = ap.parse_args(argv)
    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]
    for bc in bcs:
        for N in parse_N(args.N):
            run_pair_unit(args.rule, N, bc, force=args.force,
                          cesaro_max_N=args.cesaro_max_N,
                          charges=not args.no_charges, r=args.r,
                          node_budget=args.node_budget)


if __name__ == "__main__":
    main()
