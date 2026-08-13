"""Close the N gaps in the unitary census, cheaply and with a gate.

The two sweeps stop at different N for different rules, for reasons that have
nothing to do with the rules being interesting.  Tier-1a stops enlarging N as
soon as a rule is classified ergodic, so the seven ergodic unitary rules end at
N <= 13 there; the Tier-1e pass ran to N = 16 for all 256 rules; and the
fragmented unitary rules were pushed to 21, with W150 at 22.  R19's census
therefore had tracks of very unequal length.

They are cheap in the strongest sense.  `graph/flip_graph.flip_components_np`
decides the sector partition of a *unitary* rule by label propagation on the
single-flip graph -- a theorem, validated against the streamed engine for all
sixteen unitary rules at both boundary conditions in tests/test_flip_graph.py --
and one uint32 array of 2^N entries is the whole memory cost.  N = 22 is 17 MB
and a quarter of a second per rule.

This module recomputes the missing units and appends them to the Tier-1e store
in the ordinary schema.  It refuses to write anything for a rule until the same
code has reproduced *every* already-stored unit of that rule, in both stores, so
an extension can never quietly disagree with the sweep it extends.

    python -m qca_fragmentation.extend_unitary --n-max 22
    python -m qca_fragmentation.extend_unitary --rules 51,54 --dry-run

Only unitary rules: the flip reduction is proved for those and for no others.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence, Tuple

from . import results_io
from .core import rules as rules_mod
from .graph.flip_graph import flip_components_np
from .graph.wcc import WCCResult

#: every rule the reduction is valid for
UNITARY: Tuple[int, ...] = tuple(sorted(rules_mod.UNITARY_RULES))

#: the subset both sweeps stopped early on
ERGODIC_UNITARY: Tuple[int, ...] = (51, 54, 57, 99, 147, 153, 195)


def sizes(rule: int, N: int, bc: str) -> List[int]:
    t = rules_mod.wolfram_to_tuple(rule)
    if not rules_mod.is_unitary(t):
        raise ValueError(f"W{rule} is not unitary; the flip reduction is not "
                         f"proved for it")
    return flip_components_np(N, t, bc)


def agrees_with_store(rule: int, bc: str) -> Dict[Tuple[str, int], bool]:
    """Recompute every stored unit of this rule, in BOTH stores, and compare.

    The gate on writing.  A mismatch means the fast path and the sweep disagree
    somewhere, and then the extension is worthless whatever it says.  Tier-1a
    records are included because for a unitary rule the strongly and weakly
    connected components coincide, so they are a check on the same numbers by a
    completely different algorithm (Tarjan, streamed).
    """
    out: Dict[Tuple[str, int], bool] = {}
    for N, rec in sorted(results_io.load_wcc_results(rule, bc).items()):
        if rec.get("aborted") or rec.get("n_wcc") is None:
            continue
        s = sizes(rule, N, bc)
        out[("1e", N)] = (len(s) == rec["n_wcc"] and s[0] == rec["d_max_wcc"]
                          and sum(s) == (1 << N))
    for N, rec in sorted(results_io.load_results(rule, bc).items()):
        if rec.get("n_recurrent") is None:
            continue
        stored = results_io.sizes_from_record(rec, "sizes_recurrent")
        if not stored:
            continue
        s = sizes(rule, N, bc)
        out[("1a", N)] = (len(s) == rec["n_recurrent"] and s[0] == stored[0]
                          and sum(s) == (1 << N))
    return out


def compute_unit(rule: int, N: int, bc: str, *, f_erg: float = 0.5) -> dict:
    t0 = time.time()
    s = sizes(rule, N, bc)
    dt = time.time() - t0
    assert sum(s) == (1 << N), (rule, N, bc, "A1 sum rule")
    res = WCCResult(N=N, rule=rule, bc=bc,
                    ergodic=bool(s[0] > f_erg * (1 << N)),
                    n_wcc=len(s), sizes_wcc=s, d_max_wcc=s[0],
                    n_frozen=sum(1 for x in s if x == 1),
                    ergodic_bound=s[0], f_erg=f_erg, runtime=dt)
    return results_io.record_from_wcc_result(
        res, checks={"source": "extend_unitary", "flip_reduction": True})


def extend(rules_to_do: Sequence[int] = UNITARY, bc: str = "obc0",
           n_max: int = 22, *, dry_run: bool = False, quiet: bool = False):
    written = []
    for rule in rules_to_do:
        checks = agrees_with_store(rule, bc)
        if not all(checks.values()):
            bad = [k for k, ok in checks.items() if not ok]
            raise AssertionError(
                f"W{rule} {bc}: the flip reduction disagrees with the stored "
                f"sweep at {bad}; refusing to extend")
        have = set(results_io.load_wcc_results(rule, bc))
        todo = [N for N in range(min(have or {6}), n_max + 1) if N not in have]
        if not quiet:
            print(f"W{rule:<4d} {bc}: {len(checks)} stored units reproduced, "
                  f"{len(todo)} to add {todo}")
        for N in todo:
            rec = compute_unit(rule, N, bc)
            if not dry_run:
                results_io.append_wcc_result(rec)
            written.append((rule, N))
            if not quiet:
                print(f"      N={N:<3d} n_wcc={rec['n_wcc']:<9d} "
                      f"d_max={rec['d_max_wcc']:<10d} "
                      f"frozen={rec['n_frozen']:<9d} "
                      f"[{rec['runtime']:.2f}s]"
                      + ("  (dry run)" if dry_run else ""))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rules", default=",".join(str(r) for r in UNITARY))
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--n-max", type=int, default=22)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    w = extend([int(x) for x in a.rules.split(",") if x], a.bc, a.n_max,
               dry_run=a.dry_run)
    print(f"\n{len(w)} unit(s) {'would be ' if a.dry_run else ''}added")


if __name__ == "__main__":
    main()
