"""
Orbit lengths at large N, without ever building the 2^N graph.

WHY THIS EXISTS.  R10 fits an exponential to the longest-cycle series of eight
rules over N = 6..24, where every point comes from a full O(2^N) walk.  Fitting
an exponential to numbers of order 400 is not a demonstration that the orbit is
exponentially long, so this module goes after the same claim from the other side:
it exhibits an actual orbit and measures its length.  A single measured orbit is
a LOWER bound on L_max, and a lower bound that grows exponentially is all the
claim needs -- and it is a proof rather than a fit, since the orbit is exhibited.

THE TRICK.  At obc0 the even sites are an independent set and so are the odd
ones, so the brick-wall sweep -- even sites ascending, then odd, each reading the
current state -- is exactly two SIMULTANEOUS sublattice updates: an even site
reads only odd bits, which the even half-sweep does not touch, and the odd sites
then read the new even bits.  A simultaneous update of a whole sublattice is a
handful of bitwise operations on the state word:

    L = (x << 1) & MASK        bit i of L is x_{i-1}
    R =  x >> 1                bit i of R is x_{i+1}
    P_lr = (L or ~L) & (R or ~R)          the four neighbour patterns
    A_s  = OR of the P_lr whose symbol is s, restricted to the sublattice
    x   -> ((x ^ A_V) & ~A_D) | A_E

so one sweep is about twenty integer operations regardless of N, and the memory
is one machine word.  Following an orbit for L steps costs O(L) time and O(1)
space, against O(2^N) for the full walk.  That is what makes N = 32 reachable for
rules whose orbit is a finite fraction of the space, and N = 48 for rules whose
orbits are short but exponentially growing.

This is emphatically NOT a replacement for the full walk: it sees one orbit at a
time and can never report n_wcc, the sector sizes or the transient structure.  It
answers exactly one question -- how long is the orbit through this state.

VALIDITY.  obc0 only.  At pbc with odd N the sublattices are not independent sets
and the identity above fails; the tests pin the agreement with the compiled
scalar step (xca.x_step) for all 256 rules, and refuse pbc.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence

from .. import results_io
from ..core import rules as rules_mod
from . import analysis

ORBIT_PATH = os.path.join(analysis.ANALYTICS, "xgate_orbits_{bc}.json")

#: the eight rules R10 sec.3.1 puts above ab = 2 in the cycle map
EXPONENTIAL_CYCLE_RULES = (57, 99, 156, 198, 201, 108, 73, 109)


def make_bitstep(rule: int, N: int, bc: str = "obc0") -> Callable[[int], int]:
    """
    One brick-wall sweep as ~20 integer ops, for obc0.

    Returns a closure over precomputed masks.  Correct for all 256 rules: I and V
    act through A_V, D and E through A_D and A_E, and the three sets are disjoint
    because each site selects exactly one symbol.
    """
    if bc != "obc0":
        raise ValueError("the sublattice identity only holds at obc0")
    t = rules_mod.wolfram_to_tuple(rule)
    mask = (1 << N) - 1
    even = sum(1 << i for i in range(0, N, 2))
    odd = sum(1 << i for i in range(1, N, 2))
    vp = tuple(i for i, s in enumerate(t) if s == "V")
    dp = tuple(i for i, s in enumerate(t) if s == "D")
    ep = tuple(i for i, s in enumerate(t) if s == "E")

    def half(x: int, sub: int) -> int:
        left = (x << 1) & mask
        right = x >> 1
        nl = left ^ mask
        nr = right ^ mask
        p = (nl & nr, nl & right, left & nr, left & right)
        if vp:
            a = 0
            for k in vp:
                a |= p[k]
            x ^= a & sub
        if dp:
            a = 0
            for k in dp:
                a |= p[k]
            x &= ~(a & sub) & mask
        if ep:
            a = 0
            for k in ep:
                a |= p[k]
            x |= a & sub
        return x

    def step(x: int) -> int:
        return half(half(x, even), odd)

    return step


def orbit_length(step: Callable[[int], int], x0: int,
                 cap: Optional[int] = None) -> Optional[int]:
    """
    The return time of x0, or None if it exceeds `cap`.

    Only meaningful when x0 is recurrent.  For a bijection every state is, so
    this is the length of the orbit through x0; for an irreversible rule call
    `settle` first.
    """
    x = step(x0)
    n = 1
    while x != x0:
        if cap is not None and n >= cap:
            return None
        x = step(x)
        n += 1
    return n


def settle(step: Callable[[int], int], x0: int, burn: int) -> int:
    """Iterate `burn` steps to land on the attractor of x0."""
    x = x0
    for _ in range(burn):
        x = step(x)
    return x


def sample_orbits(rule: int, N: int, bc: str = "obc0", samples: int = 256,
                  cap: Optional[int] = None, burn: int = 0,
                  seed: int = 0, budget_s: Optional[float] = None) -> Dict:
    """
    Measure the orbit length through `samples` random states.

    Sampling a state uniformly samples an orbit with probability proportional to
    its LENGTH, so long orbits are over-represented and the sample maximum is a
    good lower bound on L_max -- for a rule whose largest orbit covers a finite
    fraction of the space, essentially the first draw finds it.

    `burn` settles an irreversible rule onto its attractor first; the returned
    length is then the attractor's period.  `cap` abandons an orbit that is
    longer than expected rather than hanging.
    """
    step = make_bitstep(rule, N, bc)
    rng = random.Random((seed << 8) ^ (rule << 4) ^ N)
    lengths: List[int] = []
    abandoned = 0
    t0 = time.time()
    for _ in range(samples):
        x = rng.getrandbits(N)
        if burn:
            x = settle(step, x, burn)
        L = orbit_length(step, x, cap)
        if L is None:
            abandoned += 1
        else:
            lengths.append(L)
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    lengths.sort()
    return {"rule": rule, "N": N, "bc": bc,
            "samples": len(lengths) + abandoned,
            "measured": len(lengths), "abandoned": abandoned,
            "max": lengths[-1] if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "min": lengths[0] if lengths else None,
            "distinct": sorted(set(lengths))[-12:],
            "runtime": time.time() - t0}


def burn_for(rule: int, N: int) -> int:
    """
    How long to iterate before an irreversible rule is on its attractor.

    R10 measures the transient DEPTH -- the longest path to a cycle -- and it is
    tens of steps for these rules, growing slowly.  Ten times a generous linear
    bound costs nothing next to the orbit itself.
    """
    t = rules_mod.wolfram_to_tuple(rule)
    return 0 if all(s in "IV" for s in t) else 40 * N + 4000


def run(rules_in: Sequence[int] = EXPONENTIAL_CYCLE_RULES,
        Ns: Sequence[int] = range(20, 33), bc: str = "obc0",
        samples: int = 256, budget_s: float = 900.0,
        seed: int = 0) -> List[Dict]:
    """Sweep the sampler over rules and sizes, printing as it goes."""
    out = []
    for rule in rules_in:
        b = burn_for(rule, 0)
        for N in Ns:
            r = sample_orbits(rule, N, bc, samples=samples,
                              burn=burn_for(rule, N), seed=seed,
                              budget_s=budget_s)
            r["burn"] = burn_for(rule, N)
            out.append(r)
            print(f"X{rule:<4} N={N:<3} max {r['max']}  median {r['median']}  "
                  f"({r['measured']} states, {r['runtime']:.1f}s)", flush=True)
    return out


def merge(new: List[Dict], bc: str = "obc0") -> Dict:
    """Append to the stored sampler results, keeping the best per (rule, N)."""
    old = load(bc) or {"bc": bc, "rows": []}
    best: Dict = {}
    for r in old["rows"] + new:
        k = (r["rule"], r["N"])
        if k not in best:
            best[k] = r
            continue
        cur = best[k]
        if (r["max"] or 0) > (cur["max"] or 0):
            best[k] = r
        elif (r["max"] or 0) == (cur["max"] or 0):
            # A re-run that TIES the stored maximum is the evidence that the
            # maximum is real, so it must not be discarded: keep the row that
            # cost more draws.  This is what confirmed that X57's dips at N=28
            # and N=30 are genuine structure and not a missed orbit.
            if (r.get("samples") or 0) > (cur.get("samples") or 0):
                best[k] = r
    out = {"bc": bc, "rows": [best[k] for k in sorted(best)]}
    os.makedirs(analysis.ANALYTICS, exist_ok=True)
    with open(ORBIT_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = ORBIT_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def series(rule: int, bc: str = "obc0", d: Optional[Dict] = None):
    """(N, sampled L_max) for one rule, from the store."""
    d = d or load(bc) or {"rows": []}
    rows = [r for r in d["rows"] if r["rule"] == rule and r["max"]]
    rows.sort(key=lambda r: r["N"])
    return [r["N"] for r in rows], [r["max"] for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description="X-gate orbit sampler")
    ap.add_argument("--bc", default="obc0", choices=["obc0"])
    ap.add_argument("--rules", default=",".join(str(r) for r in
                                                EXPONENTIAL_CYCLE_RULES))
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--budget", type=float, default=900.0,
                    help="seconds per (rule, N) before stopping early")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rs = [int(v) for v in a.rules.split(",") if v.strip()]
    rows = run(rs, range(a.n_min, a.n_max + 1), a.bc, a.samples, a.budget,
               a.seed)
    d = merge(rows, a.bc)
    print(f"\nstored {len(d['rows'])} (rule, N) rows")


if __name__ == "__main__":
    main()
