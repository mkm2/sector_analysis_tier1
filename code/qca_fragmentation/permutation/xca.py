"""
The 256 QCA with V = X instead of V = Hadamard: permutation circuits.

WHAT CHANGES.  The symbol table is unchanged -- each entry says what happens at a
site for one neighbour pattern -- but the active gate is now the bit flip:

    I  identity            x_i -> x_i
    V  X gate              x_i -> 1 - x_i        (was the Hadamard)
    D  reset to |0>        x_i -> 0
    E  reset to |1>        x_i -> 1

Every one of those four is a FUNCTION on the computational basis, so the whole
circuit is, and the consequences are large:

  * succ(x) is a single state.  There is no superposition, so no interference,
    so nothing for the exact-amplitude machinery of core/cycle.py to do.  The
    transition graph has out-degree exactly 1 -- a FUNCTIONAL GRAPH.
  * In a functional graph every weakly connected component contains exactly one
    cycle.  So the number of terminal SCCs equals the number of WCCs, the basin
    of a cycle is its whole component, and no state is shared between basins.
    That is a theorem here, and it is exactly the identity that fails so badly
    in the Hadamard case (R9: rule 22 has three attractors inside two sectors).
  * The 16 rules built from I and V alone are BIJECTIONS -- reversible cellular
    automata -- so every state is recurrent, each component is a single cycle,
    and the sector decomposition is the orbit decomposition of a permutation.
    D and E are the only source of irreversibility.

WHY PORT RATHER THAN PARAMETRISE.  The site order and the neighbour conventions
must match the validated engine exactly -- in particular the brick-wall order is
even sites ascending then odd sites ascending, with each site reading the CURRENT
state, which is NOT the same as two simultaneous sublattice updates once the
even sublattice is not an independent set (pbc at odd N).  So rather than
reimplement that, we take the compiled step list straight from
core.cycle._compile and only swap the local action.  Nothing in the Hadamard
engine is touched.
"""

from __future__ import annotations

from array import array
from typing import Callable, Dict, List, Optional, Tuple

from ..core import rules as rules_mod
from ..core.cycle import _compile
from ..core.rules import Tuple4


def x_step(x: int, N: int, t: Tuple4, bc: str) -> int:
    """One brick-wall cycle of the X-gate rule, as a function on bitmasks."""
    for (i, lpos, rpos, tt) in _compile(N, t, bc):
        left = (x >> lpos) & 1 if lpos >= 0 else 0
        right = (x >> rpos) & 1 if rpos >= 0 else 0
        s = tt[2 * left + right]
        if s == "I":
            continue
        bit = 1 << i
        if s == "V":
            x ^= bit
        elif s == "D":
            x &= ~bit
        else:                      # "E"
            x |= bit
    return x


def make_step(rule: int, N: int, bc: str,
              t: Optional[Tuple4] = None) -> Callable[[int], int]:
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)
    return lambda x: x_step(x, N, t, bc)


def step_table(rule: int, N: int, bc: str, t: Optional[Tuple4] = None):
    """
    The full map as an array f[x] = successor, x = 0..2^N-1.

    Kept as array('i') rather than a list: at N = 22 a Python list of 4.2M ints
    costs a few hundred MB, and the sweep wants headroom.  The compiled step
    list is hoisted out of the loop.
    """
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)
    steps = _compile(N, t, bc)
    out = array("i", bytes(4 * (1 << N)))
    for x in range(1 << N):
        y = x
        for (i, lpos, rpos, tt) in steps:
            left = (y >> lpos) & 1 if lpos >= 0 else 0
            right = (y >> rpos) & 1 if rpos >= 0 else 0
            s = tt[2 * left + right]
            if s == "I":
                continue
            bit = 1 << i
            if s == "V":
                y ^= bit
            elif s == "D":
                y &= ~bit
            else:
                y |= bit
        out[x] = y
    return out


def succ_fn(rule: int, N: int, bc: str, t: Optional[Tuple4] = None):
    """Adapter for graph.wcc.union_find_components, which wants a set-valued
    successor."""
    step = make_step(rule, N, bc, t)
    return lambda x: (step(x),)


def is_reversible(rule: int, N: int, bc: str) -> bool:
    """True iff the map is a bijection.  Must hold exactly for the I/V rules."""
    f = step_table(rule, N, bc)
    return len(set(f)) == len(f)

def step_table_np(rule: int, N: int, bc: str, t: Optional[Tuple4] = None):
    """
    Vectorised form of step_table: the same brick-wall sweep, but carrying all
    2^N states through each site step at once.

    The site order still matters and is still taken from _compile, so this is a
    faithful vectorisation, not a simultaneous-update approximation: at each step
    the neighbour bits are read from the CURRENT y, exactly as in the scalar
    version.  Only the loop over x is lifted into numpy, which turns an
    O(N 2^N) Python loop into 4N array passes and is what lets the sweep reach
    N = 24 rather than N = 20.
    """
    import numpy as np
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)
    n = 1 << N
    y = np.arange(n, dtype=np.int64)
    for (i, lpos, rpos, tt) in _compile(N, t, bc):
        left = ((y >> lpos) & 1) if lpos >= 0 else np.zeros(1, dtype=np.int64)
        right = ((y >> rpos) & 1) if rpos >= 0 else np.zeros(1, dtype=np.int64)
        idx = (left << 1) | right          # 0..3, broadcast if a side is fixed
        bit = np.int64(1 << i)
        for k, sym in enumerate(tt):
            if sym == "I":
                continue
            m = (idx == k)
            if not m.any():
                continue
            if sym == "V":
                y[m] ^= bit
            elif sym == "D":
                y[m] &= ~bit
            else:
                y[m] |= bit
    return y

