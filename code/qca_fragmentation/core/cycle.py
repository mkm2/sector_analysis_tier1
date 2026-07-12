"""
One QCA cycle: brickwork branch propagation and succ(x).

Context Tier 1 sec.1.3 and sec.3.  One timestep applies the local update at all
EVEN sites (0, 2, 4, ...) first, then at all ODD sites (1, 3, 5, ...).  Controls
read the CURRENT bitmask (odd-layer controls see the post-even-layer state).

A branch is an exact-amplitude dict  {bitmask: (a, b)}  with one shared dyadic
exponent m (amplitude = (a + b*sqrt2) / 2^m).  Kraus branches are NEVER merged
(that would fabricate interference between incoherent alternatives).

succ(x) = { y : <y| Phi_C(|x><x|) |y> != 0 } = union of the supports of all
surviving Kraus branches after one cycle.

Performance: a rule is "compiled" once per (N, bc) into an ordered list of site
steps with precomputed neighbour bit-positions and a 4-entry symbol table.
Unitary rules (single Kraus branch, no D/E) take a specialised fast path.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Tuple

from . import ring
from .rules import Tuple4, channel_kraus_symbols

Branch = Tuple[Dict[int, Tuple[int, int]], int]  # (amps, m)


def even_sites(N: int) -> List[int]:
    return list(range(0, N, 2))


def odd_sites(N: int) -> List[int]:
    return list(range(1, N, 2))


def neighbor_bits(x: int, i: int, N: int, bc: str) -> Tuple[int, int]:
    """Return (m, n) = (left value x_{i-1}, right value x_{i+1}) under bc."""
    if bc == "pbc":
        left = (x >> ((i - 1) % N)) & 1
        right = (x >> ((i + 1) % N)) & 1
    elif bc == "obc0":
        left = (x >> (i - 1)) & 1 if i > 0 else 0
        right = (x >> (i + 1)) & 1 if i < N - 1 else 0
    else:
        raise ValueError(f"unknown boundary convention {bc!r}")
    return left, right


def trigger_symbol(x: int, i: int, N: int, t: Tuple4, bc: str) -> str:
    m, n = neighbor_bits(x, i, N, bc)
    return t[2 * m + n]


# --- compiled step list ------------------------------------------------------
# Each step is (i, lpos, rpos, symtab) where lpos/rpos are the bit positions of
# the left/right neighbour (or -1 meaning "fixed vacuum 0" at an obc0 edge), and
# symtab[2*left+right] is the local channel symbol at that site.

@lru_cache(maxsize=None)
def _compile(N: int, t: Tuple4, bc: str):
    steps = []
    for layer in (range(0, N, 2), range(1, N, 2)):
        for i in layer:
            if bc == "pbc":
                lpos = (i - 1) % N
                rpos = (i + 1) % N
            elif bc == "obc0":
                lpos = i - 1 if i > 0 else -1
                rpos = i + 1 if i < N - 1 else -1
            else:
                raise ValueError(f"unknown boundary convention {bc!r}")
            steps.append((i, lpos, rpos, t))
    return tuple(steps)


def _mn(x: int, lpos: int, rpos: int) -> int:
    left = (x >> lpos) & 1 if lpos >= 0 else 0
    right = (x >> rpos) & 1 if rpos >= 0 else 0
    return 2 * left + right


def _apply_site_unitary(amps: Dict[int, Tuple[int, int]], m: int,
                        i: int, lpos: int, rpos: int, t: Tuple4) -> Branch:
    """Single-branch fast path (rule has only I / V)."""
    bit = 1 << i
    # detect whether any Hadamard fires
    has_fire = False
    for x in amps:
        if t[_mn(x, lpos, rpos)] == "V":
            has_fire = True
            break

    out: Dict[int, Tuple[int, int]] = {}
    get = out.get
    if not has_fire:
        # every entry is I -> identity (m unchanged)
        return amps, m
    for x, (a, b) in amps.items():
        if t[_mn(x, lpos, rpos)] == "V":
            num0 = 2 * b
            if x & bit:  # |1> -> (|0> - |1>)/sqrt2 : + on |0>, - on |1>
                k0 = x & ~bit
                p = get(k0)
                out[k0] = (num0, a) if p is None else (p[0] + num0, p[1] + a)
                p = get(x)
                out[x] = (-num0, -a) if p is None else (p[0] - num0, p[1] - a)
            else:        # |0> -> (|0> + |1>)/sqrt2 : + on both
                p = get(x)
                out[x] = (num0, a) if p is None else (p[0] + num0, p[1] + a)
                k1 = x | bit
                p = get(k1)
                out[k1] = (num0, a) if p is None else (p[0] + num0, p[1] + a)
        else:  # I : value unchanged, but m increments so double the numerator
            p = get(x)
            da, db = 2 * a, 2 * b
            out[x] = (da, db) if p is None else (p[0] + da, p[1] + db)
    # prune exact zeros produced by interference
    out = {k: v for k, v in out.items() if v[0] or v[1]}
    return out, m + 1


def _split_site(branch: Branch, i: int, lpos: int, rpos: int, t: Tuple4):
    """General path (I/V/D/E). Returns (a0_branch_or_None, a1_branch_or_None):
    the no-jump (A0) sub-branch and the jump (A1) sub-branch at site i."""
    amps, m = branch
    bit = 1 << i

    syms = {}
    has_fire = False
    has_jump = False
    for x in amps:
        s = t[_mn(x, lpos, rpos)]
        syms[x] = s
        if s == "V":
            has_fire = True
        elif s == "D" and (x & bit):
            has_jump = True
        elif s == "E" and not (x & bit):
            has_jump = True

    a0: Dict[int, Tuple[int, int]] = {}
    g0 = a0.get

    def acc0(key, va, vb):
        p = g0(key)
        a0[key] = (va, vb) if p is None else (p[0] + va, p[1] + vb)

    for x, (a, b) in amps.items():
        s = syms[x]
        if s == "I":
            if has_fire:
                acc0(x, 2 * a, 2 * b)
            else:
                acc0(x, a, b)
        elif s == "D":
            if not (x & bit):
                if has_fire:
                    acc0(x, 2 * a, 2 * b)
                else:
                    acc0(x, a, b)
        elif s == "E":
            if x & bit:
                if has_fire:
                    acc0(x, 2 * a, 2 * b)
                else:
                    acc0(x, a, b)
        else:  # V
            num0 = 2 * b
            if x & bit:
                acc0(x & ~bit, num0, a)
                acc0(x, -num0, -a)
            else:
                acc0(x, num0, a)
                acc0(x | bit, num0, a)
    a0 = {k: v for k, v in a0.items() if v[0] or v[1]}
    m0 = m + 1 if has_fire else m
    a0_branch = (a0, m0) if a0 else None

    a1_branch = None
    if has_jump:
        a1: Dict[int, Tuple[int, int]] = {}
        g1 = a1.get
        for x, (a, b) in amps.items():
            s = syms[x]
            if s == "D" and (x & bit):
                k = x & ~bit
                p = g1(k)
                a1[k] = (a, b) if p is None else (p[0] + a, p[1] + b)
            elif s == "E" and not (x & bit):
                k = x | bit
                p = g1(k)
                a1[k] = (a, b) if p is None else (p[0] + a, p[1] + b)
        a1 = {k: v for k, v in a1.items() if v[0] or v[1]}
        if a1:
            a1_branch = (a1, m)

    return a0_branch, a1_branch


def _apply_site_general(branch: Branch, i: int, lpos: int, rpos: int,
                        t: Tuple4) -> List[Branch]:
    """General path returning the surviving sub-branches as a list."""
    a0, a1 = _split_site(branch, i, lpos, rpos, t)
    out = []
    if a0 is not None:
        out.append(a0)
    if a1 is not None:
        out.append(a1)
    return out


def apply_site(branch: Branch, i: int, N: int, t: Tuple4, bc: str) -> List[Branch]:
    """Apply the local channel at site i to one branch (public, general)."""
    if bc == "pbc":
        lpos, rpos = (i - 1) % N, (i + 1) % N
    else:
        lpos = i - 1 if i > 0 else -1
        rpos = i + 1 if i < N - 1 else -1
    return _apply_site_general(branch, i, lpos, rpos, t)


def one_cycle_branches(x: int, N: int, t: Tuple4, bc: str) -> List[Branch]:
    """Propagate |x> through one brickwork cycle; return surviving branches."""
    steps = _compile(N, t, bc)
    unitary = channel_kraus_symbols(t) is False
    branches: List[Branch] = [({x: (1, 0)}, 0)]
    if unitary:
        amps, m = branches[0]
        for (i, lpos, rpos, tt) in steps:
            amps, m = _apply_site_unitary(amps, m, i, lpos, rpos, tt)
        return [(amps, m)]
    for (i, lpos, rpos, tt) in steps:
        nxt: List[Branch] = []
        for br in branches:
            nxt.extend(_apply_site_general(br, i, lpos, rpos, tt))
        branches = nxt
    return branches


def one_cycle_branches_labeled(x: int, N: int, t: Tuple4, bc: str):
    """
    Like one_cycle_branches, but tags each surviving branch with the set of
    sites at which the A1 (jump) Kraus operator was applied.  That set uniquely
    labels the GLOBAL Kraus operator K_b (product of per-site A_{b_i}(i) in the
    fixed even-then-odd processing order), so branches of different input states
    can be paired by label to build the exact channel superoperator (Tier 1b).

    Returns list of (amps, m, frozenset_of_jump_sites).
    """
    steps = _compile(N, t, bc)
    branches = [({x: (1, 0)}, 0, frozenset())]
    for (i, lpos, rpos, tt) in steps:
        nxt = []
        for (amps, m, lab) in branches:
            a0, a1 = _split_site((amps, m), i, lpos, rpos, tt)
            if a0 is not None:
                nxt.append((a0[0], a0[1], lab))
            if a1 is not None:
                nxt.append((a1[0], a1[1], lab | {i}))
        branches = nxt
    return branches


def succ(x: int, N: int, t: Tuple4, bc: str) -> List[int]:
    """
    succ(x): sorted list of basis states y with <y|Phi_C(|x><x|)|y> != 0.
    Union of supports over all surviving Kraus branches (exact; no tolerance).
    """
    steps = _compile(N, t, bc)
    if channel_kraus_symbols(t) is False:
        amps = {x: (1, 0)}
        m = 0
        for (i, lpos, rpos, tt) in steps:
            amps, m = _apply_site_unitary(amps, m, i, lpos, rpos, tt)
        return sorted(amps.keys())
    out = set()
    for amps, _m in one_cycle_branches(x, N, t, bc):
        out.update(amps.keys())
    return sorted(out)


# --- validation helpers ------------------------------------------------------

def branch_norms(x: int, N: int, t: Tuple4, bc: str) -> Fraction:
    """
    Total probability = sum over branches of branch squared-norm.  Must equal 1
    exactly (Fraction).  For unitary rules there is a single branch.
    """
    total = Fraction(0)
    for amps, m in one_cycle_branches(x, N, t, bc):
        r, s = ring.branch_norm(amps, m)
        assert s == 0, "branch norm has non-vanishing sqrt2 part (bug)"
        total += r
    return total


def is_single_branch_unitary(x: int, N: int, t: Tuple4, bc: str) -> bool:
    """Unitary rules must yield exactly one branch with unit norm."""
    brs = one_cycle_branches(x, N, t, bc)
    if len(brs) != 1:
        return False
    r, s = ring.branch_norm(*brs[0])
    return r == 1 and s == 0
