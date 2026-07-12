"""
Rule encoding: Wolfram number  <->  local-channel tuple  (r00, r01, r10, r11).

Context Tier 1 sec.1.2.  A classical ECA rule R in 0..255 has bits c_k with
R = sum_k c_k 2^k, k = 4*x_{i-1} + 2*x_i + x_{i+1}.  For each neighbour
configuration (m, n) = (x_{i-1}, x_{i+1}):

    c_p = c_{4m+n}      (output when centre = 0)
    c_q = c_{4m+n+2}    (output when centre = 1)

    (c_p, c_q) = (0, 1) -> 'I'   identity
    (c_p, c_q) = (1, 0) -> 'V'   unitary gate on centre (V = Hadamard H)
    (c_p, c_q) = (0, 0) -> 'D'   reset centre to |0>  (decay channel)
    (c_p, c_q) = (1, 1) -> 'E'   reset centre to |1>  (excitation channel)

So each rule is a tuple (r00, r01, r10, r11) in {I,V,D,E}^4; the map is a
bijection (4^4 = 256).

HSF cross-check numbering (../../HSF): the Julia oracle labels the 16 purely
unitary rules 0..15 by a 4-bit number (b3 b2 b1 b0) = (r11 r10 r01 r00) with
1 = V (Hadamard), 0 = I, obc0 boundary.  See `hsf_to_wolfram` below.
"""

from __future__ import annotations

from typing import Dict, Tuple

Tuple4 = Tuple[str, str, str, str]

# Order of the tuple slots is (r00, r01, r10, r11); the neighbour pair (m, n) is
# (left = x_{i-1}, right = x_{i+1}).
_MN_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))

# (c_p, c_q) -> channel symbol.
_CPCQ_TO_SYM = {(0, 1): "I", (1, 0): "V", (0, 0): "D", (1, 1): "E"}
_SYM_TO_CPCQ = {v: k for k, v in _CPCQ_TO_SYM.items()}


def wolfram_to_tuple(rule: int) -> Tuple4:
    """Wolfram number 0..255 -> (r00, r01, r10, r11) in {I,V,D,E}."""
    if not 0 <= rule <= 255:
        raise ValueError(f"Wolfram rule must be in 0..255, got {rule}")
    out = []
    for (m, n) in _MN_ORDER:
        cp = (rule >> (4 * m + n)) & 1
        cq = (rule >> (4 * m + n + 2)) & 1
        out.append(_CPCQ_TO_SYM[(cp, cq)])
    return tuple(out)  # type: ignore[return-value]


def tuple_to_wolfram(t: Tuple4) -> int:
    """(r00, r01, r10, r11) -> Wolfram number 0..255 (inverse of the above)."""
    rule = 0
    for sym, (m, n) in zip(t, _MN_ORDER):
        cp, cq = _SYM_TO_CPCQ[sym]
        rule |= cp << (4 * m + n)
        rule |= cq << (4 * m + n + 2)
    return rule


def is_unitary(t: Tuple4) -> bool:
    """A rule is unitary iff every local channel is I or V (no D/E)."""
    return all(s in ("I", "V") for s in t)


def has_V(t: Tuple4) -> bool:
    return "V" in t


def reflect_tuple(t: Tuple4) -> Tuple4:
    """
    Left-right reflection swaps the neighbour roles (m, n) -> (n, m), i.e. it
    swaps r01 and r10.

    Sector-size invariance under this reflection holds for UNITARY rules at both
    boundary conventions and all N (verified empirically and cross-checked
    against HSF: rules 156 and 198 share sector data).  It does NOT hold for
    dissipative rules under the even-first brickwork convention, because
    reflection swaps the even/odd layers whose order matters for the directed
    transition graph.  Hence the reflection reduction is applied only to the
    unitary sweep; the dissipative sweep computes every rule.
    """
    r00, r01, r10, r11 = t
    return (r00, r10, r01, r11)


def reflect_wolfram(rule: int) -> int:
    return tuple_to_wolfram(reflect_tuple(wolfram_to_tuple(rule)))


def reflection_representative(rule: int) -> int:
    """Canonical representative of the reflection pair (the smaller Wolfram #)."""
    return min(rule, reflect_wolfram(rule))


def reflection_pairs(rules=range(256)):
    """Yield canonical representatives (deduplicated) over the given rules."""
    seen = set()
    reps = []
    for r in rules:
        rep = reflection_representative(r)
        if rep not in seen:
            seen.add(rep)
            reps.append(rep)
    return reps


def spinflip_wolfram(rule: int) -> int:
    """
    Global spin-flip 0<->1 conjugation.  VALID ONLY for V-free rules (Hadamard
    breaks it: X H X != +-H).  Maps c_k -> 1 - c_{7-k}; equivalently swaps the
    neighbour bits and centre bit.  Provided for symmetry bookkeeping; callers
    must guard with `has_V`.
    """
    t = wolfram_to_tuple(rule)
    if has_V(t):
        raise ValueError("spin-flip conjugation invalid for rules containing V")
    # Under 0<->1 on all sites: (m,n,centre) -> (1-m,1-n,1-centre).  Rebuild.
    new = 0
    for k in range(8):
        # k = 4*xm + 2*xc + xn
        xm = (k >> 2) & 1
        xc = (k >> 1) & 1
        xn = k & 1
        out = (rule >> k) & 1
        kk = 4 * (1 - xm) + 2 * (1 - xc) + (1 - xn)
        new |= (1 - out) << kk
    return new


# --- HSF oracle numbering (unitary rules only, V = Hadamard, obc0) ------------
#
# HSF binary_coefficients(rule_num, mode=16) = last 4 bits of rule_num as
# (c1 c2 c3 c4) MSB->LSB, and local_gate applies V^c1 for neighbours (1,1),
# V^c2 for (1,0), V^c3 for (0,1), V^c4 for (0,0).  So bit3=r11, bit2=r10,
# bit1=r01, bit0=r00 with 1 = V, 0 = I.

def hsf_to_tuple(hsf: int) -> Tuple4:
    """HSF rule 0..15 -> (r00, r01, r10, r11) in {I, V}."""
    if not 0 <= hsf <= 15:
        raise ValueError("HSF unitary rule must be in 0..15")
    b = lambda k: "V" if (hsf >> k) & 1 else "I"
    return (b(0), b(1), b(2), b(3))  # r00=bit0, r01=bit1, r10=bit2, r11=bit3


def hsf_to_wolfram(hsf: int) -> int:
    return tuple_to_wolfram(hsf_to_tuple(hsf))


def wolfram_to_hsf(rule: int):
    """Wolfram number -> HSF 0..15 if the rule is unitary, else None."""
    t = wolfram_to_tuple(rule)
    if not is_unitary(t):
        return None
    r00, r01, r10, r11 = t
    bit = lambda s: 1 if s == "V" else 0
    return bit(r00) | (bit(r01) << 1) | (bit(r10) << 2) | (bit(r11) << 3)


# Precomputed HSF<->Wolfram table for the 16 unitary rules.
HSF_TO_WOLFRAM: Dict[int, int] = {h: hsf_to_wolfram(h) for h in range(16)}
WOLFRAM_TO_HSF: Dict[int, int] = {v: k for k, v in HSF_TO_WOLFRAM.items()}

# The 16 unitary Wolfram rule numbers.
UNITARY_RULES = sorted(HSF_TO_WOLFRAM.values())


def unitary_reflection_reps():
    """Canonical reflection representatives among the 16 unitary rules (valid
    reduction; see reflect_tuple)."""
    return reflection_pairs(UNITARY_RULES)


def channel_kraus_symbols(t: Tuple4):
    """
    Whether the rule needs a second Kraus branch (has any D or E).  Unitary
    rules -> False (single Kraus operator).
    """
    return any(s in ("D", "E") for s in t)
