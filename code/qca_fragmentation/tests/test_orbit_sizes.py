"""Regression tests for R31: the sector-size distribution with V = X.

The chain of reductions, pinned end to end:

    orbits of the 2^N permutation
      == the lcm assembly over the label's free segments
      == R30's label decomposition with the scalar kernel replaced by a cycle type

and the two things that are NOT inherited from R30 -- the transitivity half of
its Theorem 1, and the wall-free label on the ring.
"""

import math

import pytest

from qca_fragmentation.permutation import xca
from qca_fragmentation.scaling import orbit_sizes as O
from qca_fragmentation.scaling import sector_sizes as H

BCS = ("obc0", "pbc")


# --- the gate change itself ---------------------------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
def test_the_one_cycle_map_is_a_permutation(bc, rule):
    """X has no diagonal, so there is no branching: sectors are ORBITS."""
    for N in (6, 9, 12):
        T = xca.step_table(rule, N, bc)
        assert len({int(v) for v in T}) == 1 << N, (rule, N, bc)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
def test_the_wall_set_is_still_conserved(bc, rule):
    """R30 Thm 1's constancy half survives, and for free: supp(U_X|x>) is inside
    supp(U_H|x>)."""
    a, b = O.WALL_WORD[rule]

    def lab(x, N):
        if bc == "obc0":
            y = [0] + [(x >> i) & 1 for i in range(N)] + [0]
            return frozenset(i for i in range(N + 1)
                             if y[i] == a and y[i + 1] == b)
        y = [(x >> i) & 1 for i in range(N)]
        return frozenset(i for i in range(N)
                         if y[i] == a and y[(i + 1) % N] == b)

    for N in (7, 10, 12):
        T = xca.step_table(rule, N, bc)
        for x in range(1 << N):
            assert lab(x, N) == lab(int(T[x]), N), (rule, N, bc, x)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
def test_the_label_is_conserved_but_no_longer_complete(bc, rule):
    """The transitivity half of R30 Thm 1 dies: X orbits strictly REFINE the
    Hadamard sectors (R11's refinement theorem, in the census)."""
    for N in (10, 12):
        n_orb = len(O.hist_bruteforce(rule, N, bc))
        n_orb = sum(O.hist_bruteforce(rule, N, bc).values())
        n_fib = sum(H.hist(rule, N, bc).values())
        assert n_orb > n_fib, (rule, N, bc, n_orb, n_fib)


def test_frozen_states_are_gate_independent():
    """A state with no active site is fixed by any gate, so the size-1 sectors
    of R30 are exactly the fixed points here."""
    for bc in BCS:
        for rule in O.RULES:
            for N in range(4, 13):
                T = xca.step_table(rule, N, bc)
                fx = sum(1 for x in range(1 << N) if int(T[x]) == x)
                assert fx == H.hist(rule, N, bc).get(1, 0), (rule, N, bc)


# --- the structure that survives ---------------------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
@pytest.mark.parametrize("N", [10, 12])
def test_the_map_factorises_over_the_free_segments(bc, rule, N):
    """The load-bearing measurement of R31: inside a label fibre the X map is a
    DIRECT PRODUCT over the free segments, and each factor depends only on the
    segment.  This is what lets the renewal apparatus survive."""
    r = O.check_factorisation(rule, N, bc)
    assert r["factorisation_failures"] == 0, (rule, N, bc, r)
    assert r["length_conflicts"] == 0, (rule, N, bc, r)


def test_the_cycle_type_refines_the_R30_kernel():
    """sum C(v) = v: forgetting the cycle structure recovers R30 exactly."""
    for rule in O.RULES:
        for g in range(0, 12):
            for kern in (H.gap_kernel, H.left_kernel, H.right_kernel):
                v = kern(rule, g)
                if v:
                    assert sum(O.segment_cycle_type(rule, v)) == v, (rule, g, v)


def test_the_parabolic_segment_is_a_single_cycle():
    """The domain wall marches ballistically and reflects, so it visits every
    admissible position once per period."""
    for rule in (156, 198):
        for v in (1, 2, 3, 5, 8, 13, 21):
            assert O.segment_cycle_type(rule, v) == (v,), (rule, v)


def test_the_hyperbolic_segment_spectrum():
    """The hard-core X automaton on a segment with frozen ends."""
    want = {2: (2,), 3: (3,), 5: (2, 3), 8: (3, 5), 13: (2, 3, 8),
            21: (3, 7, 11), 34: (2, 3, 5, 10, 14), 55: (3, 9, 13, 13, 17)}
    for rule in (108, 201):
        for v, ct in want.items():
            assert O.segment_cycle_type(rule, v) == ct, (rule, v)
    # the longest cycle is 3l - 7 in this module's indexing (Tier-2 R-T21's
    # 3l - 13, verified there to l = 26, under the two-site shift)
    for l in range(4, 13):
        assert max(O._hyperbolic_type(l)) == 3 * l - 7, l


# --- the assembly -------------------------------------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
def test_the_lcm_assembly_reproduces_the_exhaustive_orbits(bc, rule):
    """The whole point: choose one cycle per segment, get prod c / lcm orbits of
    length lcm.  Checked term by term against the 2^N permutation."""
    for N in range(4, 14):
        assert O.hist(rule, N, bc) == O.hist_bruteforce(rule, N, bc), \
            (rule, N, bc)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", O.RULES)
def test_the_sum_rule(bc, rule):
    for N in (6, 11, 15):
        h = O.hist(rule, N, bc)
        assert sum(k * v for k, v in h.items()) == 1 << N


# --- the wall-free label: a segment on the chain, not on the ring -------------

def test_the_wall_free_label_is_a_segment_on_the_chain():
    """The frozen 0-pad plays the part of a wall's collar, so a wall-free chain
    is one long segment.  (Tier-2 R-T21 caught this; an earlier draft here said
    it was never a segment, which is false at obc0.)"""
    for rule in O.RULES:
        for N in range(4, 13):
            wf = O.wallfree_cycle_type(rule, N, "obc0")
            assert wf == O.segment_cycle_type(rule, sum(wf)), (rule, N)


def test_the_wall_free_label_is_not_a_segment_on_the_ring():
    """No boundary at all, so it is a different automaton: two fixed points for
    the ascent rules, and the hard-core X automaton on a ring for the others."""
    for rule in (156, 198):
        for N in range(4, 13):
            assert O.wallfree_cycle_type(rule, N, "pbc") == (1, 1), (rule, N)
    for rule in (108, 201):
        for N in range(4, 11):
            wf = O.wallfree_cycle_type(rule, N, "pbc")
            # its size is LUCAS, and R30 already showed Lucas numbers are not
            # generally Fibonacci -- so usually no segment has that many states
            # at all, which is even stronger than the two types differing
            assert sum(wf) == H.lucas(N), (rule, N, wf)
            try:
                same = O.segment_cycle_type(rule, sum(wf))
            except ValueError:
                continue                      # no segment of that size exists
            assert wf != same, (rule, N, wf)


def test_the_ring_defect_is_no_longer_a_defect():
    """Under the Hadamard 0^N and 1^N shared a label and had to be split by
    hand.  Under X they are simply two fixed points and the assembly gets them
    right with no special case."""
    for rule in (156, 198):
        for N in (6, 9, 12):
            assert O.hist(rule, N, "pbc")[1] >= 2


# --- D_max: a Landau-type maximum, hence subexponential ----------------------

def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _landau(budget, exact=False):
    frontier = {1: 0}
    for g in range(1, budget + 1):
        nxt = dict(frontier)
        for L, c in frontier.items():
            nc = c + g + 1
            if nc <= budget:
                nl = _lcm(L, g)
                if nl not in nxt or nxt[nl] > nc:
                    nxt[nl] = nc
        frontier = nxt
    if exact:
        return max((L for L, c in frontier.items() if c == budget), default=1)
    return max(frontier)


def test_parabolic_d_max_is_a_landau_maximum():
    """max lcm of the gap kernels at one letter of cost per gap -- with the
    budget spent EXACTLY on the ring, the cyclic analogue of R30's parity rule."""
    for rule in (156, 198):
        for N in range(4, 19):
            assert max(O.hist(rule, N, "obc0")) == _landau(N + 1), (rule, N)
            assert max(O.hist(rule, N, "pbc")) == _landau(N, exact=True), \
                (rule, N)


def test_d_max_collapses_relative_to_the_hadamard():
    """Under X the largest sector is subexponential, exp(Theta(sqrt(N log N))).

    The collapse is dramatic for the SYMMETRIC pair, whose Hadamard D_max is
    phi^N -- more than an order of magnitude already at N = 18.  For the ascent
    pair the Hadamard D_max is only 4^{N/5}, itself slow, so at these sizes the
    two are merely comparable and the separation appears later; asserting a
    factor there would be asserting something not yet visible."""
    for bc in BCS:
        for rule in O.RULES:
            for N in (14, 16, 18):
                assert max(O.hist(rule, N, bc)) <= max(H.hist(rule, N, bc)), \
                    (rule, N, bc)
    for bc in BCS:
        for rule in (108, 201):
            assert max(O.hist(rule, 18, bc)) < max(H.hist(rule, 18, bc)) / 10


def test_d_max_is_not_monotone_in_N():
    """The lcm arithmetic showing: a hallmark that this is not a growth law."""
    seq = [max(O.hist(201, N, "pbc")) for N in range(14, 19)]
    assert any(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), seq
