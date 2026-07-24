"""succ(x) and branch-semantics tests (context Tier 1 sec.3)."""
from fractions import Fraction

import pytest

from qca_fragmentation.core import cycle, rules

UNITARY = rules.UNITARY_RULES


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_unitary_single_branch_and_norm(bc, N):
    for r in UNITARY:
        t = rules.wolfram_to_tuple(r)
        for x in range(1 << N):
            brs = cycle.one_cycle_branches(x, N, t, bc)
            assert len(brs) == 1                      # single Kraus branch
            assert cycle.branch_norms(x, N, t, bc) == Fraction(1)


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("rule", [22, 0, 28, 29, 22 ^ 0])
def test_dissipative_norm_sums_to_one(bc, rule):
    t = rules.wolfram_to_tuple(rule)
    for N in (3, 4, 5):
        for x in range(1 << N):
            assert cycle.branch_norms(x, N, t, bc) == Fraction(1)


def test_dissipative_succ_matches_dense_channel_oracle():
    # Build per-site Kraus A0,A1 as dense 2^N matrices, compose the cycle
    # superoperator S = prod_sites (A0 (x) A0* + A1 (x) A1*), and check that the
    # support of M_{yx} = <yy|S|xx> equals the engine's succ(x).  Also verifies
    # per-site Kraus completeness A0^dag A0 + A1^dag A1 = I.
    np = pytest.importorskip("numpy")
    Hm = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])

    def site_kraus(N, t, bc, i):
        dim = 1 << N; bit = 1 << i
        A0 = np.zeros((dim, dim)); A1 = np.zeros((dim, dim)); useA1 = False
        for x in range(dim):
            m, n = cycle.neighbor_bits(x, i, N, bc); s = t[2 * m + n]
            xi = (x >> i) & 1; x0 = x & ~bit; x1 = x | bit
            if s == "I":
                A0[x, x] += 1
            elif s == "V":
                if xi == 0:
                    A0[x0, x] += Hm[0, 0]; A0[x1, x] += Hm[1, 0]
                else:
                    A0[x0, x] += Hm[0, 1]; A0[x1, x] += Hm[1, 1]
            elif s == "D":
                if xi == 0:
                    A0[x, x] += 1
                else:
                    A1[x0, x] += 1; useA1 = True
            elif s == "E":
                if xi == 1:
                    A0[x, x] += 1
                else:
                    A1[x1, x] += 1; useA1 = True
        return A0, (A1 if useA1 else None)

    def cycle_super(N, t, bc):
        dim = 1 << N; S = np.eye(dim * dim)
        for layer in (cycle.even_sites(N), cycle.odd_sites(N)):
            for i in layer:
                A0, A1 = site_kraus(N, t, bc, i)
                Si = np.kron(A0, A0.conj())
                if A1 is not None:
                    Si = Si + np.kron(A1, A1.conj())
                S = Si @ S
        return S

    for bc in ("obc0", "pbc"):
        for N in (3, 4):
            for r in (22, 0, 28, 29, 50, 200, 232, 90, 150):
                t = rules.wolfram_to_tuple(r)
                # per-site Kraus completeness
                for i in range(N):
                    A0, A1 = site_kraus(N, t, bc, i)
                    M = A0.T @ A0 + (A1.T @ A1 if A1 is not None else 0)
                    assert np.allclose(M, np.eye(1 << N), atol=1e-9)
                S = cycle_super(N, t, bc); dim = 1 << N
                for x in range(dim):
                    col = S[:, x * dim + x]
                    supp = {y for y in range(dim) if abs(col[y * dim + y]) > 1e-9}
                    assert supp == set(cycle.succ(x, N, t, bc)), (bc, N, r, x)


def test_identity_rule_204_is_identity():
    # rule 204 (I,I,I,I): succ(x) = {x} for all x
    t = rules.wolfram_to_tuple(204)
    for bc in ("obc0", "pbc"):
        for N in (4, 6):
            for x in range(1 << N):
                assert cycle.succ(x, N, t, bc) == [x]


def test_all_hadamard_51_full_support():
    # rule 51 (V,V,V,V): H^{tensor N}, succ(x) = all 2^N states
    t = rules.wolfram_to_tuple(51)
    for bc in ("obc0", "pbc"):
        for N in (3, 4, 5):
            full = list(range(1 << N))
            for x in range(1 << N):
                assert cycle.succ(x, N, t, bc) == full


def test_reflection_isomorphism_succ_obc0_odd_N():
    # Under obc0 with ODD N the even/odd sublattices are each reflection-
    # invariant AND there is no periodic seam, so reflecting the rule +
    # bit-reversing the states permutes succ sets EXACTLY.  (For even N
    # reflection swaps the two brickwork layers, and for pbc odd N the wrap-
    # around seam breaks per-state equality; only sector SIZES survive there,
    # and only for unitary rules -- see test_scc.)
    import random
    bc = "obc0"
    rng = random.Random(1)
    N = 5
    def revbits(x):
        return int(f"{x:0{N}b}"[::-1], 2)
    for _ in range(40):
        r = rng.randrange(256)
        t = rules.wolfram_to_tuple(r)
        tr = rules.reflect_tuple(t)
        for x in range(1 << N):
            s1 = {revbits(y) for y in cycle.succ(x, N, t, bc)}
            s2 = set(cycle.succ(revbits(x), N, tr, bc))
            assert s1 == s2


def _naive_support_succ(x, N, t, bc):
    """succ(x) by SUPPORT propagation only: no amplitudes, no cancellation.
    Always a superset of the exact succ; equal iff no path ever interferes.
    Mirrors the branch structure of one_cycle_branches (context R1 sec.3)."""
    steps = cycle._compile(N, t, bc)
    branches = [{x}]
    for (i, lpos, rpos, tt) in steps:
        bit = 1 << i
        nxt = []
        for S in branches:
            a0, a1 = set(), set()
            for s in S:
                sym = tt[cycle._mn(s, lpos, rpos)]
                if sym == "I":
                    a0.add(s)
                elif sym == "V":
                    a0.add(s & ~bit)
                    a0.add(s | bit)
                elif sym == "D":
                    (a1 if s & bit else a0).add((s & ~bit) if s & bit else s)
                else:  # E
                    (a0 if s & bit else a1).add(s if s & bit else (s | bit))
            if a0:
                nxt.append(a0)
            if a1:
                nxt.append(a1)
        branches = nxt
    out = set()
    for S in branches:
        out |= S
    return out


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_single_cycle_has_no_interference(bc, N):
    """R1 no-interference theorem: a single brick-wall cycle cannot cancel any
    amplitude, so exact succ equals naive support propagation -- for EVERY rule.
    The exact Z[1/sqrt2] arithmetic is therefore redundant for the transition
    graph (it is kept for norm certification and Tier 1b/1d)."""
    for r in range(256):
        t = rules.wolfram_to_tuple(r)
        for x in range(1 << N):
            assert set(cycle.succ(x, N, t, bc)) == _naive_support_succ(x, N, t, bc)


def test_repeated_hadamard_does_cancel():
    """The guard is not vacuous: two H on the SAME qubit (which the brick-wall
    never does) cancel exactly.  H.H|0> = |0>, so the support collapses to {0}
    where naive propagation would keep {0,1}."""
    t = ("V", "V", "V", "V")
    amps, m = {0: (1, 0)}, 0
    amps, m = cycle._apply_site_unitary(amps, m, 0, -1, -1, t)
    assert sorted(amps) == [0, 1]                 # after one H: superposition
    amps, m = cycle._apply_site_unitary(amps, m, 0, -1, -1, t)
    assert sorted(amps) == [0]                    # after H.H: |1> cancelled
