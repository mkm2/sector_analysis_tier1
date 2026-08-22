"""Regression tests for R23: the six bounded rules' attractors are subcubes
built from a wall tiling, every attractor is decoherence-free, and only 29/71
put more than one attractor in a weak component."""

import numpy as np
import pytest

from qca_fragmentation.scaling import stripes as st

SIX = (28, 70, 157, 199, 29, 71)


@pytest.mark.parametrize("rule", SIX)
@pytest.mark.parametrize("N", [8, 10, 11])
def test_every_attractor_is_a_subcube(rule, N):
    d = st.decompose(rule, N)
    for c, is_term in enumerate(d["terminal"]):
        if not is_term:
            continue
        m = d["members"][c]
        assert st.is_subcube(m, st.attractor_word(m, N)), (rule, N, c)


@pytest.mark.parametrize("rule", SIX)
@pytest.mark.parametrize("N", [8, 10, 11])
def test_attractor_words_are_exactly_the_tilings(rule, N):
    d = st.decompose(rule, N)
    got = {st.attractor_word(d["members"][c], N)
           for c, t in enumerate(d["terminal"]) if t}
    assert got == st.predicted_words(rule, N), (rule, N)


@pytest.mark.parametrize("N", [8, 10, 12])
def test_largest_attractor_is_all_wide_tiles(N):
    # 2^floor((N+1)/3) -- R17 measured this, the tiling grammar explains it
    for rule in SIX:
        d = st.decompose(rule, N)
        dim = max(len(d["members"][c])
                  for c, t in enumerate(d["terminal"]) if t)
        assert dim == 2 ** ((N + 1) // 3), (rule, N, dim)


@pytest.mark.parametrize("rule,N,expect_split", [
    (28, 10, False), (70, 10, False), (157, 10, False), (199, 10, False),
    (73, 10, False), (109, 10, False), (29, 10, True), (71, 10, True)])
def test_only_29_and_71_split_a_weak_component(rule, N, expect_split):
    d = st.decompose(rule, N)
    seen = {}
    for c, t in enumerate(d["terminal"]):
        if t:
            seen.setdefault(d["wcc"][d["members"][c][0]], []).append(c)
    split = any(len(v) > 1 for v in seen.values())
    assert split is expect_split, (rule, N, {k: len(v) for k, v in seen.items()})


@pytest.mark.parametrize("rule", [29, 71])
def test_the_branch_is_a_fair_coin_resolved_in_one_period(rule):
    N = 9
    d, A, tc, lab, P = st.absorption(rule, N)
    split = [x for x in range(1 << N) if (A[x] > 1e-12).sum() >= 2]
    assert split, "29/71 must have shared-route states"
    for x in split:
        nz = sorted(float(v) for v in A[x] if v > 1e-12)
        k = len(nz)
        assert k in (2, 4) and k & (k - 1) == 0, (rule, x, nz)
        assert np.allclose(nz, 1.0 / k), (rule, x, nz)
        # every successor has already committed
        for y in P[x]:
            assert (A[y] > 1e-12).sum() == 1, (rule, x, y)


@pytest.mark.parametrize("rule", [28, 199, 29, 73, 109])
def test_attractors_are_decoherence_free(rule):
    N = 9
    d = st.decompose(rule, N)
    tc = sorted((c for c, t in enumerate(d["terminal"]) if t),
                key=lambda c: -len(d["members"][c]))
    tr = st.Trajectory(rule, N)
    m = d["members"][tc[0]]
    assert len(m) > 1
    rng = np.random.default_rng(0)
    psi = np.zeros(1 << N)
    v = rng.normal(size=len(m))
    psi[list(m)] = v / np.linalg.norm(v)
    for _ in range(12):
        pj, dn = tr.jump_budget(psi)
        assert pj == 0.0 and dn < 1e-12, (rule, pj, dn)
        psi = tr.step(psi, np.random.default_rng(0))
        assert set(np.nonzero(np.abs(psi) > 1e-11)[0].tolist()) <= set(m)


@pytest.mark.parametrize("rule", [28, 199, 29])
def test_period_two_and_zero_entanglement(rule):
    N, tr = 11, None
    tr = st.Trajectory(rule, 11)
    rng = np.random.default_rng(3)
    psi = st.basis_state(11, int(rng.integers(1 << 11)))
    for _ in range(80):
        psi = tr.step(psi, rng)
    o0 = tr.occupation(psi)
    psi = tr.step(psi, rng)
    psi = tr.step(psi, rng)
    assert np.abs(tr.occupation(psi) - o0).max() < 1e-12
    assert abs(tr.entropy(psi, 5)) < 1e-12


@pytest.mark.parametrize("rule,dim", [(73, "F(N+2)"), (109, "F(N)")])
def test_73_and_109_reach_the_hard_core_space(rule, dim):
    N = 11
    d = st.decompose(rule, N)
    tc = sorted((c for c, t in enumerate(d["terminal"]) if t),
                key=lambda c: -len(d["members"][c]))
    m = d["members"][tc[0]]
    fib = [0, 1]
    while len(fib) < N + 4:
        fib.append(fib[-1] + fib[-2])
    assert len(m) == (fib[N + 2] if rule == 73 else fib[N])
    bad = "11" if rule == 73 else "00"
    words = ["".join(str((x >> i) & 1) for i in range(N)) for x in m]
    assert all(bad not in w for w in words)
    # and it is NOT a product subspace across the middle cut
    cut = N // 2
    L = {x >> cut for x in m}
    R = {x & ((1 << cut) - 1) for x in m}
    assert len(m) < len(L) * len(R)


@pytest.mark.parametrize("rule,a", [(73, 0), (109, 1)])
def test_induced_unitary_is_blockade_controlled_hadamard(rule, a):
    """R23: on its largest attractor each of 73/109 acts as the brickwork
    product of P^a H P^a -- PXP with the spin flip replaced by a Hadamard."""
    N = 9
    inv = 1 / np.sqrt(2)
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) * inv

    d = st.decompose(rule, N)
    tc = sorted((c for c, t in enumerate(d["terminal"]) if t),
                key=lambda c: -len(d["members"][c]))
    m = sorted(d["members"][tc[0]])
    idx = {x: j for j, x in enumerate(m)}
    tr = st.Trajectory(rule, N)

    A = np.zeros((len(m), len(m)))
    for j, x in enumerate(m):
        psi = tr.step(st.basis_state(N, x), np.random.default_rng(0))
        for y in np.nonzero(np.abs(psi) > 1e-12)[0]:
            assert int(y) in idx, "the channel left the attractor"
            A[idx[int(y)], j] = psi[y]
    assert np.abs(A @ A.T - np.eye(len(m))).max() < 1e-12

    U = np.eye(1 << N)
    for layer in (range(0, N, 2), range(1, N, 2)):
        for i in layer:
            M = np.zeros((1 << N, 1 << N))
            for x in range(1 << N):
                left = (x >> (i - 1)) & 1 if i > 0 else 0
                right = (x >> (i + 1)) & 1 if i < N - 1 else 0
                if left == a and right == a:
                    b = (x >> i) & 1
                    M[x & ~(1 << i), x] += H[0, b]
                    M[x | (1 << i), x] += H[1, b]
                else:
                    M[x, x] += 1.0
            U = M @ U
    assert np.abs(A - U[np.ix_(m, m)]).max() < 1e-12


def _induced_on_recurrent(rule, N):
    """The map the channel induces on the span of all terminal SCCs."""
    d = st.decompose(rule, N)
    rec = sorted(x for c, t in enumerate(d["terminal"]) if t
                 for x in d["members"][c])
    idx = {x: j for j, x in enumerate(rec)}
    tr = st.Trajectory(rule, N)
    A = np.zeros((len(rec), len(rec)))
    for j, x in enumerate(rec):
        psi = tr.step(st.basis_state(N, x), np.random.default_rng(0))
        for y in np.nonzero(np.abs(psi) > 1e-12)[0]:
            assert int(y) in idx, "recurrent space is not closed"
            A[idx[int(y)], j] = psi[y]
    return A


@pytest.mark.parametrize("rule", [28, 70, 157, 199, 29, 71])
def test_recurrent_space_carries_an_involution(rule):
    """R23: for the six, the induced map on the WHOLE recurrent space is
    unitary with exactly two eigenvalues, so U^2 = 1 and period 2 is forced."""
    A = _induced_on_recurrent(rule, 6)
    assert np.abs(A @ A.T - np.eye(len(A))).max() < 1e-12
    assert np.abs(A @ A - np.eye(len(A))).max() < 1e-12
    ev = np.linalg.eigvals(A)
    assert len({complex(np.round(z, 6)) for z in ev}) == 2


@pytest.mark.parametrize("rule", [28, 199, 29, 73, 109])
def test_coherence_between_two_attractors_survives(rule):
    """R23: the no-jump Kraus operator serves every enclosure at once, so a
    superposition drawn from two different attractors is never decohered."""
    N = 9
    d = st.decompose(rule, N)
    tc = sorted((c for c, t in enumerate(d["terminal"]) if t),
                key=lambda c: -len(d["members"][c]))
    a, b = d["members"][tc[0]], d["members"][tc[1]]
    tr = st.Trajectory(rule, N)
    psi = np.zeros(1 << N)
    psi[a[0]] = psi[b[0]] = 1 / np.sqrt(2)
    for _ in range(12):
        pj, dn = tr.jump_budget(psi)
        assert pj == 0.0 and dn < 1e-12
        psi = tr.step(psi, np.random.default_rng(0))
        assert set(np.nonzero(np.abs(psi) > 1e-11)[0].tolist()) <= set(a) | set(b)
    assert abs(sum(psi[x] ** 2 for x in a) - 0.5) < 1e-12


def _constrained(N, bad):
    return [x for x in range(1 << N)
            if bad not in "".join(str((x >> i) & 1) for i in range(N))]


def _cycle_on(rule, N, basis):
    tr = st.Trajectory(rule, N)
    idx = {x: j for j, x in enumerate(basis)}
    M = np.zeros((len(basis), len(basis)))
    for j, x in enumerate(basis):
        psi = tr.step(st.basis_state(N, x), np.random.default_rng(0))
        for y in np.nonzero(np.abs(psi) > 1e-13)[0]:
            assert int(y) in idx, "amplitude left the constrained space"
            M[idx[int(y)], j] = psi[y]
    return M


@pytest.mark.parametrize("child,parent,bad", [(73, 201, "11"), (109, 108, "00")])
@pytest.mark.parametrize("N", [6, 8, 9])
def test_child_acts_as_its_coherent_parent_on_the_constrained_space(
        child, parent, bad, N):
    """R23: the induced unitary is literally the coherent parent -- 201 IS the
    P^0 H P^0 Floquet model, 108 IS P^1 H P^1, and the reset cannot fire."""
    b = _constrained(N, bad)
    tr = st.Trajectory(child, N)
    for x in b:
        assert tr.jump_budget(st.basis_state(N, x))[0] == 0.0
    Mc, Mp = _cycle_on(child, N, b), _cycle_on(parent, N, b)
    assert np.abs(Mc - Mp).max() < 1e-12
    assert np.abs(Mc.T @ Mc - np.eye(len(b))).max() < 1e-12


@pytest.mark.parametrize("N", [9, 11])
def test_constrained_space_is_irreducible_for_73_but_not_109(N):
    """R23: invariance is not irreducibility.  73's no-11 space is one
    enclosure; 109's no-00 space splits into exactly four, no transients."""
    for rule, bad, want in [(73, "11", 1), (109, "00", 4)]:
        C = set(_constrained(N, bad))
        d = st.decompose(rule, N)
        inside = [c for c, t in enumerate(d["terminal"])
                  if t and set(d["members"][c]) <= C]
        straddle = [c for c, t in enumerate(d["terminal"])
                    if t and set(d["members"][c]) & C
                    and not set(d["members"][c]) <= C]
        assert len(inside) == want, (rule, N, len(inside))
        assert not straddle
        assert sum(len(d["members"][c]) for c in inside) == len(C)


def _protected_dim(rule, N):
    d = st.decompose(rule, N)
    return sum(len(d["members"][c]) for c, t in enumerate(d["terminal"]) if t)


@pytest.mark.parametrize("rule,expect", [
    (73,  [13, 24, 44, 81, 149, 274, 504]),
    (109, [9, 17, 31, 57, 105, 193, 355]),
])
def test_protected_dimension_is_tribonacci(rule, expect):
    """R23: D = dim of the whole protected space obeys the tribonacci
    recurrence, so it grows at 1.8393 -- faster than any single enclosure."""
    got = [_protected_dim(rule, N) for N in range(4, 11)]
    assert got == expect
    for i in range(3, len(got)):
        assert got[i] == got[i - 1] + got[i - 2] + got[i - 3]


@pytest.mark.parametrize("rule,expect", [
    (28,  [9, 14, 20, 33, 49, 74, 116]),
    (199, [5, 9, 11, 19, 29, 41, 67]),
    (29,  [5, 9, 11, 19, 29, 41, 67]),
])
def test_protected_dimension_of_the_six(rule, expect):
    """R23: the six share the root of x^3 = x + 2 through two different
    integer recurrences."""
    got = [_protected_dim(rule, N) for N in range(4, 11)]
    assert got == expect
    if rule == 28:
        for i in range(4, len(got)):
            assert got[i] == got[i-1] + got[i-2] + got[i-3] - 2 * got[i-4]
    else:
        for i in range(3, len(got)):
            assert got[i] == got[i - 2] + 2 * got[i - 3]


@pytest.mark.parametrize("child,parent", [
    (28, 156), (157, 156), (29, 156), (70, 198), (199, 198), (71, 198)])
def test_the_six_also_act_as_their_coherent_parent(child, parent):
    """R23: every non-trivial terminal SCC of the six is exactly a Krylov
    sector of 156/198, and the channel matches the parent on it."""
    N = 9
    d, dp = st.decompose(child, N), st.decompose(parent, N)
    seen = 0
    for c, t in enumerate(d["terminal"]):
        if not t or len(d["members"][c]) < 2:
            continue
        seen += 1
        m = sorted(d["members"][c])
        assert set(dp["members"][dp["comp"][m[0]]]) == set(m)
        assert np.abs(_cycle_on(child, N, m) - _cycle_on(parent, N, m)).max() < 1e-12
    assert seen > 0


@pytest.mark.parametrize("rule", [28, 70, 157, 199, 29, 71])
def test_induced_unitary_is_a_tensor_product_of_hadamards(rule):
    """R23: on each attractor of the six the induced unitary is exactly
    (x) H over the free sites -- whence period 2 and zero entanglement."""
    N = 9
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    d = st.decompose(rule, N)
    for c, t in enumerate(d["terminal"]):
        if not t or len(d["members"][c]) < 2:
            continue
        m = sorted(d["members"][c])
        free = [i for i, ch in enumerate(st.attractor_word(m, N)) if ch == "f"]
        order = sorted(m, key=lambda x: tuple((x >> i) & 1 for i in free))
        perm = [m.index(x) for x in order]
        M = _cycle_on(rule, N, m)[np.ix_(perm, perm)]
        ref = np.array([[1.0]])
        for _ in free:
            ref = np.kron(ref, H)
        assert np.abs(M - ref).max() < 1e-12
