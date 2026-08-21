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
