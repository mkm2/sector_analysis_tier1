"""
Tier 1d pair-graph (G2) validation against the exact 4^N superoperator.

The pair graph is a SUPPORT-level upper bound on the unmonitored channel's
peripheral structure (T3), and the T4 certificate is a sufficient (not
necessary) condition.  These tests pin the two properties that must hold
exactly:

  * SOUNDNESS -- certified=True never coexists with genuine within-sector
    coherence (checked against the WCC-restricted superoperator);
  * CONTAINMENT (T3) -- every peripheral eigenoperator of the full channel has
    entrywise support inside the cross-sector P_rec;

plus the inequality chain K_M <= cesaro_rank <= fix_upper and the specific
anchor rules from task_tier1d_pair_graph.md.
"""
import numpy as np
import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.core.rules import wolfram_to_tuple, is_unitary, has_V
from qca_fragmentation.graph import scc
from qca_fragmentation.graph.pair_graph import (
    analyze_pair, labeled_supports, succ2_from, wcc_labels)
from qca_fragmentation.quantum import peripheral as pp

SAMPLE = [0, 22, 27, 28, 50, 76, 106, 178, 200, 232]


def _km(rule, N, bc, t):
    return len(scc.recurrent_classes(rule, N, bc, t))


def _within_sector_all_diagonal(rule, N, bc, t):
    """Oracle: every WCC-restricted peripheral eigenoperator is basis-diagonal.

    The full channel is block-diagonal across sectors (T1), so a WCC S is
    span-invariant and restricted_superoperator(S) is exact on that block.
    """
    wcc = wcc_labels(N, t, bc)
    groups = {}
    for x in range(1 << N):
        groups.setdefault(wcc[x], []).append(x)
    for S in groups.values():
        if len(S) == 1:
            continue
        Phi = pp.restricted_superoperator(sorted(S), N, t, bc)
        k = len(S)
        vals, vecs = np.linalg.eig(Phi)
        for j in np.where(np.abs(vals) > 1 - 1e-9)[0]:
            M = vecs[:, j].reshape(k, k)
            sc = np.abs(M).max() or 1.0
            if np.abs(M - np.diag(np.diag(M))).max() / sc > 1e-6:
                return False
    return True


# --------------------------------------------------------------------------- #
# Structural facts.                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rule", [22, 27, 28, 50, 178])
@pytest.mark.parametrize("bc", ["pbc", "obc0"])
def test_diagonal_subgraph_is_M(rule, bc):
    """The diagonal {(x,x)} of G2 is exactly the monitored graph M."""
    N = 4
    t = wolfram_to_tuple(rule)
    succ_fn = scc._make_succ(rule, N, bc, t)
    for x in range(1 << N):
        sx = labeled_supports(x, N, t, bc)
        diag_img = {xp for (xp, yp) in succ2_from(sx, sx) if xp == yp}
        assert diag_img == set(succ_fn(x)), (rule, bc, x)


@pytest.mark.parametrize("rule", [22, 28, 50, 106])
def test_swap_symmetry(rule):
    """(x,y)->(x',y') is an edge iff (y,x)->(y',x') is."""
    N = 4
    bc = "pbc"
    t = wolfram_to_tuple(rule)
    sup = {x: labeled_supports(x, N, t, bc) for x in range(1 << N)}
    for x in range(1 << N):
        for y in range(1 << N):
            fwd = set(succ2_from(sup[x], sup[y]))
            swp = {(b, a) for (a, b) in succ2_from(sup[y], sup[x])}
            assert fwd == swp, (rule, x, y)


# --------------------------------------------------------------------------- #
# Bounds and containment against the exact superoperator (N <= 5).            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rule", SAMPLE)
@pytest.mark.parametrize("bc", ["pbc", "obc0"])
def test_bound_chain(rule, bc):
    """K_M <= cesaro_rank <= fix_upper (cross-sector) -- T2 and T3."""
    N = 4
    t = wolfram_to_tuple(rule)
    K = _km(rule, N, bc, t)
    cz = pp.cesaro_rank(N, t, bc)
    fix_upper = analyze_pair(rule, N, bc, t, cross_sector=True).fix_upper
    assert K <= cz <= fix_upper, (rule, bc, K, cz, fix_upper)


@pytest.mark.parametrize("rule", SAMPLE)
def test_peripheral_support_containment(rule):
    """Every peripheral eigenoperator's support lies inside the cross-sector
    P_rec (T3), for both boundary conventions."""
    N = 4
    for bc in ("pbc", "obc0"):
        t = wolfram_to_tuple(rule)
        prec = analyze_pair(rule, N, bc, t, cross_sector=True,
                            return_support=True).prec
        S = pp.full_channel_superoperator(N, t, bc)
        dim = 1 << N
        vals, vecs = np.linalg.eig(S)
        for j in np.where(np.abs(vals) > 1 - 1e-9)[0]:
            M = vecs[:, j].reshape(dim, dim)
            sc = np.abs(M).max() or 1.0
            for a, b in np.argwhere(np.abs(M) > 1e-6 * sc):
                assert (int(a), int(b)) in prec, (rule, bc, int(a), int(b))


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
def test_certificate_is_sound(bc):
    """certified=True must never coexist with within-sector coherence.  Checked
    exhaustively over all 240 dissipative rules at N=4 against the WCC-restricted
    superoperator oracle."""
    N = 4
    for rule in range(256):
        t = wolfram_to_tuple(rule)
        if is_unitary(t):
            continue
        cert = analyze_pair(rule, N, bc, t, cross_sector=False).certified
        if cert is True:
            assert _within_sector_all_diagonal(rule, N, bc, t), (rule, bc)


# --------------------------------------------------------------------------- #
# Anchor rules.                                                               #
# --------------------------------------------------------------------------- #

def test_rule22_immortal_transient_and_bound():
    """Rule 22: the diagonal part of P_rec contains the fixed-operator support
    (11 states incl. M-transient ones); dim Fix = 25 <= |P_rec|."""
    N, bc = 4, "pbc"
    t = wolfram_to_tuple(22)
    res = analyze_pair(22, N, bc, t, cross_sector=True, return_support=True)
    diag = {x for (x, y) in res.prec if x == y}
    # exact fixed-operator support from the superoperator
    S = pp.full_channel_superoperator(N, t, bc)
    dim = 1 << N
    ns = pp._null_space(S - np.eye(dim * dim))
    supp = set()
    for j in range(ns.shape[1]):
        M = ns[:, j].reshape(dim, dim)
        sc = np.abs(M).max() or 1.0
        supp |= {int(a) for a, b in np.argwhere(np.abs(M) > 1e-7 * sc) if a == b}
    assert len(supp) == 11 and supp <= diag        # 11-state attractor support
    assert pp.cesaro_rank(N, t, bc) == 25 <= res.fix_upper
    # immortal transient support really is there: some diagonal (x,x) with x
    # not M-recurrent survives.
    m_rec = {x for cl in scc.recurrent_classes(22, N, bc, t) for x in cl}
    assert diag - m_rec, "rule 22 must have immortal transient support"


def test_rule27_offdiagonal_and_uncertified():
    """Rule 27: pure stationary superposition -> off-diagonal pairs present,
    certificate false."""
    N, bc = 4, "pbc"
    t = wolfram_to_tuple(27)
    res = analyze_pair(27, N, bc, t, cross_sector=False)
    assert res.pair_offdiag > 0
    assert res.certified is False


def test_rule28_uncertified_yet_zero_leakage():
    """Rule 28: certificate false (coherent attractors) even though the Tier-1a
    faithfulness leak vanishes -- the certificate is strictly stronger."""
    N, bc = 4, "pbc"
    t = wolfram_to_tuple(28)
    assert analyze_pair(28, N, bc, t, cross_sector=False).certified is False
    d = pp.graph_faithfulness(N, t, bc)
    assert d["leak_fix"] < 1e-9 and abs(d["leak_flow"]) < 1e-9


def test_vfree_cycle_coherence_refinement():
    """'V-free => certified' is FALSE.  A V-free rule with a period>=2 limit
    cycle carries a protected rotating coherence between the synchronised
    configurations and certifies False; a frozen-only V-free rule certifies
    True; an additive rule whose cycle labels de-synchronise dephases the
    coherence and certifies True even with a cycle."""
    N = 4
    # rule 10 (DEDD): a 2-cycle with synchronised Kraus labels -> coherence
    assert analyze_pair(10, N, "pbc", wolfram_to_tuple(10)).certified is False
    # rule 4 (IDDD, Fibonacci): frozen attractors only -> certified
    assert analyze_pair(4, N, "pbc", wolfram_to_tuple(4)).certified is True
    # rule 90 (DEED, additive): 3-cycle but labels de-synchronise -> certified
    r90 = analyze_pair(90, N, "obc0", wolfram_to_tuple(90))
    assert r90.certified is True
    # cross-check rule 10 against the superoperator: a real off-diagonal
    # peripheral eigenoperator exists.
    S = pp.full_channel_superoperator(N, wolfram_to_tuple(10), "pbc")
    vals, vecs = np.linalg.eig(S)
    dim = 1 << N
    nondiag = 0
    for j in np.where(np.abs(vals) > 1 - 1e-9)[0]:
        M = vecs[:, j].reshape(dim, dim)
        sc = np.abs(M).max() or 1.0
        if np.abs(M - np.diag(np.diag(M))).max() / sc > 1e-6:
            nondiag += 1
    assert nondiag > 0


def test_unitary_rules_are_skipped():
    """Unitary rules carry a single Kraus label; G2 = M x M, nothing to certify
    beyond Tier 1a/R2."""
    for rule in (150, 105, 90, 51):  # 90/51 unitary reps if applicable
        t = wolfram_to_tuple(rule)
        if not is_unitary(t):
            continue
        res = analyze_pair(rule, 4, "pbc", t)
        assert res.certified is True and res.pair_rec_size == 0
