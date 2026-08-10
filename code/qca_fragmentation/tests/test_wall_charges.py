"""R18: are 108/201 and 156/198 explained by two conservation laws, or by walls?

The claims are checked on freshly built sectors at small N rather than against
the cached JSON, so a regression in `succ` or in the wall detector fails here.
"""

import pytest

from qca_fragmentation.scaling import wall_charges as WC

BC = "obc0"

#: The words Tier-2's independent wall-grammar module (qca/walls.py) reports for
#: these rules.  The detector in wall_charges.py is a separate implementation;
#: this is the cross-check between the two.
TIER2_WORDS = {108: "00", 201: "11", 156: "01", 198: "10"}


@pytest.mark.parametrize("rule,word", sorted(TIER2_WORDS.items()))
def test_the_wall_detector_agrees_with_tier2(rule, word):
    """Two independent implementations, one answer. If this fails the Tier-1
    detector and the Tier-2 grammar have diverged and R18 rests on neither."""
    words = WC.wall_words(rule)
    assert words == [word], (rule, words)


@pytest.mark.parametrize("rule", WC.RULES)
def test_there_is_no_charge_in_the_computational_basis(rule):
    """Magnetisation is not conserved -- the Hadamard flips a site outright --
    so the charge/dipole ladder never starts. Checked directly rather than via
    the null-space detector."""
    for N in (8, 10):
        g = {}
        for x, cid in enumerate(WC.sector_ids(rule, N, BC)):
            g.setdefault(cid, []).append(x)
        moved = any(len({bin(x).count("1") for x in mem}) > 1
                    for mem in g.values())
        assert moved, (rule, N)


@pytest.mark.parametrize("rule,expect", [(108, "S"), (201, "S"),
                                         (156, "D"), (198, "D")])
def test_exactly_one_of_the_two_domain_wall_charges_is_conserved(rule, expect):
    """The PXP-like pair conserves the STAGGERED wall number, the chiral pair the
    TOTAL one, and in each case the other is broken. That contrast is the whole
    reason the two families have different exponents."""
    for N in (8, 10):
        for bc in ("obc0", "pbc"):
            st = WC.charge_status(rule, N, bc)
            assert st[expect] is True, (rule, N, bc, st)
            other = "D" if expect == "S" else "S"
            assert st[other] is False, (rule, N, bc, st)


@pytest.mark.parametrize("rule", WC.RULES)
def test_the_wall_set_is_a_complete_invariant(rule):
    """The headline: same wall occurrences <=> same Krylov sector. Constant on
    sectors AND injective across them, and the label is computed from a single
    state with no reference to the sector."""
    for N in (8, 10, 12):
        r = WC.wall_completeness(rule, N, BC, TIER2_WORDS[rule])
        assert r["constant_on_sectors"], (rule, N)
        assert r["complete"], (rule, N, r)
        assert r["n_wall_sets"] == r["n_sectors"]


@pytest.mark.parametrize("rule", WC.RULES)
def test_the_wall_partition_reproduces_every_sector_size(rule):
    """Not just the count: the multiset of preimage sizes of the wall map equals
    the multiset of sector sizes, so D_max comes out of the grammar too."""
    for N in (8, 10, 12):
        r = WC.wall_completeness(rule, N, BC, TIER2_WORDS[rule])
        assert r["sizes_match"], (rule, N)
        assert r["d_max_predicted"] == r["d_max_actual"]


def test_obc0_must_search_the_padded_chain():
    """A boundary detail with teeth. obc0 pads with a frozen 0 at each end and
    those zeros are wall material for W108, whose word is '00'. Searching the
    bare chain loses more than a third of the sectors, so this is pinned."""
    N = 12
    bare = {frozenset(i for i in range(N - 1)
                      if ((x >> i) & 1) == 0 and ((x >> (i + 1)) & 1) == 0)
            for x in range(1 << N)}
    padded = {WC.wall_set(x, N, "00", "obc0") for x in range(1 << N)}
    assert len(padded) > len(bare)
    r = WC.wall_completeness(108, N, "obc0", "00")
    assert len(padded) == r["n_sectors"]


@pytest.mark.parametrize("rule", WC.RULES)
def test_the_charge_is_a_function_of_the_walls(rule):
    """The adjudication. The conserved charge carries strictly less information
    than the wall set, so it is a shadow of the grammar rather than a second
    independent ingredient."""
    cf = WC.charge_from_walls(rule, 10, BC, TIER2_WORDS[rule])
    live = [v for v in cf.values() if v]
    assert live, rule
    assert all(v["determined_by_walls"] for v in live), (rule, cf)


@pytest.mark.parametrize("rule", (156, 198))
def test_the_chiral_charge_is_twice_the_wall_count(rule):
    """For the chiral pair the relation is explicit: D = 2|W|, because on a ring
    the '01' and '10' occurrences are equinumerous and D counts both."""
    cf = WC.charge_from_walls(rule, 10, BC, TIER2_WORDS[rule])
    assert cf["D"]["affine_in_wall_count"] == [2.0, 0.0]


@pytest.mark.parametrize("rule", WC.RULES)
def test_symmetry_sectors_are_polynomial_while_krylov_sectors_are_not(rule):
    """Two conservation laws cannot explain the fragmentation because they do
    not have the cardinality to: their level sets grow like a polynomial while
    the sectors grow like an exponential, so the ratio must diverge."""
    ratios = [WC.resolving_power(rule, N, BC)["sectors_per_level"]
              for N in (8, 10, 12)]
    assert ratios[0] > 1.0
    assert ratios[1] > ratios[0] and ratios[2] > ratios[1]
    assert ratios[2] / ratios[0] > 3.0


@pytest.mark.parametrize("rule,root", [(108, 1.754878), (201, 1.754878),
                                       (156, 1.618034), (198, 1.618034)])
def test_the_sector_exponent_falls_out_of_the_wall_recurrence(rule, root):
    """rho^2 for the PXP-like pair, phi for the chiral one -- from an exact
    integer recurrence on the number of wall sets, not from a fit."""
    counts = WC.wall_set_counts(rule, BC, TIER2_WORDS[rule], range(4, 15))
    rec = WC.linear_recurrence(counts)
    assert rec is not None, rule
    assert rec["dominant_root"] == pytest.approx(root, abs=1e-5)
    assert rec["dominant_root"] == pytest.approx(WC.R9_SECTOR_BASE[rule],
                                                 abs=1e-5)


def test_the_ring_defect_is_exactly_the_two_uniform_states():
    """On the ring the chiral pair has one collision at every N: 0^N and 1^N are
    both frozen singletons with no '01' between them. Stated as an exact
    off-by-one rather than hidden, because it is the only place where the wall
    label is not complete."""
    for rule in (156, 198):
        for N in (8, 10):
            r = WC.wall_completeness(rule, N, "pbc", TIER2_WORDS[rule])
            assert r["constant_on_sectors"]
            assert r["defect"] == 1, (rule, N, r)
            assert r["n_collisions"] == 1
    for rule in (108, 201):                    # no such defect here
        for N in (8, 10):
            assert WC.wall_completeness(rule, N, "pbc",
                                        TIER2_WORDS[rule])["complete"]
