"""R15: the attractor tower, and whether the resets are idle inside it."""

import pytest

from qca_fragmentation.core import rules as R
from qca_fragmentation.graph import wcc
from qca_fragmentation.viz import tower

PHI = (1 + 5 ** 0.5) / 2

IDLE = [1, 9, 65, 110, 111, 124, 125, 126, 127, 129, 137, 193]


@pytest.mark.parametrize("rule", [0, 7, 28, 90, 156, 157, 201, 203, 232, 255])
@pytest.mark.parametrize("N", [6, 7, 8])
def test_the_hand_sweep_reproduces_the_engine(rule, N):
    """Everything in this report rests on the hand-rolled brick wall being the
    same map the engine uses; if it is not, the firing flags are meaningless."""
    t = R.wolfram_to_tuple(rule)
    succ = wcc.make_succ(rule, N, "obc0")
    for x in range(1 << N):
        mine = sorted({y for y, _ in tower.sweep_labeled(x, N, t, "obc0")})
        assert mine == sorted(succ(x)), (rule, N, x)


def test_a_reset_is_only_recorded_when_it_changes_the_bit():
    """D writes 0, E writes 1; neither does work on a bit already there."""
    # W204 = IIII: no reset anywhere, so nothing can ever fire
    assert tower.firing_sites(0b101010, 6, R.wolfram_to_tuple(204)) == set()
    # W0 = DDDD: every site is reset to 0, so exactly the set bits fire
    t0 = R.wolfram_to_tuple(0)
    assert "".join(t0) == "DDDD"
    for x in (0b000000, 0b000001, 0b101010, 0b111111):
        want = {i for i in range(6) if (x >> i) & 1}
        assert tower.firing_sites(x, 6, t0) == want, bin(x)


def test_the_tower_is_68_rules():
    tw = tower.tower_rules("obc0")
    assert len(tw) == 68
    assert all(abs(a - 1.0) < tower.A_FLAT and b > tower.B_GROW
               for _, a, b in tw)


def test_exactly_twelve_rules_have_idle_resets():
    rows = tower.census("obc0", 10)
    idle = sorted(r["rule"] for r in rows if r["idle"])
    assert idle == IDLE
    assert len(rows) == 68


def test_the_idle_rules_all_sit_at_phi_and_inherit_a_parent_sector():
    bt = {r: b for r, _, b in tower.tower_rules("obc0")}
    for rule in IDLE:
        assert bt[rule] == pytest.approx(PHI, abs=1e-5), rule
        c = tower.idle_census(rule, 10, "obc0")
        assert c["idle"] and c["agrees_with_parent"] and c["is_parent_sector"]
        assert c["size"] == c["parent_d_max"]


def test_the_two_idle_families():
    """V at r00 with resets from {D,I} -> parent 201; V at r11 with resets from
    {E,I} -> parent 108.  The reset target agrees with what the sites carry."""
    fam201, fam108 = [], []
    for rule in IDLE:
        t = "".join(R.wolfram_to_tuple(rule))
        c = tower.idle_census(rule, 10, "obc0")
        if c["parent"] == 201:
            fam201.append(t)
            assert t[0] == "V" and set(t[1:]) <= {"D", "I"}, t
        else:
            fam108.append(t)
            assert c["parent"] == 108
            assert t[3] == "V" and set(t[:3]) <= {"E", "I"}, t
    assert len(fam201) == 6 and len(fam108) == 6


def test_the_firing_rules_fire_throughout_the_bulk():
    """Not a boundary effect: the firing-site count grows with N."""
    rows = tower.census("obc0", 10)
    firing = [r["rule"] for r in rows if not r["idle"]]
    assert len(firing) == 56
    for rule in firing[:8]:                      # a sample, this is expensive
        n8 = tower.idle_census(rule, 8, "obc0")["n_firing_sites"]
        n12 = tower.idle_census(rule, 12, "obc0")["n_firing_sites"]
        assert n12 > n8, rule


def test_the_constant_does_not_identify_the_mechanism():
    """Nine rules reach phi WITH firing resets, so a base of phi does not imply
    the resets are idle."""
    rows = {r["rule"]: r for r in tower.census("obc0", 10)}
    at_phi = [r for r, _a, b in tower.tower_rules("obc0")
              if abs(b - PHI) < 1e-5]
    assert len(at_phi) == 21
    not_idle = sorted(r for r in at_phi if not rows[r]["idle"])
    assert not_idle == [11, 47, 81, 117, 139, 161, 171, 209, 241]


def test_most_bases_are_exact_algebraic_constants():
    dist = tower.base_distribution("obc0")
    total = sum(len(v["rules"]) for v in dist.values())
    exact = sum(v["n_exact"] for v in dist.values())
    assert total == 68
    assert exact == 61
    named = {v["name"] for v in dist.values() if v["name"]}
    assert any("varphi" in n for n in named)
    assert any("sqrt{3}" in n for n in named)


# --- the mechanism for the rules whose resets DO fire (R15 sec.3.4) -----------

SFT_GROUPS = {2.000000: [191, 231, 247], 2.414214: [167, 181],
              3.000000: [3, 183, 243], 3.414214: [35, 49, 211],
              3.561553: [19, 179, 227]}


@pytest.mark.parametrize("perron,rs", sorted(SFT_GROUPS.items()))
def test_the_attractor_is_a_subshift_on_the_unit_cell(perron, rs):
    """The grammar learned FROM the attractor must regenerate the attractor and
    nothing else -- at two sizes, so it is not a coincidence of one N."""
    for rule in rs:
        for N in (10, 12):
            c = tower.sft_check(rule, N, "obc0")
            assert c["exact"], (rule, N, c["size"], c["closure"])
            assert c["perron"] == pytest.approx(perron, abs=1e-5), rule


@pytest.mark.parametrize("rule,perron", [(191, 2.0), (167, 1 + 2 ** 0.5),
                                         (11, (3 + 5 ** 0.5) / 2), (3, 3.0),
                                         (35, 2 + 2 ** 0.5),
                                         (19, (3 + 17 ** 0.5) / 2)])
def test_the_base_is_the_square_root_of_the_block_perron_root(rule, perron):
    """b_att = sqrt(lambda_max(T)); the square root because a block is 2 sites."""
    bt = {r: b for r, _a, b in tower.tower_rules("obc0")}
    c = tower.sft_check(rule, 12, "obc0")
    assert c["perron"] == pytest.approx(perron, abs=1e-6)
    assert c["base"] == pytest.approx(perron ** 0.5, abs=1e-9)
    assert c["base"] == pytest.approx(bt[rule], abs=3e-3), rule


def test_the_subshift_names_a_constant_the_fit_only_approximated():
    """Rules 167/181 were fitted at 1.553861 and 1.553687; the transfer matrix
    says sqrt(1+sqrt2) = 1.5537740 exactly."""
    want = (1 + 2 ** 0.5) ** 0.5
    for rule in (167, 181):
        c = tower.sft_check(rule, 12, "obc0")
        assert c["exact"]
        assert c["base"] == pytest.approx(want, abs=1e-9)
    assert want == pytest.approx(1.5537740, abs=1e-7)


def test_the_closure_always_contains_the_attractor():
    """The grammar is learned from A, so A is a subset of its closure -- which
    is what makes sqrt(perron) an upper bound when the test fails."""
    for rule in (97, 169, 225, 191, 3):
        c = tower.sft_check(rule, 10, "obc0")
        assert c["closure"] >= c["size"], rule


def test_five_rules_are_not_subshifts_at_this_order():
    """Recorded so the mechanism is not overclaimed as universal."""
    for rule in (97, 117, 123, 169, 225):
        assert not tower.sft_check(rule, 12, "obc0")["exact"], rule
    # and for 169/225 the closure bound is strictly above the measured base
    bt = {r: b for r, _a, b in tower.tower_rules("obc0")}
    for rule in (169, 225):
        c = tower.sft_check(rule, 12, "obc0")
        assert bt[rule] < c["base"] - 1e-3, rule
        assert c["base"] == pytest.approx(3 ** 0.5, abs=1e-6)
