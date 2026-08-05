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
