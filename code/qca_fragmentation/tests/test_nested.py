"""R13: nested fragmentation -- can a rule fragment twice?"""

import pytest

from qca_fragmentation.scaling.fits import find_integer_recurrence
from qca_fragmentation.viz import nested


def _att(rule, Ns):
    return nested.census(rule, Ns, "obc0")["n_att"]


PSI = 1.4655712318767682
RHO = 1.3247179572447454
X4 = 1.3802775690976143


@pytest.mark.parametrize("rule", [203, 217, 219, 36, 44, 100])
def test_the_psi_family_obeys_the_narayana_recurrence(rule):
    """a_N = a_{N-1} + a_{N-3}, root of x^3 = x^2 + 1.  Exact, not fitted."""
    ys = _att(rule, range(6, 15))
    rec = find_integer_recurrence(list(ys))
    assert rec["ok"] and rec["coeffs"] == [1, 0, 1]
    assert rec["base"] == pytest.approx(PSI, abs=1e-9)


@pytest.mark.parametrize("rule", [104, 233])
def test_the_x4_family(rule):
    """a_N = a_{N-1} + a_{N-4}, root of x^4 = x^3 + 1."""
    rec = find_integer_recurrence(list(_att(rule, range(6, 15))))
    assert rec["ok"] and rec["coeffs"] == [1, 0, 0, 1]
    assert rec["base"] == pytest.approx(X4, abs=1e-9)


@pytest.mark.parametrize("rule", [29, 71])
def test_the_plastic_family(rule):
    """a_N = a_{N-2} + a_{N-3}, root of x^3 = x + 1 (Padovan)."""
    rec = find_integer_recurrence(list(_att(rule, range(6, 15))))
    assert rec["ok"] and rec["coeffs"] == [0, 1, 1]
    assert rec["base"] == pytest.approx(RHO, abs=1e-9)


def test_the_attractors_per_sector_are_consecutive_narayana_numbers():
    """R13's prettiest fact: for rules 44 and 100 the sorted per-sector
    attractor counts ARE a run of the same sequence that gives the total."""
    nar = set(nested.narayana(24))
    for rule in (44, 100):
        dist = nested.census(rule, [16], "obc0")["dist"]
        head = dist[:10]
        assert head == [189, 129, 88, 60, 41, 28, 19, 13, 9, 6]
        assert all(v in nar for v in head)
        # consecutive: each is the next Narayana number below the previous
        order = sorted(nar)
        idx = [order.index(v) for v in head]
        assert idx == list(range(idx[0], idx[0] - len(idx), -1))


def test_three_rules_share_a_total_and_split_it_differently():
    """36, 44 and 100 all have 595 attractors at N=16; 36 puts 586 in one
    sector, 44/100 spread them as a Narayana staircase."""
    tot = {}
    for rule in (36, 44, 100):
        c = nested.census(rule, [16], "obc0")
        tot[rule] = (sum(c["dist"]), max(c["dist"]), len(c["dist"]))
    assert {v[0] for v in tot.values()} == {595}
    assert tot[36][1] == 586 and tot[36][2] == 10
    assert tot[44][1] == 189 and tot[100][1] == 189


def test_one_sector_can_hold_exponentially_many_attractors():
    """The settled half of R13's question: rules 203/217 have n_wcc = 1 at every
    N while their attractor count grows like psi^N."""
    for rule in (203, 217):
        c = nested.census(rule, range(6, 15), "obc0")
        assert set(c["n_wcc"]) == {1}
        assert c["n_att"] == [6, 9, 13, 19, 28, 41, 60, 88, 129]
        assert c["max_att"] == c["n_att"]        # all of them in that one sector


def test_rule_29_is_recorded_as_open_not_as_a_result():
    """R13 sec.4.  The classifier calls it exponential; the model comparison
    says quadratic and exponential are tied.  The report must not claim it."""
    c = nested.rule29_model_comparison()
    assert c["verdict"] == "open"
    assert abs(c["quadratic"]["bic"] - c["exponential"]["bic"]) < 3.0
    assert abs(c["quadratic"]["at_22"] - c["exponential"]["at_22"]) < 1.0
    assert c["second_differences"] == [0, 0, 0, 1, 1, 1]


def test_only_29_and_71_have_an_exponential_sector_count():
    """So they are the only candidates for the fully nested case."""
    d = nested.load("obc0") or nested.build(bc="obc0")
    r = nested.exclusion_check(data=d)
    assert sorted(r["exponential_sectors"]) == [29, 71]
