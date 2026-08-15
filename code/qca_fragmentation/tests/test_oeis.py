"""
The OEIS identifications, checked offline against the cached entries.

Nothing here touches the network: analytics/oeis_terms.json holds the entries
as fetched, and the tests re-derive our own side from the engine and the archive
every time.  So a test failure means our series moved, not that OEIS did.
"""

import pytest

from qca_fragmentation.core.rules import wolfram_to_tuple
from qca_fragmentation.scaling import oeis as OE

CACHE = OE.load_cache()

pytestmark = pytest.mark.skipif(
    not CACHE, reason="analytics/oeis_terms.json missing; run oeis --refresh")


@pytest.mark.parametrize("ident", OE.IDENTIFICATIONS,
                         ids=[f"{i.aid}-{i.rules[0]}-{i.key}"
                              for i in OE.IDENTIFICATIONS])
def test_every_identification_is_an_exact_contiguous_block(ident):
    r = OE.check(ident, CACHE)
    for row in r["rows"]:
        assert row["contiguous"], (ident.aid, row["rule"], "not contiguous")
        assert row["shift_ok"], (ident.aid, row["rule"],
                                 row["shift"], ident.shift)


@pytest.mark.parametrize("ident", [i for i in OE.IDENTIFICATIONS
                                   if len(i.rules) > 1],
                         ids=lambda i: f"{i.aid}-{i.key}-{i.bc}")
def test_grouped_rules_were_each_measured_not_inherited(ident):
    """A claim covering several rules must hold for each of them separately.

    Reflection is not a guaranteed sector-size symmetry for dissipative rules
    under the even-first convention, so W28 = W70 has to be a measurement.
    """
    ref = OE.local_series(ident.rules[0], ident.bc, ident.key)
    for rule in ident.rules[1:]:
        assert OE.local_series(rule, ident.bc, ident.key) == ref, rule


def test_no_identification_rests_on_fewer_than_twelve_terms():
    for e in OE.verify(CACHE):
        for row in e["rows"]:
            assert row["n_terms"] >= 12, (e["aid"], row)


def test_the_engine_and_the_archive_agree_on_every_dissipative_overlap():
    """diss_series raises if they disagree; call it for every rule in play."""
    rules = sorted({r for i in OE.IDENTIFICATIONS for r in i.rules
                    if not is_unitary_rule(r)}
                   | {r for rs, _, _ in OE.NO_MATCH for r in rs})
    for bc in ("obc0", "pbc"):
        for rule in rules:
            s = OE.diss_series(rule, bc)
            assert s["N"] == list(range(1, 18)), (rule, bc, s["N"])


def is_unitary_rule(rule: int) -> bool:
    from qca_fragmentation.core.rules import is_unitary
    return is_unitary(wolfram_to_tuple(rule))


def test_rule_28_and_rule_70_land_on_the_dying_rabbit_recurrence():
    """The sequence that falsifies the phi prediction of task sec.6.

    test_wcc pins the recurrence a(n) = a(n-1) + a(n-2) - a(n-4); this pins
    that it is the OEIS entry of that name, and that W70 gives the same.
    """
    for rule in (28, 70):
        _, ys = OE.local_series(rule, "obc0", "n_recurrent")
        for n in range(4, len(ys)):
            assert ys[n] == ys[n - 1] + ys[n - 2] - ys[n - 4], (rule, n)
    ident = next(i for i in OE.IDENTIFICATIONS if i.aid == "A023434")
    assert ident.rules == (28, 70)
    assert OE.check(ident, CACHE)["ok"]


def test_the_copy_paste_entry_reproduces_the_four_fifths_ladder():
    """A178715's own recurrence a(n) = 4 a(n-5) is where 4^(1/5) comes from."""
    terms = OE.entry("A178715", CACHE)["terms"]
    off = OE.entry("A178715", CACHE)["offset"]
    for n in range(16, len(terms) + off):
        assert terms[n - off] == 4 * terms[n - 5 - off], n
    _, ys = OE.local_series(156, "obc0", "d_max")
    assert ys == [terms[N - off] for N in range(6, 6 + len(ys))]


def test_the_closed_forms_stated_in_the_table_actually_hold():
    """The three closed forms that are cheap to re-evaluate."""
    _, dm = OE.local_series(28, "obc0", "d_max")
    assert dm == [2 ** ((N + 1) // 3) for N in range(1, len(dm) + 1)]

    from math import comb
    Ns, dm = OE.local_series(105, "obc0", "d_max")
    assert dm == [comb(N + 1, (N + 1) // 2) for N in Ns]

    Ns, fr = OE.local_series(150, "obc0", "n_frozen")
    assert fr == [1 + (N % 2) for N in Ns]


def test_the_unmatched_series_are_recorded_and_still_computable():
    assert OE.NO_MATCH
    for rules, bc, key in OE.NO_MATCH:
        ref = OE.local_series(rules[0], bc, key)
        assert len(ref[1]) >= 12
        for rule in rules[1:]:
            assert OE.local_series(rule, bc, key) == ref, (rule, bc, key)


def test_the_two_pbc_branches_of_the_rho_group_are_genuinely_different():
    """28/70/157/199 and 29/71 split on the ring but not at obc0.

    Their class counts differ at every N >= 2; their D_max ladders differ only
    at N = 1, where the "ring" is a single site.  Both facts are why NO_MATCH
    lists the branches separately.
    """
    a = OE.local_series(28, "pbc", "n_recurrent")[1]
    b = OE.local_series(29, "pbc", "n_recurrent")[1]
    assert a != b
    assert a[1:] != b[1:]
    assert OE.local_series(28, "pbc", "d_max")[1][1:] == \
        OE.local_series(29, "pbc", "d_max")[1][1:]
    assert OE.local_series(29, "obc0", "n_recurrent")[1] == \
        OE.local_series(157, "obc0", "n_recurrent")[1]


def test_the_unmatched_pbc_ladder_still_obeys_an_exact_recurrence():
    """No OEIS entry, but not structureless: a(N) = 2 a(N-3) from N = 5."""
    for rule in (28, 29, 70, 71, 157, 199):
        Ns, ys = OE.local_series(rule, "pbc", "d_max")
        for i, N in enumerate(Ns):
            if N >= 5:
                assert ys[i] == 2 * ys[i - 3], (rule, N)


def test_every_cached_entry_is_used_and_every_used_entry_is_cached():
    used = {i.aid for i in OE.IDENTIFICATIONS}
    cached = {k for k in CACHE if k != "_fetched"}
    assert used == cached, (used - cached, cached - used)


def test_the_counts_R20_quotes_are_the_counts_in_the_table():
    """R20's abstract states 24 identifications over 38 (rule, observable)
    pairs from 18 distinct entries.  Pinned so the prose cannot drift from the
    table it describes."""
    assert len(OE.IDENTIFICATIONS) == 24
    assert sum(len(i.rules) for i in OE.IDENTIFICATIONS) == 38
    assert len({i.aid for i in OE.IDENTIFICATIONS}) == 18


def test_the_growth_constants_are_the_ones_the_reports_quote():
    assert abs(OE.RHO ** 3 - OE.RHO - 1) < 1e-12
    assert abs(OE.PSI ** 3 - OE.PSI ** 2 - 1) < 1e-12
    assert abs(OE.PHI ** 2 - OE.PHI - 1) < 1e-12
    assert abs(OE.QUARTIC ** 5 - 4) < 1e-12
    assert abs(OE.CBRT2 ** 3 - 2) < 1e-12
