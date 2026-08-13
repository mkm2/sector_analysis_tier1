"""The gate on the census extension (extend_unitary.py).

The extension appends new N to the Tier-1e store using the flip reduction
rather than either sweep's own engine. What makes that safe is not the theorem
alone -- tests/test_flip_graph.py already checks the theorem -- but that the
extending code reproduces every unit the sweeps already computed for the rule
it is about to extend, in both stores and by two unrelated algorithms.
"""

import pytest

from qca_fragmentation import extend_unitary as EU
from qca_fragmentation import results_io

BC = "obc0"


def test_the_reduction_is_offered_only_where_it_is_proved():
    with pytest.raises(ValueError):
        EU.sizes(90, 8, BC)          # rule 90 has D/E channels
    assert len(EU.sizes(51, 8, BC)) == 1


@pytest.mark.parametrize("rule", EU.UNITARY)
def test_every_stored_value_is_a_value_computed_for_that_rule(rule):
    """No number in the census is inherited from a symmetry partner.

    Reflection is a symmetry of these series and the sweeps offer a
    representatives-only mode, so `W198 = W156` could in principle have been a
    substitution rather than a measurement. This recomputes every stored unit
    of every unitary rule from that rule's OWN gate assignment -- label
    propagation on its own flip graph, consulting no partner -- and requires it
    to reproduce what both stores hold. Tier-1e ran union-find and Tier-1a ran
    Tarjan, so three unrelated algorithms have to agree, rule by rule.

    That is also the gate the extension writes behind.
    """
    checks = EU.agrees_with_store(rule, BC)
    assert checks, rule
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    assert any(store == "1a" for store, _ in checks)
    assert any(store == "1e" for store, _ in checks)


def test_the_reflection_pairs_were_measured_separately():
    """The stronger form of the same point for the four pairs that agree: each
    member has its own runtimes in the archive, i.e. its own sweep, so the
    agreement in Table 2 is a result rather than a definition."""
    for a, b in ((156, 198), (60, 102), (57, 99), (153, 195)):
        ra = results_io.load_results(a, BC)
        rb = results_io.load_results(b, BC)
        common = sorted(set(ra) & set(rb))
        assert common, (a, b)
        ta = [ra[N].get("runtime") for N in common]
        tb = [rb[N].get("runtime") for N in common]
        assert all(t is not None for t in ta + tb), (a, b)
        assert ta != tb, (a, b, "identical runtimes suggest a copied record")


def test_the_extended_units_carry_their_provenance():
    """A record written by the extension says so, so a later reader can tell
    which units came from a sweep and which from the reduction."""
    tagged = 0
    for rule in EU.UNITARY:
        for rec in results_io.load_wcc_results(rule, BC).values():
            src = (rec.get("checks") or {}).get("source")
            if src:
                assert src == "extend_unitary", src
                assert rec["N"] >= 17
                tagged += 1
    assert tagged > 0


def test_a_dry_run_writes_nothing():
    before = {r: sorted(results_io.load_wcc_results(r, BC)) for r in (51, 57)}
    EU.extend([51, 57], BC, n_max=24, dry_run=True, quiet=True)
    after = {r: sorted(results_io.load_wcc_results(r, BC)) for r in (51, 57)}
    assert before == after
