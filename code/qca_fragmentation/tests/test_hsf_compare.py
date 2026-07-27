"""
Cross-check of the C150 findings against the pre-existing Julia HSF eigendata
(R8 sec.8).  Skipped cleanly when the HSF checkout is not present.
"""

import numpy as np
import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.quantum import hsf_compare as hc
from qca_fragmentation.quantum import rule150_spectra as rs

HSF = hc.find_hsf_root()
needs_hsf = pytest.mark.skipif(HSF is None, reason="HSF checkout not found")


def test_hsf_rule_6_is_wolfram_150():
    """The numbering claim the whole comparison rests on."""
    assert rules.hsf_to_wolfram(6) == 150
    assert rules.hsf_to_tuple(6) == ("I", "V", "V", "I")
    assert rules.wolfram_to_hsf(150) == 6


@needs_hsf
@pytest.mark.parametrize("N", range(8, 15))
def test_hsf_reproduces_every_c150_claim(N):
    r = hc.compare(N)
    assert r is not None, N
    assert r["sizes_match_closed_form"], N
    assert r["distinct_match"], (N, r["n_distinct"], r["distinct_closed_form"])
    assert r["dim_fix_matches"], N
    assert r["all_sectors_subset_of_universal"], N
    assert r["largest_sector_coverage"] == 1.0, N
    # the sign convention is moot because the spectrum is conjugation closed
    assert r["spectrum_conjugation_symmetric"] < 1e-12, N


@needs_hsf
def test_hsf_confirms_the_out_of_sample_point():
    """N=14: the law was written down from N<=13 and predicts 3^7 = 2187."""
    r = hc.compare(14)
    assert r["n_distinct"] == 2187 == 3 ** 7


@needs_hsf
def test_hsf_degeneracy_fraction_grows():
    fr = [hc.compare(N)["frac_states_in_degenerate_eigenspace"]
          for N in (8, 11, 14)]
    assert fr[0] < fr[1] < fr[2]
    assert fr[0] > 0.4 and fr[2] > 0.7


@pytest.mark.parametrize("N,w", [(8, 4), (10, 6)])
def test_entanglement_entropy_is_basis_dependent_on_degenerate_levels(N, w):
    """Why the HSF entropy column needs care for this rule: rotating inside a
    degenerate eigenspace changes the per-eigenstate entropy, while on a simple
    eigenvalue it does not."""
    d = rs.eigenspace_ee_spread(N, w)
    assert d["n_degenerate_clusters_tested"] > 0
    assert d["max_spread_degenerate"] > 0.01
    assert d["max_spread_nondegenerate"] < 1e-9


def test_half_chain_entropy_of_a_product_state_is_zero():
    N = 6
    v = np.zeros(1 << N)
    v[0b101010] = 1.0
    assert rs.half_chain_entropy(v, N) < 1e-12
    # a maximally entangled cut across the middle gives ln 2
    v = np.zeros(1 << N)
    v[0] = v[(1 << (N // 2)) | 1] = 1 / np.sqrt(2)
    assert abs(rs.half_chain_entropy(v, N) - np.log(2)) < 1e-12
