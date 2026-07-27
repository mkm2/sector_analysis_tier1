"""
C150 entanglement: the kinematic ceiling set by the sector, and where it stops
being enough (R8 sec.9, answering QCA_Circuits Fig. 3a).
"""

from math import comb, log

import numpy as np
import pytest

from qca_fragmentation import c150
from qca_fragmentation.quantum import rule150_entanglement as re
from qca_fragmentation.quantum import rule150_spectra as rs


# --- the three Fig. 3a states sit in three very different sectors ------------

@pytest.mark.parametrize("N", [10, 12, 14, 16, 20])
def test_fig3_state_sectors(N):
    s = re.fig3_states(N)
    w = {k: c150.wall_number(v, N, "obc0") for k, v in s.items()}
    assert w["single"] == 2                      # two walls, always
    assert w["neel"] == (N if N % 2 == 0 else N - 1)
    assert abs(w["pair"] - N / 2) <= 2
    # the premise "sectors are exponentially large" fails at both ends
    assert comb(N + 1, w["single"]) == N * (N + 1) // 2      # polynomial
    if N % 2 == 0:
        assert comb(N + 1, w["neel"]) == N + 1               # linear!
    assert comb(N + 1, w["pair"]) > 2 ** (N / 2)             # exponential


def test_neel_is_a_one_hole_state():
    """Every bond but one carries a wall, so the hard-core walls are jammed."""
    N = 12
    x = re.fig3_states(N)["neel"]
    b = c150.wall_string(x, N, "obc0")
    assert bin(b).count("1") == N          # N of the N+1 bonds occupied
    assert (N + 1) - bin(b).count("1") == 1


# --- the ceiling is exact ----------------------------------------------------

@pytest.mark.parametrize("N", range(6, 14))
def test_schmidt_bound_equals_the_actual_compressed_dimension(N):
    """The closed form of eq. (entbound) IS the compressed Schmidt matrix size,
    including the lower limit that matters at high filling."""
    for w in range(0, N + 2, 2):
        if comb(N + 1, w) < 2:
            continue
        states = rs.sector_states(N, w)
        _, _, nb, na = rs.schmidt_index(N, states)
        assert re.schmidt_bound(N, w) == min(comb(N + 1, w), nb, na), (N, w)


def test_dropping_the_lower_limit_would_overcount_at_high_filling():
    """The N=12 Neel shell: with the lower limit 7 patterns, without it 64."""
    N, w, c = 12, 12, 6
    naive = sum(comb(c, k) for k in range(min(w, c) + 1))
    assert naive == 64
    assert re.schmidt_bound(N, w) == 7
    assert re.kinematic_ceiling(N, w) < log(naive)


@pytest.mark.parametrize("N,w", [(10, 4), (12, 6), (12, 2), (14, 14)])
def test_entropy_never_exceeds_the_ceiling(N, w):
    U, states = rs.sector_unitary(N, w, check=False)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(len(states))
    tr = re.entropy_trace(N, w, v, t_max=25, U=U, states=states)
    assert tr.max() <= re.kinematic_ceiling(N, w) + 1e-9


# --- area law vs volume law --------------------------------------------------

def test_single_excitation_and_neel_are_area_law():
    """S_plateau stops growing with N; the ceiling still grows like ln N."""
    for name in ("single", "neel"):
        vals = []
        for N in (14, 22, 30):
            x = re.fig3_states(N)[name]
            w, tr = re.basis_trace(N, x, t_max=160)
            vals.append(re.plateau(tr, burn=100))
        assert max(vals) - min(vals) < 0.15, (name, vals)   # flat in N
        assert max(vals) < 1.5


def test_pair_state_grows_and_its_ceiling_is_the_volume_law_value():
    vals = []
    for N in (10, 12, 14):
        x = re.fig3_states(N)["pair"]
        w, tr = re.basis_trace(N, x, t_max=160)
        vals.append(re.plateau(tr, burn=100))
    assert vals[0] < vals[1] < vals[2]                      # grows with N
    assert vals[-1] - vals[0] > 0.5
    # at half filling the sector imposes no constraint at all
    N = 20
    w = 10
    assert re.kinematic_ceiling(N, w) == pytest.approx(re.volume_law_value(N))


# --- where the sector is NOT enough ------------------------------------------

def test_product_states_of_one_shell_agree_but_dark_states_do_not():
    N, w = 12, 6
    U, states = rs.sector_unitary(N, w, check=False)
    rng = np.random.default_rng(1)
    prods = []
    for k in rng.choice(len(states), 4, replace=False):
        v = np.zeros(len(states))
        v[k] = 1.0
        prods.append(re.plateau(re.entropy_trace(N, w, v, t_max=160,
                                                 U=U, states=states), burn=100))
    assert max(prods) - min(prods) < 0.4, prods      # product states agree

    d = re.dark_state_entropy(N, w)
    assert d["dim_fix"] == d["dim_fix_closed_form"] == comb(6, 3) == 20
    for row in d["dark"]:
        assert row["s_time_std"] < 1e-12             # constant in time
        assert row["s_mean"] > max(prods)            # and higher than the product
