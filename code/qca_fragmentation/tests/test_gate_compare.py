"""
R11: the two gates on one rule space.

The tests here pin the refinement theorem, which is what makes the comparison
meaningful at all, and the corollary that the V-free rules are the same circuit
under both gates -- a cross-implementation check between two engines that share
no code below core/cycle.py::_compile.
"""

import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import wcc
from qca_fragmentation.permutation import compare, functional as fnl, xca


@pytest.mark.parametrize("N", [7, 8])
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_every_X_edge_is_a_Hadamard_edge(N, bc):
    """R11 Thm 1: flipping the target bit is one of the Hadamard outcomes."""
    for rule in range(256):
        succ = wcc.make_succ(rule, N, bc)
        T = xca.step_table(rule, N, bc)
        for x in range(1 << N):
            assert T[x] in succ(x), (rule, N, bc, x)


@pytest.mark.parametrize("N", [8])
def test_X_sectors_refine_hadamard_sectors(N):
    rc = compare.refinement_check(N, "obc0")
    assert rc["edge_failures"] == []
    assert rc["refine_failures"] == []


@pytest.mark.parametrize("N", [8, 9])
def test_the_two_inequalities(N):
    """n_wcc can only go up and D_max only down when the gate becomes X."""
    for rule in range(256):
        h = wcc.weak_components(rule, N, "obc0")
        x = fnl.analyze(rule, N, "obc0")
        assert x["n_wcc"] >= h.n_wcc, (rule, N, h.n_wcc, x["n_wcc"])
        assert x["d_max_wcc"] <= h.d_max_wcc, (rule, N)


@pytest.mark.parametrize("N", [8, 9])
def test_v_free_rules_are_the_same_circuit(N):
    """81 rules never invoke the gate, so both engines must agree exactly."""
    shared = [r for r in range(256) if not compare.has_V(r)]
    assert len(shared) == 81
    for rule in shared:
        h = wcc.weak_components(rule, N, "obc0")
        x = fnl.analyze(rule, N, "obc0")
        assert sorted(h.sizes_wcc) == sorted(x["sizes_wcc"]), (rule, N)


def test_rule_51_is_the_extreme_case():
    """One sector of 2^N under H; 2^(N-1) two-cycles under X."""
    for N in (8, 9):
        h = wcc.weak_components(51, N, "obc0")
        x = fnl.analyze(51, N, "obc0")
        assert h.n_wcc == 1 and h.d_max_wcc == (1 << N)
        assert x["n_wcc"] == (1 << (N - 1)) and x["d_max_wcc"] == 2


def test_the_common_window_is_actually_common():
    """Every comparison must use the Hadamard window, or the bases are not
    comparable (R11 sec.1)."""
    assert compare.COMMON_N_CAP == 16
    p = compare.x_point(156, "obc0")
    assert p is not None and p["N_max"] <= compare.COMMON_N_CAP


def test_no_rule_loses_its_sector_growth_class():
    """The class cross-tabulation must be upper-triangular."""
    d = compare.load("obc0") or compare.build("obc0")
    order = {"constant": 0, "polynomial": 1, "exponential": 2}
    for r in d["rows"]:
        h, x = r["cls_a_h"], r["cls_a_x"]
        if h in order and x in order:
            assert order[x] >= order[h], (r["rule"], h, x)


# --- HSF reading (R11 sec.7) --------------------------------------------------

def test_hsf_phase_labels():
    assert compare.hsf_phase(1.0, 2.0) == "unfragmented"
    assert compare.hsf_phase(1.6, 2.0) == "weak"
    assert compare.hsf_phase(1.618, 1.3195) == "strong"
    assert compare.hsf_phase(2.0, 1.0) == "shattered"
    assert compare.hsf_phase(1.0, 1.9) == "degenerate"     # forbidden corner
    assert compare.hsf_phase(None, 1.5) == "irregular"


def test_the_gate_never_reduces_fragmentation():
    """Refinement forbids the lower-left of the phase cross-tabulation."""
    d = compare.load("obc0") or compare.build("obc0")
    order = {"unfragmented": 0, "weak": 1, "strong": 2, "shattered": 3}
    for r in d["phases"]["per_rule"]:
        if r["h"] in order and r["x"] in order:
            assert order[r["x"]] >= order[r["h"]], r


@pytest.mark.parametrize("rule", [156, 198])
def test_156_sectors_factorise_into_equal_orbits(rule):
    """R11 sec.7.2: dim(H sector) = (#X orbits) x (orbit length), exactly."""
    for N in (10, 12):
        sp = compare.sector_orbit_split(rule, N, "obc0")
        assert sp["unequal_sectors"] == 0, (rule, N)
        for dim, n, L, _ in sp["rows"]:
            assert dim == n * L, (rule, N, dim, n, L)


def test_the_reversible_rules_are_the_eca_of_the_same_number():
    """R11 sec.7.5: for the 16 I/V rules the local update is
    x_i -> x_i XOR f(l, r), and that map is the ECA of the same Wolfram
    number -- which is what identifies X54 with the Rule 54 automaton."""
    for rule in rules.UNITARY_RULES:
        t = rules.wolfram_to_tuple(rule)
        f = [1 if s == "V" else 0 for s in t]
        n = 0
        for i, (l, c, r) in enumerate([(1, 1, 1), (1, 1, 0), (1, 0, 1),
                                       (1, 0, 0), (0, 1, 1), (0, 1, 0),
                                       (0, 0, 1), (0, 0, 0)]):
            n |= (c ^ f[2 * l + r]) << (7 - i)
        assert n == rule, (rule, n)
    # and rule 54 is the OR one, x_i -> x_i XOR (x_{i-1} OR x_{i+1})
    assert "".join(rules.wolfram_to_tuple(54)) == "IVVV"


# --- WCC against SCC: the two new views (R9 F11/F12, R10 X8/X9) --------------

def _h_rows():
    from qca_fragmentation.scaling import sectors, sector_figure as sf
    d = sectors.load("obc0") or sectors.build("obc0")
    sec = {p["rule"]: p for p in d["points"]}

    def pt(r):
        a, b = sec.get(r), sectors.attractor_point(r, "obc0")
        if a is None or b is None:
            return None
        return {"family": a["family"], "n_wcc": a["n_wcc"],
                "n_recurrent": b["n_recurrent"]}

    return sf.wcc_scc_rows(
        range(256), lambda r: sectors.load_series(r, "obc0",
                                                  sectors.UNIFORM_N_CAP), pt)


def _x_rows():
    from qca_fragmentation.permutation import analysis
    from qca_fragmentation.scaling import sector_figure as sf
    d = analysis.load("obc0") or analysis.build("obc0")
    pts = {x["rule"]: x for x in d["points"]}
    return sf.wcc_scc_rows(
        range(256),
        lambda r: analysis.load_series(r, "obc0", analysis.UNIFORM_N_CAP),
        lambda r: pts.get(r))


def test_every_weak_component_contains_a_terminal_scc():
    """R9 F11.  n_scc >= n_wcc is exact, not fitted: a finite digraph's weak
    component always contains at least one terminal SCC."""
    for rows, gate in ((_h_rows(), "H"), (_x_rows(), "X")):
        bad = [(r["rule"], r["n_wcc"], r["n_scc"]) for r in rows
               if r["n_scc"] < r["n_wcc"]]
        assert bad == [], (gate, bad[:6])


def test_the_x_gate_puts_every_rule_on_the_diagonal():
    """R10 X8/X9.  A functional graph has one cycle per weak component, so the
    sector count IS the attractor count -- in the raw numbers and in the bases."""
    rows = _x_rows()
    assert len(rows) == 256
    assert [r["rule"] for r in rows if r["n_scc"] != r["n_wcc"]] == []
    off = [r["rule"] for r in rows
           if r["base_wcc"] is not None and r["base_scc"] is not None
           and abs(r["base_scc"] - r["base_wcc"]) > 1e-12]
    assert off == []


def test_the_hadamard_multi_attractor_sectors_are_exactly_twelve():
    """R9 F11.  Superposition lets one sector hold several attractors; under the
    Hadamard exactly twelve rules do, and all twelve carry a V and a reset."""
    from qca_fragmentation.core import rules as R
    rows = _h_rows()
    off = sorted(r["rule"] for r in rows if r["n_scc"] != r["n_wcc"])
    assert off == [29, 36, 44, 71, 100, 104, 203, 217, 219, 233, 235, 249]
    for rule in off:
        t = R.wolfram_to_tuple(rule)
        assert "V" in t and any(c in "DE" for c in t), rule
    # 203 and 217: one sector, 277 attractors inside it
    for rule in (203, 217):
        r = next(x for x in rows if x["rule"] == rule)
        assert (r["n_wcc"], r["n_scc"]) == (1, 277)


def test_a_constant_attractor_ratio_cancels_out_of_the_base():
    """Why F12 exists next to F11: 235 and 249 carry two attractors per sector at
    every N, so they are off the count diagonal and ON the base diagonal."""
    rows = {r["rule"]: r for r in _h_rows()}
    for rule in (235, 249):
        r = rows[rule]
        assert r["n_scc"] == 2 * r["n_wcc"]
        assert r["base_scc"] == pytest.approx(r["base_wcc"])
