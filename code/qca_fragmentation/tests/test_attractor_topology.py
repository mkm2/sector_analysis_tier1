"""R17: the terminal SCCs of the exponential-exponential classes are not cycles.

The claims worth defending are structural, so they are tested on freshly built
graphs at small N rather than against the cached JSON -- a regression in `succ`
or in Tarjan should fail here, not be papered over by a stale analytics file.
"""

import networkx as nx
import pytest

from qca_fragmentation.core.cycle import succ
from qca_fragmentation.core.rules import has_V, wolfram_to_tuple
from qca_fragmentation.graph import scc
from qca_fragmentation.scaling import attractor_topology as AT

BC = "obc0"
ALL_RULES = [r for c in AT.CLASSES for r in c["rules"]]


def test_the_classes_are_the_ones_r9_places_off_the_axis():
    """R17 is about a specific set of twelve rules. If the class list drifted
    from R9's map the report would be describing something else."""
    from qca_fragmentation.scaling import sectors
    d = sectors.load(BC) or sectors.build(BC)
    att = {x["rule"]: x for x in d["attractor_deficits"]}
    for cls in AT.CLASSES:
        for r in cls["rules"]:
            assert att[r]["a_att"] == pytest.approx(cls["a"], abs=5e-4), r
            assert att[r]["b_att"] == pytest.approx(cls["b"], abs=5e-4), r
            assert att[r]["a_att"] > 1.0 and att[r]["b_att"] > 1.0, r


def test_the_named_constants_are_the_algebraic_numbers_claimed():
    """phi^2 = phi+1, psi^3 = psi^2+1, rho^3 = rho+1, and the first class's
    constant is rho SQUARED -- the coincidence the report leans on."""
    phi, psi, rho = 1.6180339887, 1.4655712319, 1.3247179572
    assert phi ** 2 - phi - 1 == pytest.approx(0, abs=1e-9)
    assert psi ** 3 - psi ** 2 - 1 == pytest.approx(0, abs=1e-9)
    assert rho ** 3 - rho - 1 == pytest.approx(0, abs=1e-9)
    assert rho ** 2 == pytest.approx(1.754878, abs=1e-5)


@pytest.mark.parametrize("rule", ALL_RULES)
def test_no_terminal_scc_leaks(rule):
    """attractor_graph asserts this internally; here it is as a test, because
    the induced-subgraph construction is only honest if nothing leaves."""
    N = 8
    t = wolfram_to_tuple(rule)
    for c in scc.recurrent_classes(rule, N, BC, t):
        S = set(c)
        for x in c:
            assert set(succ(x, N, t, BC)) <= S


@pytest.mark.parametrize("rule", ALL_RULES)
def test_the_largest_attractor_is_not_a_cycle(rule):
    """The headline. A cycle has m = n and out-degree 1; these have neither,
    and every node carries a self-loop."""
    G = AT.largest_terminal(rule, 10, BC)
    n, m = G.number_of_nodes(), G.number_of_edges()
    assert n > 1
    assert m > n
    assert not AT.measure(G)["is_cycle"]
    assert nx.number_of_selfloops(G) == n


@pytest.mark.parametrize("rule", (28, 29, 70, 71, 157, 199))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_the_plastic_class_is_complete_digraphs_all_the_way_down(rule, N):
    """Not just the largest -- EVERY terminal SCC of these six rules is K_n with
    all n^2 edges, self-loops included."""
    t = wolfram_to_tuple(rule)
    for c in scc.recurrent_classes(rule, N, BC, t):
        G = AT.attractor_graph(rule, N, c, BC)
        assert G.number_of_edges() == len(c) ** 2, (rule, N, len(c))


def test_the_complete_case_mixes_in_one_step():
    """A uniform transition matrix has every non-Perron eigenvalue at 0, so the
    monitored walk inside these attractors is memoryless after one period."""
    d = AT.measure(AT.largest_terminal(199, 12, BC))
    assert d["is_complete_with_loops"]
    assert d["slem"] == pytest.approx(0.0, abs=1e-9)
    assert d["diameter"] == 1


@pytest.mark.parametrize("rule", (108, 201, 73, 109))
def test_diameter_two_while_the_attractor_grows(rule):
    """The size grows like phi^N; the diameter does not grow at all."""
    diams, sizes = [], []
    for N in (8, 10, 12):
        d = AT.measure(AT.largest_terminal(rule, N, BC))
        diams.append(d["diameter"])
        sizes.append(d["n"])
    assert diams == [2, 2, 2], (rule, diams)
    assert sizes[-1] > 4 * sizes[0]


def test_the_spectral_gap_does_not_close():
    """|lambda_2| is flat in n for the phi class -- an O(1) mixing time inside a
    Theta(phi^N) attractor. Bounded away from 1 is the claim; it sits near 0.3."""
    vals = [AT.measure(AT.largest_terminal(201, N, BC))["slem"]
            for N in (8, 10, 12)]
    assert all(0.2 < v < 0.4 for v in vals), vals
    assert max(vals) - min(vals) < 0.02


def test_every_dissipative_child_inherits_its_parents_sectors_whole():
    """R17's main result, and the one most worth pinning: for all eight children
    the terminal SCCs are parent Krylov sectors with the SAME states AND the same
    induced edges. Set equality alone would not be enough -- a reset that fired
    would merge branches and change the edges while keeping the vertex set."""
    ident = AT.parent_identity(BC, (8, 10))
    assert {p["child"] for p in ident} == {73, 109, 28, 29, 70, 71, 157, 199}
    for pi in ident:
        for row in pi["rows"]:
            assert row["child_sets_are_parent_sectors"], pi["child"]
            assert row["graph_identical"], pi["child"]
            assert row["child_attractors"] < row["parent_sectors"]


def test_the_plastic_class_selects_the_parents_complete_sectors():
    """What distinguishes class 4 from class 2 is WHICH sectors the reset spares.
    Exactly the complete ones for the pure-D children, a strict subset for the
    ones carrying an E -- a distinction the report states rather than smooths."""
    by_child = {p["child"]: p for p in AT.parent_identity(BC, (8, 10))}
    for child in (28, 29, 70, 71, 157, 199):
        for row in by_child[child]["rows"]:
            assert row["child_inside_complete_sectors"], child
    for child in (28, 70):                       # a single D reset
        assert all(r["child_is_all_complete_sectors"]
                   for r in by_child[child]["rows"]), child
    for child in (29, 71, 157, 199):             # carries an E
        assert not all(r["child_is_all_complete_sectors"]
                       for r in by_child[child]["rows"]), child
    # and the psi class does NOT select complete sectors -- it keeps big ones
    for child in (73, 109):
        assert not any(r["child_inside_complete_sectors"]
                       for r in by_child[child]["rows"]), child


def test_the_complete_attractor_size_is_the_closed_form_r17_derives():
    """b_att = 2^(1/3) is a fitted constant in R9. The size is 2^floor((N+1)/3)
    exactly, which derives it."""
    for rule in (28, 199):
        for N in range(6, 15):
            t = wolfram_to_tuple(rule)
            d = max(len(c) for c in scc.recurrent_classes(rule, N, BC, t))
            assert d == 2 ** ((N + 1) // 3), (rule, N, d)


def test_the_vfree_contrast_is_by_definition_not_by_luck():
    """Every V-free rule has a deterministic map, so out-degree is 1 and a
    terminal SCC can only be a cycle. This checks the premise rather than
    assuming it, and records how short those cycles actually are."""
    v = AT.vfree_control(10, BC)
    assert v["n_rules"] == 81
    assert v["all_deterministic"]
    assert v["fixed_point_only"] > 0.8 * v["n_rules"]


def test_measure_survives_a_single_fixed_point():
    """The V-free controls hand `measure` a one-node graph with one self-loop.
    Several networkx routines are undefined on that input, and an exception
    here would take out the whole census."""
    G = nx.DiGraph()
    G.add_edge(0, 0)
    d = AT.measure(G)
    assert d["n"] == 1 and d["m"] == 1
    assert d["is_cycle"] and d["is_complete_with_loops"]
    assert d["transitivity"] == 0.0          # no non-loop edges to close
    assert d["diameter"] == 0                # no ordered pair with u != v


def test_clustering_is_reported_against_its_null():
    """R17 declines to call clustering a finding. The measure must therefore
    always ship with the density-matched null it has to be read against,
    otherwise a reader sees 0.92 and concludes 'highly clustered'."""
    d = AT.measure(AT.largest_terminal(201, 10, BC))
    assert "transitivity_er_null" in d
    assert d["transitivity"] - d["transitivity_er_null"] < 0.15
