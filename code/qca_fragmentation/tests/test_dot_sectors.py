"""The Graphviz sector pictures ported from HSF/visualization_V3.jl and _V4.jl."""

import os

import pytest

from qca_fragmentation.graph import wcc
from qca_fragmentation.viz import dot_sectors as D


def test_the_sectors_are_the_r9_sectors():
    """The picture must partition the basis exactly as R9's WCC pass does,
    otherwise it is illustrating something other than the report."""
    for rule in (29, 157, 44, 104):
        _, sectors, _ = D.sector_graph(rule, 9, "obc0")
        res = wcc.weak_components(rule, 9, "obc0")
        assert sorted((len(s) for s in sectors), reverse=True) == res.sizes_wcc
        assert sum(len(s) for s in sectors) == 1 << 9
        flat = sorted(v for s in sectors for v in s)
        assert flat == list(range(1 << 9))          # a partition, no repeats


def test_terminal_sccs_refine_the_sectors():
    """Every terminal SCC lies inside ONE weak component -- the containment R9
    F11 counts.  A violation would mean the two decompositions disagree."""
    for rule in (29, 44, 157):
        _, sectors, rec, _, _ = D.decompositions(rule, 9, "obc0")
        owner = {v: i for i, s in enumerate(sectors) for v in s}
        for cls in rec:
            assert len({owner[v] for v in cls}) == 1, (rule, cls[:4])


def test_every_state_drains_somewhere():
    """A finite graph has no state that reaches no terminal SCC."""
    for rule in (29, 44):
        _, _, _, basin, _ = D.decompositions(rule, 9, "obc0")
        assert len(basin) == 1 << 9


def test_layout_is_finite_and_centred():
    _, sectors, adj = D.sector_graph(157, 9, "obc0")
    for s in sectors[:4]:
        pos = D.stress_layout(s, adj, seed=157)
        assert len(pos) == len(s)
        assert all(abs(x) < 1e6 and abs(y) < 1e6 for x, y in pos)
        if len(s) > 2:
            cx = sum(p[0] for p in pos) / len(pos)
            assert abs(cx) < 1e-6                    # mean-centred


@pytest.mark.parametrize("variant", ["V3", "V4"])
def test_dot_files_are_written_and_well_formed(tmp_path, variant):
    paths = D.write_dot_set(157, 9, "obc0", variant, out_dir=str(tmp_path))
    assert set(paths) == {"wcc", "scc", "joint"}
    for kind, p in paths.items():
        text = open(p).read()
        assert text.startswith("graph G {") and text.rstrip().endswith("}")
        assert "layout=neato;" in text
        # absolute coordinates: the file is meant for `neato -n2`
        assert text.count("!\"") > 0
        assert ('outputorder="edgesfirst"' in text) == (variant == "V4")
    # the wcc and joint pictures carry the whole basis; scc only the recurrent
    assert open(paths["wcc"]).read().count('pos="') == 1 << 9
    assert open(paths["joint"]).read().count('pos="') == 1 << 9
    _, _, rec, _, _ = D.decompositions(157, 9, "obc0")
    assert open(paths["scc"]).read().count('pos="') == sum(len(c) for c in rec)


def test_wcc_and_joint_share_their_geometry(tmp_path):
    """The joint picture recolours the sector picture; if the positions drifted
    the two could not be read against each other."""
    paths = D.write_dot_set(29, 9, "obc0", "V4", out_dir=str(tmp_path))

    def positions(p):
        out = {}
        for line in open(p):
            if 'pos="' in line and " -- " not in line:
                node = line.split("[")[0].strip()
                out[node] = line.split('pos="')[1].split('"')[0]
        return out

    assert positions(paths["wcc"]) == positions(paths["joint"])


def test_self_loops_are_dropped_like_the_julia_does():
    edges, _, _ = D.sector_graph(204, 8, "obc0")     # IIII: every state frozen
    assert all(u != v for u, v in edges)
    assert edges == []                               # nothing but self-loops


def test_the_exception_list_is_the_multi_attractor_rules_AT_N_MAX():
    """R9 sec.7.3's twelve, and the scope of that number.

    The comparison is made at the largest N computed, so the twelve are the
    rules with a multi-attractor sector THERE -- not the rules that ever have
    one.  Rule 37 is the difference; see the next test."""
    from qca_fragmentation.scaling import sectors, sector_figure as sf
    d = sectors.load("obc0") or sectors.build("obc0")
    sec = {p["rule"]: p for p in d["points"]}

    def pt(r):
        a, b = sec.get(r), sectors.attractor_point(r, "obc0")
        if a is None or b is None:
            return None
        return {"family": a["family"], "n_wcc": a["n_wcc"],
                "n_recurrent": b["n_recurrent"]}

    rows = sf.wcc_scc_rows(
        range(256), lambda r: sectors.load_series(r, "obc0",
                                                  sectors.UNIFORM_N_CAP), pt)
    off = sorted(r["rule"] for r in rows if r["n_scc"] != r["n_wcc"])
    assert off == sorted(D.EXCEPTIONS)


def test_six_exceptions_are_also_survivors():
    """The two groups overlap, so the union is 28 rules and not 34; a driver
    that draws both must not draw those six twice."""
    surv = set(D.EXPONENTIAL_SURVIVORS) | set(D.POLYNOMIAL_SURVIVORS)
    assert sorted(set(D.EXCEPTIONS) & surv) == [29, 44, 71, 100, 104, 233]
    assert len(set(D.EXCEPTIONS) | surv) == 28


def test_a_multi_attractor_sector_is_what_makes_a_shared_basin_possible():
    """Two attractors in one weak component is a NECESSARY condition for a
    transient state to be able to drain to more than one of them."""
    from collections import Counter
    for rule in (44, 29, 157, 20):
        _, sectors_, rec, basin, _ = D.decompositions(rule, 9, "obc0")
        owner = {v: i for i, s in enumerate(sectors_) for v in s}
        multi = max(Counter(owner[c[0]] for c in rec).values(), default=0) > 1
        shared = any(v == -1 for v in basin.values())
        assert shared <= multi, (rule, shared, multi)


def test_rule_37_is_a_thirteenth_exception_at_N_congruent_2_mod_3():
    """R9 sec.7.3/8.2.  W37 (EDDV) has one sector at every N and TWO attractors
    exactly when N = 2 (mod 3), so it is a one-sector multi-attractor rule at
    N = 8, 11, 14 and unremarkable elsewhere.  N_max = 16 = 1 (mod 3), which is
    why the descriptor comparison never sees it.  This is the test that stops
    "the twelve" being reread as a statement about the whole rule space."""
    from collections import Counter
    from qca_fragmentation.core import rules as R
    from qca_fragmentation.graph import scc as S
    t = R.wolfram_to_tuple(37)
    assert "".join(t) == "EDDV"
    for N in range(6, 15):
        _, sectors, rec, _, _ = D.decompositions(37, N, "obc0")
        assert len(sectors) == 1, (N, len(sectors))
        owner = {v: i for i, s in enumerate(sectors) for v in s}
        multi = max(Counter(owner[c[0]] for c in rec).values()) > 1
        assert multi == (N % 3 == 2), (N, len(rec))


def test_the_twelve_are_stable_across_N():
    """The complement of the rule-37 caveat: the twelve are not artefacts of the
    size they were measured at.  Every one has a multi-attractor sector at every
    N in 7..12 (71 is the one that needs N > 6)."""
    from collections import Counter
    for rule in D.EXCEPTIONS:
        for N in (7, 9, 12):
            _, sectors, rec, _, _ = D.decompositions(rule, N, "obc0")
            owner = {v: i for i, s in enumerate(sectors) for v in s}
            assert max(Counter(owner[c[0]] for c in rec).values()) > 1, (rule, N)
