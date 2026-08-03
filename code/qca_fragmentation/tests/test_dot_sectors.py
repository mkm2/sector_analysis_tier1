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
