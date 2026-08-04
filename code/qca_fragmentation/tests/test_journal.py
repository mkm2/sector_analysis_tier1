"""R14: the condensed WCC/SCC figure."""

import re

import pytest

from qca_fragmentation.viz import journal


def test_the_off_axis_rules_are_exactly_the_multi_attractor_rules():
    """R14's organising fact, and a theorem rather than an observation: one
    attractor per sector forces n_scc = n_wcc, hence a_att = a, hence a purely
    vertical move.  So a horizontal component can only come from a sector that
    holds several attractors."""
    from qca_fragmentation.viz import dot_sectors as D
    off = {r["rule"] for r in journal.off_axis("obc0")}
    multi = set(D.EXCEPTIONS)                    # R9 sec.7.3's twelve
    assert off == multi - {235, 249}, sorted(off ^ (multi - {235, 249}))
    assert len(off) == 10


def test_235_and_249_are_multi_attractor_yet_move_vertically():
    """Their attractor/sector ratio is a constant 2, which cancels out of a
    growth base, so both bases stay at 1 and the move has no horizontal part.
    The figure would be wrong to colour them as exceptions."""
    off = {r["rule"] for r in journal.off_axis("obc0")}
    assert 235 not in off and 249 not in off


def test_the_flow_is_one_dimensional():
    """Of the rules that move, all decrease b and none moves horizontally."""
    mv = journal.moves("obc0")
    moved = [m for m in mv if abs(m[0] - m[2]) > 1e-9 or abs(m[1] - m[3]) > 1e-9]
    n_moved = sum(m[5] for m in moved)
    vertical = sum(m[5] for m in moved if abs(m[0] - m[2]) < 1e-9)
    down = sum(m[5] for m in moved if m[3] < m[1] - 1e-9)
    horizontal = sum(m[5] for m in moved if abs(m[1] - m[3]) < 1e-9)
    assert n_moved == 230
    assert vertical == 220
    assert down == n_moved           # every move decreases b
    assert horizontal == 0           # none is purely horizontal


@pytest.mark.parametrize("style", ["overlay2", "stack"])
def test_the_two_candidates_render(tmp_path, style):
    import os
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    fn = {"overlay2": journal.style_overlay2, "stack": journal.style_stack}[style]
    out = str(tmp_path / f"{style}.pdf")
    fn("obc0", out=out, data=d)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_module_guard_is_last():
    """The recurring trap in this project: appending a figure function after the
    __main__ block makes `python -m ...` fail on a NameError halfway through."""
    src = open(journal.__file__).read()
    guard = src.index('if __name__ == "__main__"')
    assert guard > max(m.start() for m in re.finditer(r"^def ", src, re.M))
    for name in journal.STYLES:
        assert callable(journal.STYLES[name])
