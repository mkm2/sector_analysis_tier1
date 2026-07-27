"""The C150 discussion deck builds, and its numbers come from the analytics."""

import os

import pytest

pptx = pytest.importorskip("pptx", reason="python-pptx not installed")

from qca_fragmentation.scaling import c150_slides as sl


def test_deck_builds_and_is_well_formed(tmp_path):
    out = str(tmp_path / "deck.pptx")
    sl.build(out)
    assert os.path.exists(out)

    prs = pptx.Presentation(out)
    assert len(prs.slides) >= 18
    assert round(prs.slide_width / prs.slide_height, 2) == 1.78      # 16:9

    # every slide carries a title and speaker notes
    for i, s in enumerate(prs.slides, 1):
        assert s.shapes.title is not None and s.shapes.title.text.strip(), i
        assert s.has_notes_slide, i
        assert s.notes_slide.notes_text_frame.text.strip(), i

    # nothing hangs off the slide
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.top is None or sh.height is None:
                continue
            assert sh.top + sh.height <= prs.slide_height + 10000, (i, sh.name)
            assert sh.left + sh.width <= prs.slide_width + 10000, (i, sh.name)

    # the figures and at least the three data tables made it in
    pics = sum(1 for s in prs.slides for sh in s.shapes if sh.shape_type == 13)
    tbls = sum(1 for s in prs.slides for sh in s.shapes if sh.has_table)
    assert pics >= 4 and tbls >= 3


def test_frontier_table_matches_the_closed_form(tmp_path):
    """The deck must not drift from the data it claims to report."""
    from qca_fragmentation import c150

    out = str(tmp_path / "deck.pptx")
    sl.build(out)
    prs = pptx.Presentation(out)
    tbl = next(sh.table for s in prs.slides for sh in s.shapes
               if sh.has_table and sh.table.cell(0, 0).text == "N")
    for i in range(1, len(tbl.rows)):
        N = int(tbl.cell(i, 0).text)
        assert tbl.cell(i, 2).text == str(len(c150.sector_sizes_obc0(N)))
        assert tbl.cell(i, 3).text == f"{c150.d_max_obc0(N):,}"
