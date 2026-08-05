"""The hand-over notebook must stay valid and runnable."""

import json
import os

import pytest

NB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                  "notebooks", "wcc_scc_stack_figure.ipynb")


def _nb():
    if not os.path.exists(NB):
        pytest.skip("notebook not present")
    return json.load(open(NB))


def test_notebook_is_valid_and_every_code_cell_compiles():
    nb = _nb()
    assert nb["nbformat"] >= 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 8
    for i, c in enumerate(code, 1):
        src = "".join(c["source"])
        compile(src, f"<nb cell {i}>", "exec")      # syntax only, no execution


def test_notebook_carries_no_stored_output():
    """Outputs in a tracked notebook make every rerun a diff."""
    nb = _nb()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert not c.get("outputs"), "clear outputs before committing"
            assert c.get("execution_count") in (None, 0)


def test_notebook_documents_the_extraction_ladder():
    """Its stated purpose is documenting how a base is obtained, so the branches
    that actually exist in sectors.series_descriptor must be named in it."""
    nb = _nb()
    md = "\n".join("".join(c["source"]) for c in nb["cells"]
                   if c["cell_type"] == "markdown")
    for phrase in ["Analytic override", "Irregularity",
                   "Volume-fraction discriminator", "Saturation",
                   "BIC model selection", "Rate correction",
                   "Exact integer recurrence", "Parity split",
                   "band guard", "re-derivation", "Clamp"]:
        assert phrase in md, phrase
    # and the caveats that stop a base being misquoted
    assert "monitored-only" in md and "obc0" in md
