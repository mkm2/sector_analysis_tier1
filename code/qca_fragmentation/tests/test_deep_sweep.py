"""Unit selection for the memory-guarded deep sweep."""
from qca_fragmentation import deep_sweep as ds


def test_ergodic_rules_are_dropped_at_every_larger_N(monkeypatch):
    """An ergodic rule STOPS, so its newest record is the one that flagged it.

    Checking only N-1 made rule 54 -- ergodic at N=6 and untouched since --
    look like fresh work at N=19, and the sweep started a 2^19 run for it.
    """
    recs = {6: {"ergodic_flag": True, "runtime": 0.02}}
    monkeypatch.setattr(ds.results_io, "load_results", lambda r, bc: recs)
    monkeypatch.setattr(ds.results_io, "has_unit", lambda r, bc, N: False)
    assert ds._units(19, ["pbc"], [54]) == []
    assert ds._units(7, ["pbc"], [54]) == []


def test_cost_comes_from_the_largest_cached_size():
    """Ranking is cheapest-first, so the predictor must be the newest record."""
    recs = {14: {"runtime": 1.0}, 15: {"runtime": 400.0}}
    import qca_fragmentation.results_io as rio
    orig_load, orig_has = rio.load_results, rio.has_unit
    try:
        rio.load_results = lambda r, bc: recs
        rio.has_unit = lambda r, bc, N: False
        assert ds._units(16, ["pbc"], [55]) == [(400.0, 55, "pbc")]
    finally:
        rio.load_results, rio.has_unit = orig_load, orig_has


def test_every_tier_table_entry_has_three_tiers():
    """sweep() zips buckets against the config, so a short entry silently
    drops the huge tier instead of raising."""
    for N, cfg in ds.TIERS.items():
        assert len(cfg) == 3, N
        for workers, mem_gb, timeout in cfg:
            assert workers >= 1 and mem_gb >= 3 and timeout > 0
    assert len(ds.DEFAULT_TIERS) == 3
