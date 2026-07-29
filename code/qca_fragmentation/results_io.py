"""
On-disk formats and the checkpoint/manifest audit trail (PLAN.md sec.3, sec.4).

Tier 1a results are append-only JSON lines, one object per (rule, N, bc), stored
in  results/{rule}_{bc}.jsonl.  A unit already present (same engine_version) is
skipped by the sweep; this makes overnight runs idempotent and resumable.

Every completed unit is also logged to  checkpoints/manifest.jsonl  with a
timestamp and wall-time (the audit trail cited by the reports).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, Optional, Tuple

from collections import Counter

from . import ENGINE_VERSION

# Store at most this many explicit sizes; longer multisets are also summarised as
# a {size: count} histogram (which reconstructs the full multiset exactly, and is
# tiny even for rule 204's 2^N singletons).
_SIZES_CAP = 2048

# repo root = two levels up from this file's package dir (code/qca_fragmentation)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
MANIFEST = os.path.join(CHECKPOINTS_DIR, "manifest.jsonl")

# Tier 1d (pair graph) results live in a SEPARATE store keyed on the same units,
# so the Tier-1a append-only records stay valid and are never force-recomputed.
TIER1D_VERSION = "1d.1"
PAIR_RESULTS_DIR = os.path.join(REPO_ROOT, "results_tier1d")
PAIR_FIELDS = [
    "rule", "bc", "N", "km", "n_recurrent_states", "cesaro_rank",
    "pair_rec_size", "pair_offdiag", "pair_diag_extra", "fix_upper", "certified",
    "bounded_only", "n_pair_nodes", "n_strong", "n_weak",
    "weak_grades_coherence", "d_values_on_coherence", "runtime",
    "tier1d_version",
]

# Schema field order (context Tier 1 sec.7).
FIELDS = [
    "rule", "bc", "N", "n_scc", "n_recurrent", "sizes_recurrent", "sizes_scc",
    "size_hist", "sizes_truncated", "sizes_basins", "shared_basin_size",
    "transient_depth", "n_transient_scc", "ergodic_flag", "ergodic_bound",
    "attractor_types", "d_max_quantum", "runtime", "engine_version",
]


def _histogram(sizes):
    """Compact {size: count} histogram (JSON keys are strings)."""
    return {str(s): c for s, c in sorted(Counter(sizes).items(), reverse=True)}


def sizes_from_record(rec, key="sizes_recurrent"):
    """Reconstruct the full sorted size multiset from a record, using the
    histogram when the explicit list was truncated."""
    if rec.get("sizes_truncated") and rec.get("size_hist"):
        out = []
        for s, c in rec["size_hist"].items():
            out.extend([int(s)] * c)
        out.sort(reverse=True)
        return out
    return rec.get(key) or []


def results_path(rule: int, bc: str) -> str:
    return os.path.join(RESULTS_DIR, f"{rule}_{bc}.jsonl")


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def load_results(rule: int, bc: str) -> Dict[int, dict]:
    """Return {N: record} for the units already computed for (rule, bc)."""
    path = results_path(rule, bc)
    out: Dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["N"]] = rec
    return out


def has_unit(rule: int, bc: str, N: int, *, engine_version: str = ENGINE_VERSION) -> bool:
    """True iff (rule, N, bc) already computed with the current engine_version."""
    rec = load_results(rule, bc).get(N)
    return rec is not None and rec.get("engine_version") == engine_version


def record_from_graph_result(res, runtime: float) -> dict:
    """Build a schema record from a graph.scc.GraphResult."""
    rec = {
        "rule": res.rule,
        "bc": res.bc,
        "N": res.N,
        "ergodic_flag": bool(res.ergodic),
        "ergodic_bound": res.ergodic_bound,
        "attractor_types": None,   # filled by Tier 1b
        "d_max_quantum": None,     # filled by Tier 1b
        "runtime": runtime,
        "engine_version": ENGINE_VERSION,
    }
    if res.ergodic:
        rec.update({
            "n_scc": None, "n_recurrent": None, "sizes_recurrent": None,
            "sizes_scc": None, "size_hist": None, "sizes_truncated": False,
            "sizes_basins": None, "shared_basin_size": None,
            "transient_depth": None, "n_transient_scc": None,
        })
    else:
        truncated = len(res.sizes_recurrent) > _SIZES_CAP
        rec.update({
            "n_scc": res.n_scc,
            "n_recurrent": res.n_recurrent,
            "sizes_recurrent": res.sizes_recurrent[:_SIZES_CAP],
            "sizes_scc": res.sizes_scc[:_SIZES_CAP],
            "size_hist": _histogram(res.sizes_recurrent),
            "sizes_truncated": truncated,
            "sizes_basins": res.sizes_basins[:_SIZES_CAP],
            "shared_basin_size": res.shared_basin_size,
            "transient_depth": res.transient_depth,
            "n_transient_scc": res.n_transient_scc,
        })
    return rec


def append_result(rec: dict) -> None:
    """Append one record to results/{rule}_{bc}.jsonl (ordered fields)."""
    _ensure_dirs()
    ordered = {k: rec.get(k) for k in FIELDS}
    with open(results_path(rec["rule"], rec["bc"]), "a") as f:
        f.write(json.dumps(ordered) + "\n")


# --- Tier 1e: the WCC / sector store -----------------------------------------
# Kept in its OWN store, exactly as Tier 1d is, for two reasons the task spells
# out: existing 1a.1/1b/1d records must stay valid and must never be
# recomputed, and the WCC pass deliberately runs over a WIDER N range than the
# Tarjan sweep (union-find is far cheaper).  Bumping the shared ENGINE_VERSION
# to "1e.1" instead would make has_unit() false for all ~6000 Tier-1a records
# and the next sweep would recompute the lot, which is precisely what the task
# forbids -- so the version bump lives here, on the new tier, where it belongs.
TIER1E_VERSION = "1e.1"
WCC_RESULTS_DIR = os.path.join(REPO_ROOT, "results_tier1e")
WCC_FIELDS = [
    "rule", "bc", "N",
    "n_wcc", "sizes_wcc", "size_hist_wcc", "wcc_truncated", "d_max_wcc",
    "n_frozen", "d_max_ratio",
    "ergodic_flag", "ergodic_bound", "aborted",
    # derived, from the paired Tier-1a record (None when it is absent)
    "transient_fraction", "att_per_sector", "basin_sum_check",
    "n_recurrent", "n_scc",
    "checks", "runtime", "tier1e_version",
]


# Independent basin recomputation (Tier 1e addendum).  The Tier-1a schema caps
# sizes_basins at _SIZES_CAP and never stored a basin histogram, so for units
# whose basin list was truncated the sum rule is UNCHECKABLE from the archive --
# the data is genuinely gone, not merely compressed.  Rather than assert the
# check away we recompute those units independently and store the result
# alongside, WITH the histogram, so the check becomes exact and stays exact.
BASIN_VERSION = "1e-basins.1"
BASIN_RESULTS_DIR = os.path.join(REPO_ROOT, "results_basins")
BASIN_FIELDS = [
    "rule", "bc", "N", "n_scc", "n_recurrent",
    "sizes_basins", "basin_hist", "basin_truncated", "shared_basin_size",
    "sizes_recurrent_hist", "transient_depth", "n_transient_scc",
    "basin_sum_check", "transient_fraction",
    "agrees_with_tier1a", "runtime", "basins_version",
]


def basin_results_path(rule: int, bc: str) -> str:
    return os.path.join(BASIN_RESULTS_DIR, f"{rule}_{bc}.jsonl")


def load_basin_results(rule: int, bc: str) -> Dict[int, dict]:
    path = basin_results_path(rule, bc)
    out: Dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["N"]] = rec
    return out


def has_basin_unit(rule: int, bc: str, N: int,
                   *, basins_version: str = BASIN_VERSION) -> bool:
    rec = load_basin_results(rule, bc).get(N)
    return rec is not None and rec.get("basins_version") == basins_version


def append_basin_result(rec: dict) -> None:
    os.makedirs(BASIN_RESULTS_DIR, exist_ok=True)
    ordered = {k: rec.get(k) for k in BASIN_FIELDS}
    with open(basin_results_path(rec["rule"], rec["bc"]), "a") as f:
        f.write(json.dumps(ordered) + "\n")


def basins_from_record(rec) -> list:
    """Full basin-size multiset, from the histogram when truncated."""
    if rec.get("basin_truncated") and rec.get("basin_hist"):
        out = []
        for s, c in rec["basin_hist"].items():
            out.extend([int(s)] * c)
        out.sort(reverse=True)
        return out
    return rec.get("sizes_basins") or []


def wcc_results_path(rule: int, bc: str) -> str:
    return os.path.join(WCC_RESULTS_DIR, f"{rule}_{bc}.jsonl")


def load_wcc_results(rule: int, bc: str) -> Dict[int, dict]:
    path = wcc_results_path(rule, bc)
    out: Dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # tolerate a partial final line (live append)
            out[rec["N"]] = rec
    return out


def has_wcc_unit(rule: int, bc: str, N: int,
                 *, tier1e_version: str = TIER1E_VERSION) -> bool:
    rec = load_wcc_results(rule, bc).get(N)
    return rec is not None and rec.get("tier1e_version") == tier1e_version


def sizes_from_wcc_record(rec) -> list:
    """Full sorted sector-size multiset, from the histogram when truncated."""
    if rec.get("wcc_truncated") and rec.get("size_hist_wcc"):
        out = []
        for s, c in rec["size_hist_wcc"].items():
            out.extend([int(s)] * c)
        out.sort(reverse=True)
        return out
    return rec.get("sizes_wcc") or []


def _basin_sum_check(scc_rec: dict, N: int, rule: Optional[int] = None,
                     bc: Optional[str] = None):
    """
    sum(sizes_basins) + shared_basin_size == 2^N.

    Returns True/False, or None when the check is impossible.  It is impossible
    from the Tier-1a archive alone whenever sizes_basins hit the _SIZES_CAP,
    because no basin histogram was stored -- so we first consult the independent
    recomputation in results_basins/ (see basin_recompute.py), which does store
    one.  Only if that is absent too do we return None.
    """
    if rule is not None and bc is not None:
        rb = load_basin_results(rule, bc).get(N)
        if rb is not None and rb.get("basin_sum_check") is not None:
            return rb["basin_sum_check"]
    basins = scc_rec.get("sizes_basins")
    if basins is None:
        return None
    if len(basins) >= _SIZES_CAP:
        return None
    return sum(basins) + (scc_rec.get("shared_basin_size") or 0) == (1 << N)


def record_from_wcc_result(res, scc_rec: Optional[dict] = None,
                           checks: Optional[dict] = None) -> dict:
    """Build a Tier-1e record; derived fields need the paired Tier-1a record."""
    total = 1 << res.N
    truncated = len(res.sizes_wcc) > _SIZES_CAP
    rec = {
        "rule": res.rule, "bc": res.bc, "N": res.N,
        "n_wcc": res.n_wcc,
        "sizes_wcc": res.sizes_wcc[:_SIZES_CAP],
        "size_hist_wcc": _histogram(res.sizes_wcc),
        "wcc_truncated": truncated,
        "d_max_wcc": res.d_max_wcc,
        "n_frozen": res.n_frozen,
        "d_max_ratio": (res.d_max_wcc / total) if res.d_max_wcc else None,
        "ergodic_flag": bool(res.ergodic),
        "ergodic_bound": res.ergodic_bound,
        "aborted": bool(res.aborted),
        "transient_fraction": None,
        "att_per_sector": None,
        "basin_sum_check": None,
        "n_recurrent": None,
        "n_scc": None,
        "checks": checks or {},
        "runtime": res.runtime,
        "tier1e_version": TIER1E_VERSION,
    }
    if scc_rec and not scc_rec.get("ergodic_flag"):
        n_rec = scc_rec.get("n_recurrent")
        rec["n_recurrent"] = n_rec
        rec["n_scc"] = scc_rec.get("n_scc")
        sizes_rec = sizes_from_record(scc_rec, "sizes_recurrent")
        if sizes_rec:
            rec["transient_fraction"] = 1.0 - sum(sizes_rec) / total
        if n_rec and res.n_wcc:
            rec["att_per_sector"] = n_rec / res.n_wcc
        rec["basin_sum_check"] = _basin_sum_check(scc_rec, res.N,
                                                  res.rule, res.bc)
    return rec


def append_wcc_result(rec: dict) -> None:
    os.makedirs(WCC_RESULTS_DIR, exist_ok=True)
    ordered = {k: rec.get(k) for k in WCC_FIELDS}
    with open(wcc_results_path(rec["rule"], rec["bc"]), "a") as f:
        f.write(json.dumps(ordered) + "\n")


def pair_results_path(rule: int, bc: str) -> str:
    return os.path.join(PAIR_RESULTS_DIR, f"{rule}_{bc}.jsonl")


def load_pair_results(rule: int, bc: str) -> Dict[int, dict]:
    path = pair_results_path(rule, bc)
    out: Dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # tolerate a partial final line (live append)
            out[rec["N"]] = rec
    return out


def has_pair_unit(rule: int, bc: str, N: int,
                  *, tier1d_version: str = TIER1D_VERSION) -> bool:
    rec = load_pair_results(rule, bc).get(N)
    return rec is not None and rec.get("tier1d_version") == tier1d_version


def append_pair_result(rec: dict) -> None:
    os.makedirs(PAIR_RESULTS_DIR, exist_ok=True)
    ordered = {k: rec.get(k) for k in PAIR_FIELDS}
    with open(pair_results_path(rec["rule"], rec["bc"]), "a") as f:
        f.write(json.dumps(ordered) + "\n")


def append_manifest(rule: int, bc: str, N: int, runtime: float,
                    ergodic: bool, extra: Optional[dict] = None) -> None:
    _ensure_dirs()
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": rule, "bc": bc, "N": N,
        "runtime": round(runtime, 4),
        "ergodic": ergodic,
        "engine_version": ENGINE_VERSION,
    }
    if extra:
        entry.update(extra)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry) + "\n")
