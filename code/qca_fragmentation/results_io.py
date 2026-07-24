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
            rec = json.loads(line)
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
