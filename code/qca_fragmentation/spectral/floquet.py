"""
The Floquet operator U of a unitary rule, from the exact engine, and the reader
for the Julia spectral data.

Conventions (shared with HSF/generate_spectral_data.jl):
  * U v = exp(-i theta) v, theta in (-pi, pi].
  * Julia/HSF basis: site 1 = most significant bit.  This repo: site 0 = least
    significant bit.  So HSF index - 1 = bit_reverse(our index).
  * HSF rule number = 8 c11 + 4 c10 + 2 c01 + c00 (bit set = gate acts).
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .. import results_io
from ..core import cycle, rules

SPECTRA_DIR = os.path.join(results_io.REPO_ROOT, "results_spectra")
JULIA_SCRIPT = os.path.join(os.path.dirname(results_io.REPO_ROOT), "HSF",
                            "generate_spectral_data.jl")

#: the 16 unitary rules in HSF order 0..15
UNITARY_WOLFRAM = None


def hsf_to_wolfram(r: int) -> int:
    t = tuple("V" if (r >> bit) & 1 else "I" for bit in (0, 1, 2, 3))   # (r00, r01, r10, r11)
    return rules.tuple_to_wolfram(t)


def wolfram_to_hsf(w: int) -> int:
    t = rules.wolfram_to_tuple(w)
    assert all(s in "IV" for s in t), f"W{w} is not unitary"
    return sum((1 << b) for b, s in enumerate(t) if s == "V")


UNITARY_WOLFRAM = [hsf_to_wolfram(r) for r in range(16)]


def floquet_matrix(rule: int, N: int, gate: str = "H", bc: str = "obc0") -> np.ndarray:
    """
    Dense U (complex) in THIS repo's basis (site 0 = LSB), from the exact
    Z[1/sqrt2] engine for gate H, or the permutation engine for gate X.
    Column x = U|x>.
    """
    n = 1 << N
    U = np.zeros((n, n))
    if gate == "H":
        t = rules.wolfram_to_tuple(rule)
        for x in range(n):
            (amps, m), = cycle.one_cycle_branches(x, N, t, bc)
            for y, (a, b) in amps.items():
                U[y, x] = (a + b * np.sqrt(2.0)) / 2.0 ** m
    elif gate == "X":
        from ..permutation import xca
        f = xca.step_table(rule, N, bc)
        U[np.asarray(f), np.arange(n)] = 1.0
    else:
        raise ValueError(gate)
    return U


def bit_reverse_perm(N: int) -> np.ndarray:
    """p[our index] = HSF index - 1."""
    idx = np.arange(1 << N)
    out = np.zeros_like(idx)
    for k in range(N):
        out |= ((idx >> k) & 1) << (N - 1 - k)
    return out


def load(path: str) -> Dict[str, np.ndarray]:
    """Read one Julia spec_*.jld2 file (JLD2 is HDF5) into plain numpy values."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k.startswith("_"):
                continue
            v = f[k][()]
            if isinstance(v, bytes):
                v = v.decode()
            out[k] = np.asarray(v) if not isinstance(v, str) else v
    for k in ("N", "rule_hsf", "rule_wolfram"):
        out[k] = int(out[k])
    return out


def spec_dir(gate: str, bc: str = "obc0") -> str:
    """results_spectra/{gate} for obc0, results_spectra/{gate}_pbc for pbc."""
    return os.path.join(SPECTRA_DIR, gate if bc == "obc0" else f"{gate}_{bc}")


def spec_path(gate: str, N: int, rule_wolfram: int, bc: str = "obc0") -> str:
    r = wolfram_to_hsf(rule_wolfram)
    return os.path.join(spec_dir(gate, bc), f"spec_N{N}_{gate}_HSF{r}_W{rule_wolfram}.jld2")


def available(gate: str = "H", bc: str = "obc0") -> Dict[int, list]:
    """{wolfram rule: sorted list of N} present on disk."""
    import re
    d = spec_dir(gate, bc)
    res: Dict[int, list] = {}
    if not os.path.isdir(d):
        return res
    for fn in os.listdir(d):
        m = re.match(rf"spec_N(\d+)_{gate}_HSF\d+_W(\d+)\.jld2$", fn)
        if m:
            res.setdefault(int(m.group(2)), []).append(int(m.group(1)))
    return {k: sorted(v) for k, v in sorted(res.items())}


def circle_distance(a, b) -> float:
    """
    Max distance between two multisets of angles, matched on the circle.

    Sorting raw angles is NOT safe: a level at theta = 0 computed as -1e-16 wraps
    to 2 pi and shifts the whole sorted list by one.  Cut the circle in the middle
    of the largest gap of `a` instead, then sort and compare.
    """
    a = np.mod(np.asarray(a, float), 2 * np.pi)
    b = np.mod(np.asarray(b, float), 2 * np.pi)
    if a.size != b.size:
        return np.inf
    s = np.sort(a)
    gaps = np.diff(np.r_[s, s[0] + 2 * np.pi])
    cut = s[np.argmax(gaps)] + gaps.max() / 2
    a2 = np.sort(np.mod(a - cut, 2 * np.pi))
    b2 = np.sort(np.mod(b - cut, 2 * np.pi))
    return float(np.max(np.abs(a2 - b2)))
