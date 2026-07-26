"""
Single-flip reduction of the transition graph of a UNITARY rule.

Theorem (flip reduction).  Let the rule be unitary (symbols in {I, V} only) and
let s_i(x) = t[2 x_{i-1} + x_{i+1}] be the symbol fired at site i when the
controls read x.  Then the sector partition (= weakly connected components of
the one-cycle transition graph, which for a unitary rule coincide with its
SCCs) equals the connected components of the UNDIRECTED graph

    F :   x  ---  x XOR 2^i      for every i with s_i(x) = V .

Proof.  A cycle is two depth-1 half-layers (even sites, then odd sites); by the
no-interference theorem (R1) each half-layer is, conditioned on the frozen
complementary sublattice, a product of single-qubit gates acting on disjoint
qubits.  Hence the image of the half-layer from a basis state x is exactly the
SUBCUBE spanned by the sites whose symbol is V, with the complementary
sublattice held fixed.  Two consequences:

  (i)  x lies in its own half-layer image (<b|I|b> = 1 and <b|H|b> = +-1/sqrt2
       are both nonzero), so the composite image from x contains both
       half-layer images:  Phi(x) contains A(x) and B(x).
  (ii) inside a half-layer image the frozen sublattice -- and therefore the
       V/I pattern itself -- does not change, so the subcube is connected by
       SINGLE flips that are themselves edges of F.

By (i) every F-edge is realised by a composite path, and by (ii) every
composite edge x -> z (z in B(y), y in A(x)) is realised by an F-path
x ~ y ~ z.  The two graphs therefore have the same components.  QED

Note the reduction is layer-agnostic: the brick-wall ORDER drops out, because
the controls of site i are read off x itself in the half-layer in which i is
frozen.  This is why the sector partition of a unitary QECA is a property of
the local symbol table alone.

Cost: O(N 2^N) with a tiny constant, versus O(2^N |succ|) for the streamed
engine (|succ| grows exponentially in N for the Hadamard-rich rules).  The
numpy path below never materialises the node index: for each site it addresses
the three relevant bits by reshaping the label array, which makes the whole
sweep a handful of contiguous memory passes.

Validated against graph.scc.analyze for all 16 unitary rules, both boundary
conventions, N = 3..12 (tests/test_flip_graph.py).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..core.rules import Tuple4, is_unitary


def v_mask(x: int, N: int, t: Tuple4, bc: str) -> int:
    """Bitmask of the sites at which the Hadamard fires when controls read x."""
    out = 0
    for i in range(N):
        if bc == "pbc":
            left = (x >> ((i - 1) % N)) & 1
            right = (x >> ((i + 1) % N)) & 1
        elif bc == "obc0":
            left = (x >> (i - 1)) & 1 if i > 0 else 0
            right = (x >> (i + 1)) & 1 if i < N - 1 else 0
        else:
            raise ValueError(f"unknown boundary convention {bc!r}")
        if t[2 * left + right] == "V":
            out |= 1 << i
    return out


def flip_components(N: int, t: Tuple4, bc: str) -> List[int]:
    """Sector sizes (descending) via union-find over the flip graph. Pure python
    reference implementation; use flip_components_np for N >= 20."""
    if not is_unitary(t):
        raise ValueError("the flip reduction is proved for unitary rules only")
    total = 1 << N
    parent = list(range(total))
    size = [1] * total

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for x in range(total):
        m = v_mask(x, N, t, bc)
        while m:
            low = m & -m
            m ^= low
            ra, rb = find(x), find(x ^ low)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
    sizes = [size[r] for r in range(total) if parent[r] == r]
    sizes.sort(reverse=True)
    return sizes


# --- vectorised path ---------------------------------------------------------

def _site_pairs(i: int, N: int, t: Tuple4, bc: str):
    """
    Describe site i's flip edges as slice pairs of a 3-axis reshape of the label
    array.  Returns (n_hi, blk, n_lo, pairs) meaning: view = label.reshape(
    n_hi, blk, n_lo) and every (j0, j1) in pairs is a flip edge between
    view[:, j0, :] and view[:, j1, :], elementwise.  Returns None when the three
    relevant bits are not contiguous (pbc wrap-around).
    """
    if bc == "obc0":
        if N < 3:
            return None
        if i == 0:
            # bits (site 0, right neighbour 1); left control is the vacuum 0.
            pairs = [(2 * b1, 2 * b1 + 1) for b1 in (0, 1)
                     if t[2 * 0 + b1] == "V"]
            return (1 << (N - 2), 4, 1, pairs)
        if i == N - 1:
            # bits (left neighbour N-2, site N-1); right control is vacuum 0.
            pairs = [(bl, bl + 2) for bl in (0, 1) if t[2 * bl + 0] == "V"]
            return (1, 4, 1 << (N - 2), pairs)
        # bits i-1, i, i+1 contiguous; index = 4 b_{i+1} + 2 b_i + b_{i-1}
        pairs = []
        for bl in (0, 1):
            for br in (0, 1):
                if t[2 * bl + br] == "V":
                    j0 = 4 * br + bl
                    pairs.append((j0, j0 + 2))
        return (1 << (N - i - 2), 8, 1 << (i - 1), pairs)
    if bc == "pbc":
        if 1 <= i <= N - 2:
            pairs = []
            for bl in (0, 1):
                for br in (0, 1):
                    if t[2 * bl + br] == "V":
                        j0 = 4 * br + bl
                        pairs.append((j0, j0 + 2))
            return (1 << (N - i - 2), 8, 1 << (i - 1), pairs)
        return None            # i = 0 or N-1 wraps; handled by the slow path
    raise ValueError(f"unknown boundary convention {bc!r}")


def _sizes_from_labels(label, chunk: int) -> List[int]:
    """
    Component sizes without a 2^N-sized bincount.  A converged label array holds
    the minimum node index of each component, and unitary QECA have few sectors,
    so we harvest the distinct values chunkwise and then count them chunkwise.
    Peak extra memory is one bool/uint temporary of `chunk` entries.
    """
    import numpy as np

    n = label.shape[0]
    roots = set()
    for lo in range(0, n, chunk):
        roots.update(int(v) for v in np.unique(label[lo:lo + chunk]))
    root_arr = np.array(sorted(roots), dtype=label.dtype)
    counts = np.zeros(root_arr.shape[0], dtype=np.int64)
    for lo in range(0, n, chunk):
        pos = np.searchsorted(root_arr, label[lo:lo + chunk])
        counts += np.bincount(pos, minlength=root_arr.shape[0]).astype(np.int64)
    assert int(counts.sum()) == n, "size histogram lost nodes (bug)"
    return sorted((int(c) for c in counts), reverse=True)


def flip_components_np(N: int, t: Tuple4, bc: str, *,
                       return_labels: bool = False,
                       chunk: int = 1 << 26,
                       progress=None):
    """
    Sector sizes (descending) via vectorised label propagation on the flip
    graph.

    Memory is one label array of 2^N entries: uint32 (4 B) up to N = 32, so
    2^32 = 4.29e9 nodes cost 17.2 GB.  Pointer jumping is done IN PLACE in
    chunks and the size histogram is harvested chunkwise, so no second array of
    2^N entries is ever allocated -- that is what puts N = 32 inside 32 GB.
    """
    import numpy as np

    if not is_unitary(t):
        raise ValueError("the flip reduction is proved for unitary rules only")
    total = 1 << N
    dt = np.uint32 if N <= 32 else np.uint64
    label = np.arange(total, dtype=dt)

    # sites whose three control/target bits are contiguous -> vectorised;
    # the rest (pbc wrap) fall back to an explicit index gather.
    fast, slow = [], []
    for i in range(N):
        sp = _site_pairs(i, N, t, bc)
        (fast if sp is not None else slow).append((i, sp))

    slow_edges = []
    if slow:
        idx = np.arange(total, dtype=np.int64)
        for i, _ in slow:
            if bc == "pbc":
                left = (idx >> ((i - 1) % N)) & 1
                right = (idx >> ((i + 1) % N)) & 1
            else:
                left = (idx >> (i - 1)) & 1 if i > 0 else np.zeros_like(idx)
                right = (idx >> (i + 1)) & 1 if i < N - 1 else np.zeros_like(idx)
            sym = np.array([1 if s == "V" else 0 for s in t], dtype=np.int8)
            fire = sym[(2 * left + right).astype(np.int8)].astype(bool)
            u = idx[fire]
            slow_edges.append((u.astype(dt), (u ^ (1 << i)).astype(dt)))
        del idx

    def checksum() -> int:
        """Monotone progress witness: labels only ever decrease, so the sum is
        strictly decreasing until the fixed point.  Computed by a reduction, so
        no array of 2^N entries is allocated for the convergence test."""
        s = 0
        for lo in range(0, total, chunk):
            s += int(label[lo:lo + chunk].sum(dtype=np.int64))
        return s

    rounds = 0
    prev = checksum()
    while True:
        rounds += 1
        for i, sp in fast:
            n_hi, blk_len, n_lo, pairs = sp
            if not pairs:
                continue
            view = label.reshape(n_hi, blk_len, n_lo)
            for j0, j1 in pairs:
                a = view[:, j0, :]
                b = view[:, j1, :]
                np.minimum(a, b, out=a)        # in place: no 2^(N-3) temporary
                b[...] = a
        for u, v in slow_edges:
            m = np.minimum(label[u], label[v])
            label[u] = m
            label[v] = m
            del m
        # pointer jumping to a fixed point, in place and in chunks: label[i] <-
        # label[label[i]] only ever replaces a label by an equal-or-smaller one
        # on the same component, so doing it in place (and thus partly with
        # already-updated entries) is correct and converges faster.
        while True:
            before = checksum()
            for lo in range(0, total, chunk):
                blk = label[lo:lo + chunk]
                blk[...] = label[blk]
            if checksum() == before:
                break
        cur = checksum()
        if progress:
            progress(rounds, cur)
        if cur == prev:
            break
        prev = cur

    out = _sizes_from_labels(label, chunk)
    if return_labels:
        return out, label
    return out
