"""
R32 thermalisation beyond level statistics (2026-10-15):

  * spectral form factor (SFF) of the symmetry-resolved blocks, against COE / CUE;
  * ETH for local observables: diagonal matrix elements in the eigenstates of the largest
    gate-algebra block (open chain) or of the complex momentum sectors (ring), and the
    scaling of their eigenstate-to-eigenstate fluctuations with the block dimension;
  * dynamics: U^t applied to product initial states with the sparse layers (N <= 16):
    half-chain entanglement, a local observable and the return probability.

Floquet target: infinite temperature inside the block the state lives in.
Site convention as core.cycle / blockfirst: site 0 = least significant bit.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import blockfirst as B
from . import stats as S


# --- spectral form factor ---------------------------------------------------------

def sff(blocks, tmax: int):
    """
    K(t) = < |sum_n exp(-i x_n t)|^2 / d >_blocks for t = 1..tmax, each block weighted by d
    (connected part not needed for t >= 1 at these sizes), plus the dimension-scaled time.
    blocks: list of 1d arrays of levels (one per resolved block).
    Returns t, K (normalised so that K -> 1 at late times), and the mean block dimension.
    """
    t = np.arange(1, tmax + 1)
    num = np.zeros(t.size); den = 0.0
    for x in blocks:
        x = np.asarray(x, float)
        d = x.size
        if d < 20:
            continue
        z = np.exp(-1j * np.outer(t, x))           # (t, d)
        num += np.abs(z.sum(1)) ** 2               # |Tr U^t|^2 restricted to the block
        den += d
    return t, num / den


def sff_rmt(tau, beta):
    """Large-D form factor per level, tau = t / D (COE beta = 1, CUE beta = 2)."""
    tau = np.asarray(tau, float)
    if beta == 2:
        return np.minimum(tau, 1.0)
    k = 2 * tau - tau * np.log1p(2 * tau)
    k2 = 2 - tau * np.log((2 * tau + 1) / np.maximum(2 * tau - 1, 1e-300))
    return np.where(tau <= 1, k, k2)


def log_binned(t, K, ref, nbins_per_decade=8):
    """Average K and the RMT reference over logarithmic time bins (single realisation: the SFF
    does not self-average; log-binning is the standard remedy).  Returns bin centres, <K>, <ref>."""
    edges = np.unique(np.round(np.logspace(0, np.log10(t[-1]), int(nbins_per_decade * np.log10(t[-1])) + 2)).astype(int))
    c, kb, rb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        if m.any():
            c.append(np.sqrt(lo * hi)); kb.append(K[m].mean()); rb.append(ref[m].mean())
    return np.array(c), np.array(kb), np.array(rb)


def thouless_time(t, K, D, beta, tol=0.3, tmax_frac=0.5, factor=None):
    """First log-bin centre from which <K>/<K_RMT> stays within 1 +- tol (or, with factor=f, within
    [1/f, f]) up to t = tmax_frac * D.  At the accessible N the SFF approaches the ramp only slowly,
    so the robust proxy used in R32 is factor = 2 ("t_2")."""
    c, kb, rb = log_binned(t, K, sff_rmt(t / D, beta))
    sel = c <= tmax_frac * D
    c, ratio = c[sel], (kb / rb)[sel]
    ok = (np.abs(ratio - 1) < tol) if factor is None else ((ratio < factor) & (ratio > 1 / factor))
    for i in range(len(c)):
        if np.all(ok[i:]):
            return float(c[i])
    return None


# --- ETH ---------------------------------------------------------------------------

def _gate_blocks(w, N, bc, seed=1):
    """Gate-algebra blocks (generic-K trick, dense; N <= 12): list of orthonormal bases."""
    from scipy.sparse.csgraph import connected_components
    gates = [B.gate_sparse(w, N, i, bc).toarray() for i in range(N)]
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.5, 1.5, N), rng.uniform(0.5, 1.5, N)
    K = sum(a[i] * gates[i] for i in range(N)); K2 = sum(b[i] * gates[i] for i in range(N))
    ev, Q = np.linalg.eigh(K)
    M = Q.T @ K2 @ Q
    nc, lab = connected_components(sp.csr_matrix(np.abs(M) > 1e-9 * np.abs(M).max()), directed=False)
    return [Q[:, lab == c] for c in range(nc)]


def _nondegenerate(x, tol=1e-8):
    """Mask of levels with no other level within tol on the circle (their eigenvectors are unique,
    so diagonal matrix elements are basis independent)."""
    z = np.exp(1j * np.asarray(x))
    d = np.abs(z[:, None] - z[None, :]); np.fill_diagonal(d, np.inf)
    return d.min(axis=1) > tol


def z_profile(N):
    """Diagonal of Z_i for every site, shape (N, 2^N)."""
    x = np.arange(1 << N)
    return np.array([1 - 2 * ((x >> i) & 1) for i in range(N)], float)


def eth_obc(w: int, N: int):
    """
    Largest gate-algebra block on the open chain: eigenvectors of U in that block, diagonal
    elements of the reflection-symmetric central magnetisation O = (Z_c + Z_c')/2, and of the
    block-averaged magnetisation density m = sum_i Z_i / N.  Returns theta, O_nn, m_nn, the
    block (infinite temperature) averages and the dimension.
    """
    U = (B.layer_sparse(w, N, 1, "obc0") @ B.layer_sparse(w, N, 0, "obc0")).toarray()
    Qb = max(_gate_blocks(w, N, "obc0"), key=lambda q: q.shape[1])
    Ub = Qb.T @ U @ Qb
    ev, V = np.linalg.eig(Ub)
    V = V / np.linalg.norm(V, axis=0)              # U_b unitary, non-degenerate: orthogonal eigenvectors
    Z = z_profile(N)
    c, c2 = (N - 1) // 2, N // 2
    Oc = (Z[c] + Z[c2]) / 2; Om = Z.mean(0)
    Psi = Qb @ V                                     # eigenvectors in the full space
    p = np.abs(Psi) ** 2
    O_nn, m_nn = Oc @ p, Om @ p
    P = Qb @ Qb.T
    th = np.angle(ev)
    nondeg = _nondegenerate(th)
    return {"theta": th, "O": O_nn, "m": m_nn, "dim": Qb.shape[1], "nondeg": nondeg,
            "O_T": float((np.diag(P) * Oc).sum() / Qb.shape[1]),
            "m_T": float((np.diag(P) * Om).sum() / Qb.shape[1])}


def eth_ring(w: int, N: int):
    """Complex momentum sectors on the ring: eigenvectors of Omega_T,j; magnetisation density."""
    n = 1 << N
    LA = B.layer_sparse(w, N, 0, "pbc"); x = np.arange(n, dtype=np.int64)
    T1 = sp.csr_matrix((np.ones(n), (B.rot(x, 1, N), x)), shape=(n, n)); Om = T1 @ LA
    m = z_profile(N).mean(0)
    out = []
    for j in range(N // 2):
        if j == 0 or (2 * j) % (N // 2) == 0:
            continue                                  # real sectors carry an unresolved Z2
        P = B.momentum_basis(N, j).toarray()
        Oj = P.conj().T @ (Om @ P)
        ev, V = np.linalg.eig(Oj)
        V = V / np.linalg.norm(V, axis=0)
        mj = np.real(np.einsum("in,i,in->n", (P @ V).conj(), m, P @ V))
        mT = float(np.real(np.trace(P.conj().T @ (m[:, None] * P))) / P.shape[1])
        ph = np.angle(ev)
        out.append({"j": j, "phi": ph, "m": mj, "m_T": mT, "dim": P.shape[1], "nondeg": _nondegenerate(ph)})
    return out


# --- dynamics ----------------------------------------------------------------------

def product_state(N, kind: str, rng=None):
    """Product initial states: 'zero', 'neel', 'hplus' (the scar), 'hplus_flip' (h- in the middle),
    'random_z' (random computational basis state), 'random_x' (random |+-> on each site)."""
    if kind == "zero":
        v = np.zeros(1 << N); v[0] = 1; return v
    if kind == "neel":
        v = np.zeros(1 << N); v[sum(1 << i for i in range(0, N, 2))] = 1; return v
    if kind == "random_z":
        rng = rng or np.random.default_rng(0)
        v = np.zeros(1 << N); v[int(rng.integers(1 << N))] = 1; return v
    facs = []
    for i in range(N):
        if kind == "hplus":
            facs.append(B.H_PLUS)
        elif kind == "hplus_flip":
            facs.append(B.H_MINUS if i == N // 2 else B.H_PLUS)
        elif kind == "random_x":
            rng = rng or np.random.default_rng(0)
            facs.append(np.array([1, 1 if rng.random() < 0.5 else -1]) / np.sqrt(2))
        else:
            raise ValueError(kind)
    v = np.ones(1)
    for f in facs:
        v = np.kron(f, v)                         # site 0 = LSB = last factor
    return v


def half_chain_entropy(v, N):
    NA = N // 2
    s = np.linalg.svd(v.reshape(1 << (N - NA), 1 << NA), compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log(s)).sum())


def quench(w: int, N: int, bc: str, kind: str, T: int, rng=None):
    """Evolve a product state for T periods; record S_half(t), <Z_c>(t), return probability."""
    LA, LB = B.layer_sparse(w, N, 0, bc), B.layer_sparse(w, N, 1, bc)
    v0 = product_state(N, kind, rng).astype(complex)
    Zc = z_profile(N)[N // 2]
    v = v0.copy()
    S_, Z_, R_ = [], [], []
    for t in range(T + 1):
        S_.append(half_chain_entropy(v, N)); Z_.append(float(np.real(np.vdot(v, Zc * v))))
        R_.append(float(abs(np.vdot(v0, v)) ** 2))
        v = LB @ (LA @ v)
    return {"S": np.array(S_), "Z": np.array(Z_), "R": np.array(R_)}


# --- figures -------------------------------------------------------------------------

#: one representative per class for the thermalisation figures (rule, class label)
CLASS_REPS = [(57, r"57 ($k=3$)"), (147, r"147 ($k=3$)"), (153, r"153 (one-sided)"),
              (60, r"60 (one-sided)"), (150, r"150 (XOR)"), (201, r"201 ($k=1$)")]


def dynamics_figure(path_npz, out_dir=None):
    """(a) random product states, open chain; (b) the same on the ring; (c) the scar with one
    |h-> defect in the middle (k = 3 rules); S_half / S_Page against time, N = 14."""
    from ..viz import thesis_style as ts
    from .figures import FIG_DIR
    import os
    plt = ts.use(ts._mode)
    d = np.load(path_npz); N = int(d["N"]); page = S.page_entropy(N // 2, N - N // 2)
    t = np.arange(int(d["T"]) + 1)
    fig, axs = ts.subplots(1, 3, aspect=0.36, sharey=True)
    for ax, bc in zip(axs[:2], ("obc0", "pbc")):
        for i, (w, lab) in enumerate(CLASS_REPS):
            ax.plot(t, d[f"{w}_{bc}_random_S"] / page, color=ts.PALETTE[i], lw=ts.LW["data"],
                    label=lab if bc == "obc0" else None)
        ax.set_title("open chain" if bc == "obc0" else "ring", fontsize=ts.FS["annot"])
        ax.set_xscale("log"); ax.set_xlabel(r"periods $t$")
    for i, w in enumerate((57, 54, 147)):
        axs[2].plot(t, d[f"{w}_obc0_hplus_flip_S"] / page, color=ts.PALETTE[i], lw=ts.LW["data"], label=f"{w}")
        axs[2].plot(t, d[f"{w}_pbc_hplus_flip_S"] / page, color=ts.PALETTE[i], lw=ts.LW["data"], ls=":")
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"periods $t$")
    axs[2].set_title(r"scar $+$ one $|h_-\rangle$", fontsize=ts.FS["annot"])
    axs[0].set_ylabel(r"$S_\mathrm{half}/S_\mathrm{Page}$"); axs[0].set_ylim(-0.03, 1.1)
    for ax in axs:
        ax.axhline(1, color=ts.GREY, lw=0.5, zorder=0)
    axs[0].legend(loc="upper left", fontsize=ts.FS["legend"] - 3, handlelength=1.0, borderaxespad=0.2)
    axs[2].legend(loc="upper left", fontsize=ts.FS["legend"] - 3, handlelength=1.0, borderaxespad=0.2,
                  title=r"open (solid), ring (dotted)", title_fontsize=ts.FS["legend"] - 4)
    for ax, lab in zip(axs, "abc"):
        ts.panel_label(ax, f"({lab})")
    stem = os.path.join(out_dir or FIG_DIR, f"fig_r32_dynamics_N{N}")
    ts.save(fig, stem); plt.close(fig)
    return stem


def ring_complex_blocks(w, N):
    """Level sets of the complex momentum sectors on the ring (Julia data N <= 14; block-first N = 16)."""
    import os
    from . import floquet as F
    if N == 16:
        f = os.path.join(F.SPECTRA_DIR, "blockfirst", f"ring_N16_W{w}.npz")
        if not os.path.exists(f):
            return []
        d = np.load(f)
        return [d[k] for k in d.files if k.startswith("phi_") and d[k].size >= 50 and not S.conj_symmetric(d[k])]
    d = F.load(F.spec_path("H", N, w, "pbc"))
    out = []
    for k, m in S.groups(S.block_keys(d)):
        x = S.block_levels_var(d, m)
        if m.size >= 50 and not S.conj_symmetric(x):
            out.append(x)
    return out


#: chaotic rules and their ring ensemble (complex momentum sectors)
SFF_RULES = [(57, 2, "57 (CUE)"), (147, 1, "147"), (54, 1, "54"), (153, 1, "153"), (60, 1, "60")]


def eth_sff_figure(eth_json, sff_json, out_dir=None):
    """(a) ETH: eigenstate-to-eigenstate fluctuation of the magnetisation density in the complex ring
    sectors against the sector dimension, with d^-1/2; (b) SFF / RMT ramp at N = 16 (log-binned);
    (c) t_2 (SFF within a factor 2 of the ramp) against N."""
    import json, os
    from ..viz import thesis_style as ts
    from .figures import FIG_DIR
    plt = ts.use(ts._mode)
    eth = json.load(open(eth_json)); sff_ = json.load(open(sff_json))
    fig, (a, b, c) = ts.subplots(1, 3, aspect=0.36)
    for i, (w, beta, lab) in enumerate(SFF_RULES):
        col = ts.PALETTE[i]
        r = np.array(eth[str(w)]["ring"], float)
        a.plot(r[:, 1], r[:, 2], "o-", color=col, ms=ts.MS, lw=ts.LW["data"], label=lab)
        bl = ring_complex_blocks(w, 16)
        if bl:
            D = np.mean([x.size for x in bl]); t, K = sff(bl, int(D))
            cc, kb, rb = log_binned(t, K, sff_rmt(t / D, beta), 6)
            b.plot(cc / D, kb / rb, color=col, lw=ts.LW["data"])
        rows = [x for x in sff_[str(w)] if x[2] is not None]
        c.plot([x[0] for x in rows], [x[2] for x in rows], "o-", color=col, ms=ts.MS, lw=ts.LW["data"])
    d = np.array([50, 3000]); a.plot(d, 0.9 * d ** -0.5, color=ts.GREY, ls="--", lw=0.6)
    a.annotate(r"$d^{-1/2}$", (2000, 0.9 * 2000 ** -0.5), color=ts.GREY, fontsize=ts.FS["annot"] - 1,
               xytext=(-4, 6), textcoords="offset points", ha="right")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"sector dimension $d$")
    a.set_ylabel(r"$\sigma$ of $\langle n|m|n\rangle$")
    a.legend(loc="lower left", fontsize=ts.FS["legend"] - 3, handlelength=1.0, borderaxespad=0.2)
    b.axhline(1, color=ts.GREY, lw=0.5); b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"$t/D$"); b.set_ylabel(r"$K(t)/K_\mathrm{RMT}(t)$"); b.set_title(r"$N=16$", fontsize=ts.FS["annot"])
    c.set_yscale("log"); c.set_xlabel(r"$N$"); c.set_ylabel(r"$t_2$ (periods)")
    for ax, lab in zip((a, b, c), "abc"):
        ts.panel_label(ax, f"({lab})")
    stem = os.path.join(out_dir or FIG_DIR, "fig_r32_eth_sff")
    ts.save(fig, stem); plt.close(fig)
    return stem


def write_thermal_table(out=None) -> str:
    """LaTeX table (R32 thermalisation): per rule, ring <r> at the largest N (complex sectors), ETH exponent
    (ring), t_2 at N = 16, and late-time S/S_Page after quenches (N = 14, t = 200..300)."""
    import json, os
    from .. import results_io
    from . import floquet as F
    from .analysis import pattern_class
    eth = json.load(open(os.path.join(F.SPECTRA_DIR, "thermal_eth.json")))
    sffj = json.load(open(os.path.join(F.SPECTRA_DIR, "thermal_sff.json")))
    dyn = np.load(os.path.join(F.SPECTRA_DIR, "thermal_dynamics_N14.npz"))
    page = S.page_entropy(7, 7); late = slice(200, 301)
    rows = []
    for w in (57, 54, 147, 153, 60, 150, 201, 156):
        bl = ring_complex_blocks(w, 16) or ring_complex_blocks(w, 14)
        Nr = 16 if ring_complex_blocks(w, 16) else 14
        r = np.concatenate([S.block_ratios(x) for x in bl]) if bl else np.empty(0)
        rr = f"{r.mean():.3f} ({Nr})" if r.size > 100 else "--"
        alpha = "--"
        if str(w) in eth and w in (57, 54, 147, 153, 60):
            e = np.array(eth[str(w)]["ring"], float)
            alpha = f"{np.polyfit(np.log(e[:, 1]), np.log(e[:, 2]), 1)[0]:.2f}"
        t2 = "--"
        if str(w) in sffj:
            last = [x for x in sffj[str(w)] if x[0] == 16]
            t2 = f"{last[0][2]:.0f}" if last and last[0][2] is not None else "--"
        f = lambda key: f"{dyn[key][late].mean() / page:.2f}"
        rows.append(f"{pattern_class(w)} & {w} & {rr} & {alpha} & {t2} & {f(f'{w}_obc0_random_S')} & "
                    f"{f(f'{w}_pbc_random_S')} & {f(f'{w}_obc0_hplus_flip_S')} & {f(f'{w}_pbc_hplus_flip_S')} \\\\")
    tex = (r"\begin{tabular}{llrrrrrrr}" "\n\\toprule\n"
           r" & & ring $\langle r\rangle$ & ETH & $t_2$ & \multicolumn{2}{c}{random product} & "
           r"\multicolumn{2}{c}{scar $+\,|h_-\rangle$} \\" "\n"
           r"$k$ & $W$ & ($N$) & exponent & ($N=16$) & open & ring & open & ring \\" "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex", "tab_r32_thermal.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    return out
