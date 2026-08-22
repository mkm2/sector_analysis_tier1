"""Q3: in which basis is the fragmentation quantum?

On a 156/198 gap the algebra is C (+) M_g(C).  The one-dimensional summand is a
DARK line: a state annihilated by every generator.  R-T15 says it is the
Rokhsar-Kivelson state sum_k t^k |k> with t = sqrt2 - 1 = tan(pi/8) in the kink
coordinate.  Check that, and check whether it is a product state in the SITE
basis -- if it is not, the fragmentation is quantum, not classical.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/mkmullenbach/Dokumente/Projects/QCA Simulations/"
                   "Claude Sector Analysis/.claude/worktrees/tier1-analysis/code")
from qca_fragmentation.core import rules as R

SQ2 = np.sqrt(2.0)
H = (1 / SQ2) * np.array([[1.0, 1.0], [1.0, -1.0]])
w, V = np.linalg.eigh(H)
hm = V[:, np.argmin(w)] / np.linalg.norm(V[:, np.argmin(w)])
Pm = np.outer(hm, hm)
PROJ = {0: np.array([[1.0, 0.0], [0.0, 0.0]]),
        1: np.array([[0.0, 0.0], [0.0, 1.0]])}


def v_context(rule):
    t = R.wolfram_to_tuple(rule)
    i = t.index("V")
    return (i >> 1) & 1, i & 1


def site_op(op, i, N):
    out = np.array([[1.0]])
    for k in range(N):
        out = np.kron(out, op if k == i else np.eye(2))
    return out


def e_op(i, N, m, n):
    return site_op(PROJ[m], i - 1, N) @ site_op(Pm, i, N) @ site_op(PROJ[n], i + 1, N)


def kink_states(N, m, n):
    out = []
    for k in range(N + 1):
        x = 0
        for s in range(N):
            if (m if s < k else n):
                x |= 1 << (N - 1 - s)
        out.append(x)
    return out                       # index = kink position k


def analyse(rule, N):
    m, n = v_context(rule)
    S = kink_states(N, m, n)
    P = np.zeros((len(S), 1 << N))
    for a, x in enumerate(S):
        P[a, x] = 1.0
    es = {i: e_op(i, N, m, n) for i in range(1, N - 1)}
    er = {i: P @ es[i] @ P.T for i in es}
    alive = [i for i in sorted(er) if np.abs(er[i]).max() > 1e-12]
    # states in the span the generators actually move
    moved = sorted({k for i in alive for k in (i, i + 1)})
    A = np.vstack([er[i][np.ix_(moved, moved)] for i in alive])
    # common kernel restricted to the moved block
    _, s, vh = np.linalg.svd(A)
    k = int(np.sum(s > 1e-10 * s[0]))
    ker = vh[k:].conj().T
    t = SQ2 - 1.0
    out = dict(N=N, module=len(S), gens=len(alive), moved=len(moved),
               ker_dim=ker.shape[1])
    if ker.shape[1] == 1:
        v = ker[:, 0]
        v = v / v[0]
        ratios = v[1:] / v[:-1]
        out["ratio_min"] = float(np.real(ratios).min())
        out["ratio_max"] = float(np.real(ratios).max())
        # the dark vector as a state on the full chain
        full = np.zeros(1 << N)
        for a, kk in enumerate(moved):
            full[S[kk]] = np.real(v[a])
        full /= np.linalg.norm(full)
        out["support"] = int(np.sum(np.abs(full) > 1e-12))
    return out, t


if __name__ == "__main__":
    print("Dark line on the kink module: is it the RK state, and is it a product?")
    print(f"{'rule':>5} {'N':>3} {'module':>7} {'gens':>5} {'moved':>6} "
          f"{'ker dim':>8} {'amp ratio':>12} {'support':>8}")
    for rule in (156, 198):
        for N in (5, 6, 7, 8, 9, 10):
            z, t = analyse(rule, N)
            r = (f"{z['ratio_min']:.6f}" if "ratio_min" in z else "-")
            print(f"{rule:>5} {N:>3} {z['module']:>7} {z['gens']:>5} "
                  f"{z['moved']:>6} {z['ker_dim']:>8} {r:>12} "
                  f"{z.get('support','-'):>8}")
    print(f"\n  sqrt2 - 1 = {SQ2-1:.6f}   tan(pi/8) = {np.tan(np.pi/8):.6f}"
          f"   -(sqrt2-1) = {-(SQ2-1):.6f}")
