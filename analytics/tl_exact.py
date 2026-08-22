"""The exact identity behind the failed TL relation, and where TL is recovered.

Claim to test:   e_i e_{i+1} e_i  =  tau * e_i * P^{n*}_{i+2}
                 e_i e_{i-1} e_i  =  tau * e_i * P^{m*}_{i-2}
with tau = |<m*|h->|^2 |<n*|h->|^2.  If so, the TL relation holds exactly on the
subspace where the SPECTATOR site is already in the V-context value, and the
effective loop parameter is delta = 1/sqrt(tau).

Then: restrict to each rule's invariant constrained space and test TL there.
Finally: check the resulting delta against the Jones admissibility condition
(orthogonal-projector representations of TL need delta >= 2 or
delta = 2 cos(pi/(m+1))).
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


def constrained_states(N, forbid):
    """States with no two ADJACENT sites both equal to `forbid`."""
    out = []
    for x in range(1 << N):
        b = [(x >> k) & 1 for k in range(N)]
        if all(not (b[k] == b[k + 1] == forbid) for k in range(N - 1)):
            out.append(x)
    return out


#: rule -> the bit that may not appear twice adjacently in the invariant space
FORBID = {201: 1, 73: 1, 108: 0, 109: 0}


def analyse(rule, N=8):
    m, n = v_context(rule)
    word = "".join(R.wolfram_to_tuple(rule))
    tau = abs(hm[m]) ** 2 * abs(hm[n]) ** 2
    delta = 1.0 / np.sqrt(tau)
    print(f"\n=== W{rule} ({word})  (m*,n*)=({m},{n})  N={N} ===")
    print(f"   tau = |<m*|h->|^2 |<n*|h->|^2 = {tau:.9f}")
    print(f"   delta = 1/sqrt(tau)           = {delta:.9f}")

    es = {i: e_op(i, N, m, n) for i in range(1, N - 1)}

    # (A) the exact spectator identity, on the FULL space
    worst_r, worst_l = 0.0, 0.0
    for i in sorted(es):
        if i + 1 in es:
            lhs = es[i] @ es[i + 1] @ es[i]
            rhs = tau * es[i] @ site_op(PROJ[n], i + 2, N)
            worst_r = max(worst_r, float(np.abs(lhs - rhs).max()))
        if i - 1 in es:
            lhs = es[i] @ es[i - 1] @ es[i]
            rhs = tau * es[i] @ site_op(PROJ[m], i - 2, N)
            worst_l = max(worst_l, float(np.abs(lhs - rhs).max()))
    print(f"   (A) e_i e_(i+1) e_i = tau e_i P^n*_(i+2) : max err {worst_r:.2e}")
    print(f"       e_i e_(i-1) e_i = tau e_i P^m*_(i-2) : max err {worst_l:.2e}")

    # (B) TL on the invariant constrained space, if the rule has one
    if rule in FORBID:
        S = constrained_states(N, FORBID[rule])
        idx = np.array(S)
        P = np.zeros((len(S), 1 << N))
        for a, x in enumerate(S):
            P[a, x] = 1.0
        er = {i: P @ es[i] @ P.T for i in es}
        # is the subspace invariant under e_i?
        leak = max(float(np.abs(es[i] @ P.T - P.T @ er[i]).max()) for i in es)
        print(f"   (B) constrained space (no two adjacent {FORBID[rule]}s): "
              f"dim {len(S)}, e_i leakage {leak:.2e}")
        proj = max(float(np.abs(er[i] @ er[i] - er[i]).max()) for i in er)
        print(f"       e_i still a projector there : max err {proj:.2e}")
        norms = {i: float(np.abs(er[i]).max()) for i in er}
        print(f"       ||e_i|| on the sector       : "
              + ", ".join(f"{v:.3f}" for v in norms.values()))
        worst = 0.0
        vanish = 0
        for i in sorted(er):
            for j in (i - 1, i + 1):
                if j not in er:
                    continue
                M = er[i] @ er[j] @ er[i]
                if np.abs(M).max() < 1e-12:
                    vanish += 1
                    continue
                worst = max(worst, float(np.abs(M - tau * er[i]).max()))
        print(f"       e_i e_j e_i = tau e_i on sector: max err {worst:.2e}"
              f"   ({vanish} products vanish identically)")

    # (C) Jones admissibility
    index = delta ** 2
    allowed = index >= 4.0 - 1e-12
    special = [(mm, 4 * np.cos(np.pi / mm) ** 2) for mm in range(3, 12)]
    near = min(special, key=lambda t: abs(t[1] - index))
    print(f"   (C) Jones index delta^2 = {index:.6f}; "
          f"continuum [4,inf) allowed: {allowed}; "
          f"nearest discrete 4cos^2(pi/{near[0]}) = {near[1]:.6f}")
    if not allowed and abs(near[1] - index) > 1e-6:
        print("       -> NOT an admissible index for orthogonal TL projectors")
    return tau, delta


if __name__ == "__main__":
    for r in (156, 198, 201, 108):
        analyse(r, N=8)
    print("\nclosed forms:")
    print(f"   4 - 2sqrt2 = {4 - 2*SQ2:.9f}")
    print(f"   2 sqrt2    = {2*SQ2:.9f}")
    print(f"   4 + 2sqrt2 = {4 + 2*SQ2:.9f}")
    print(f"   product of the two symmetric deltas = {(4-2*SQ2)*(4+2*SQ2):.9f}"
          f"   ( (2sqrt2)^2 = {(2*SQ2)**2:.9f} )")
