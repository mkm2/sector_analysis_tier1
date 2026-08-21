"""
What the six "plastic class" rules actually do, and why 73/109 are different.

The question this module answers came from the trajectory pictures: rules 28,
29, 70, 71, 157 and 199 all settle into a STRIPE pattern whose gaps appear to
oscillate slightly, and the natural worry is that they are therefore trivial.
R17 established the graph-theoretic half of the story -- every terminal SCC of
those six is a complete digraph K_n of size 2^floor((N+1)/3).  What was missing
was what the stripes are, what the oscillation is, and how both follow from
that graph statement.  They do, exactly:

  (1) EVERY terminal SCC of the six rules is a SUBCUBE of the Hilbert space:
      the attractor's states agree on a set of "frozen" sites and range freely
      over the remaining "free" ones, so |A| = 2^(#free).  Verified exhaustively
      for N = 6..14.  Combined with R17's completeness, "K_n on a subcube" says
      precisely that the free bits are redrawn independently and uniformly every
      period -- the graph is the product of k unbiased coins and nothing else.

  (2) The frozen part is a WALL TILING.  Write each attractor as a word over
      {0, 1, f}.  The words are EXACTLY the tilings of the chain by

          199 / 71     tiles  10 , f10
          157 / 29     the mirror image of those
          28  / 70     tiles  10 , 1f0 , plus a leading run of vacuum tiles 0

      with the final tile allowed to be truncated provided the remnant keeps its
      wall.  Set equality, not containment, for N = 6..14.  This DERIVES the two
      constants R17 fitted: the tiling count grows like the plastic number rho
      (tiles of length 2 and 3), and the largest attractor is all-wide-tiles, so
      its dimension is 2^floor((N+1)/3).  The E-carrying children lose the
      vacuum tile, which is why they keep a strict subset.

  (3) Each attractor is an exact DECOHERENCE-FREE SUBSPACE.  On it the jump
      probability is 0 identically and the no-jump Kraus operator does not
      project: the channel restricted to a terminal SCC is a UNITARY.  True for
      all eight rules, which is the sharp form of R17's "the reset never fires
      on its own attractor".

  (4) For the six, that unitary is a product of single-qubit Hadamards on the
      free sites, so each free qubit runs H^t and oscillates with PERIOD 2
      between a definite value and an equal superposition.  That is the observed
      "slightly oscillating gap": the stripes are the walls and the flicker is
      one dark qubit per wide tile.  The trajectory state is a product state,
      half-chain entanglement identically 0 at every N.
      The period-2 signal is a SINGLE-SHOT effect: averaged over trajectories it
      falls by more than an order of magnitude, because different trajectories
      land on different tilings with different phases.

  (5) Rules 29 and 71 -- and only they -- have WCCs holding more than one
      terminal SCC.  n_wcc = n_rec exactly for the other six.  In the R22
      dictionary that means a strong-symmetry block with several enclosures, so
      the steady state inside a block is not unique.  The selection is a set of
      independent FAIR COINS resolved in ONE period: absorption probabilities
      are uniform on 2^k attractors, k = 1 or 2, never anything else.  What the
      coin chooses is the length of one tile, and it needs both an E and a D --
      the E writes the new wall, the D erases the old one -- which is exactly
      what 29 and 71 have and the other four do not.  It is multistability with
      noise-induced selection, not a bifurcation: no parameter is varied and the
      branching ratio is 1/2 at every N.

  (6) 73 and 109 are a different animal.  Their attractors are NOT subcubes.
      The largest is the full hard-core constrained space -- rule 73's states are
      the bitstrings with no two adjacent 1s (dimension F(N+2)), rule 109's the
      bitstrings with no two adjacent 0s (dimension F(N)).  The dissipation
      prepares an exponentially large decoherence-free subspace carrying a
      kinetically constrained unitary, and the trajectory state inside it is
      genuinely entangled with an entropy that grows with N.  That is the
      structural counterpart of the MPO result: the six saturate at an
      N-independent operator Schmidt rank, 73 and 109 grow exponentially.

Reproduce with
    python -m qca_fragmentation.scaling.stripes --rebuild
    python -m qca_fragmentation.scaling.stripes --figure
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.cycle import one_cycle_branches, succ
from ..core.rules import wolfram_to_tuple
from . import sectors

BC = "obc0"
OUT_JSON = os.path.join(sectors.ANALYTICS, "stripes_{bc}.json")

#: the six bounded rules and the two growing ones
BOUNDED = (28, 70, 157, 199, 29, 71)
GROWING = (73, 109)
ALL_RULES = BOUNDED + GROWING

#: tile alphabets.  A tile is a wall '1' plus the empty sites belonging to it;
#: 'f' marks the free (dark) site a wide tile carries.  `part` lists the
#: truncated remnants allowed at the right edge, keyed by the space left; a
#: remnant must still contain its wall.  `vac` allows a leading vacuum run, and
#: `end_open` forbids a full tile from being the last one.
FAMILIES = {
    199: dict(tiles=["10", "f10"], part={1: ["1"], 2: ["f1"]},
              vac=False, end_open=False),
    28: dict(tiles=["10", "1f0"], part={1: ["1"], 2: ["1f"]},
             vac=True, end_open=True),
}
#: rule -> (family key, mirrored?)
FAMILY_OF = {199: (199, False), 71: (199, False), 157: (199, True),
             29: (199, True), 28: (28, False), 70: (28, True)}


# --------------------------------------------------------------- graph layer

def build_graph(rule: int, N: int, bc: str = BC) -> List[List[int]]:
    t = wolfram_to_tuple(rule)
    return [succ(x, N, t, bc) for x in range(1 << N)]


def _tarjan(S: Sequence[Sequence[int]]):
    """Iterative Tarjan; returns (comp_id per node, number of components)."""
    n = len(S)
    index = [0] * n
    low = [0] * n
    onstk = bytearray(n)
    comp = [-1] * n
    stk: List[int] = []
    ctr, ncomp = 1, 0
    for root in range(n):
        if index[root]:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = ctr
                ctr += 1
                stk.append(v)
                onstk[v] = 1
            descend = False
            for i in range(pi, len(S[v])):
                w = S[v][i]
                if not index[w]:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    descend = True
                    break
                if onstk[w] and index[w] < low[v]:
                    low[v] = index[w]
            if descend:
                continue
            if low[v] == index[v]:
                while True:
                    w = stk.pop()
                    onstk[w] = 0
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            work.pop()
            if work and low[v] < low[work[-1][0]]:
                low[work[-1][0]] = low[v]
    return comp, ncomp


def _wcc(S: Sequence[Sequence[int]]) -> List[int]:
    n = len(S)
    p = list(range(n))

    def find(a):
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    for x in range(n):
        for y in S[x]:
            rx, ry = find(x), find(y)
            if rx != ry:
                p[ry] = rx
    return [find(x) for x in range(n)]


def decompose(rule: int, N: int, bc: str = BC) -> Dict:
    """SCCs, terminal SCCs and weak components of the one-cycle digraph."""
    S = build_graph(rule, N, bc)
    comp, ncomp = _tarjan(S)
    members: List[List[int]] = [[] for _ in range(ncomp)]
    for x, c in enumerate(comp):
        members[c].append(x)
    terminal = [True] * ncomp
    for x in range(len(S)):
        cx = comp[x]
        for y in S[x]:
            if comp[y] != cx:
                terminal[cx] = False
    return dict(S=S, comp=comp, members=members, terminal=terminal,
                wcc=_wcc(S), N=N, rule=rule, bc=bc)


# ------------------------------------------------------- attractor structure

def attractor_word(states: Sequence[int], N: int) -> str:
    """'0'/'1' where the attractor is frozen, 'f' where the site is free."""
    out = []
    for i in range(N):
        v = {(x >> i) & 1 for x in states}
        out.append("f" if len(v) == 2 else str(v.pop()))
    return "".join(out)


def is_subcube(states: Sequence[int], word: str) -> bool:
    return len(states) == 2 ** word.count("f")


def tilings(spec: Dict, N: int) -> set:
    out = set()

    def rec(w: str):
        r = N - len(w)
        if r == 0:
            out.add(w)
            return
        for t in spec["tiles"]:
            if len(t) < r or (len(t) == r and not spec["end_open"]):
                rec(w + t)
        for t in spec["part"].get(r, []):
            out.add(w + t)

    if spec["vac"]:
        out.add("0" * N)
        for j in range(N):
            rec("0" * j)
    else:
        rec("")
    return out


def predicted_words(rule: int, N: int) -> set:
    key, mirrored = FAMILY_OF[rule]
    w = tilings(FAMILIES[key], N)
    return {s[::-1] for s in w} if mirrored else w


# ------------------------------------------------------------ absorption law

def transition_row(x: int, N: int, t, bc: str = BC) -> Dict[int, float]:
    """Monitored one-period transition probabilities out of basis state x."""
    out: Dict[int, float] = defaultdict(float)
    for amps, m in one_cycle_branches(x, N, t, bc):
        sc = 2.0 ** (-m)
        for y, (a, b) in amps.items():
            out[y] += ((a + b * np.sqrt(2.0)) * sc) ** 2
    return dict(out)


def absorption(rule: int, N: int, bc: str = BC):
    """(decomposition, per-state absorption probabilities, terminal comp ids)."""
    t = wolfram_to_tuple(rule)
    d = decompose(rule, N, bc)
    tc = [c for c in range(len(d["members"])) if d["terminal"][c]]
    lab = {c: i for i, c in enumerate(tc)}
    n = 1 << N
    P = [transition_row(x, N, t, bc) for x in range(n)]
    assert max(abs(sum(r.values()) - 1.0) for r in P) < 1e-9, "rows must be stochastic"
    A = np.zeros((n, len(tc)))
    absorbed = np.zeros(n, bool)
    for c in tc:
        for x in d["members"][c]:
            A[x, lab[c]] = 1.0
            absorbed[x] = True
    for _ in range(500):
        B = A.copy()
        for x in range(n):
            if absorbed[x]:
                continue
            v = np.zeros(len(tc))
            for y, p in P[x].items():
                v += p * A[y]
            B[x] = v
        if np.abs(B - A).max() < 1e-14:
            A = B
            break
        A = B
    return d, A, tc, lab, P


# ------------------------------------------------- exact trajectory dynamics

INV_SQRT2 = 1.0 / np.sqrt(2.0)


class Trajectory:
    """State-vector quantum trajectory: the per-site two-Kraus split of
    core.cycle._split_site, sampled.  `order` selects the brickwork convention
    ('even_first' matches the Tier-1 engine, 'odd_first' matches pyqca)."""

    def __init__(self, rule: int, N: int, bc: str = BC, order: str = "even_first"):
        self.rule, self.N, self.bc, self.order = rule, N, bc, order
        self.t = wolfram_to_tuple(rule)
        even, odd = list(range(0, N, 2)), list(range(1, N, 2))
        self.layers = [even, odd] if order == "even_first" else [odd, even]
        self.masks: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        xs = np.arange(1 << N, dtype=np.int64)
        for i in range(N):
            if bc == "pbc":
                lpos, rpos = (i - 1) % N, (i + 1) % N
            else:
                lpos = i - 1 if i > 0 else -1
                rpos = i + 1 if i < N - 1 else -1
            left = ((xs >> lpos) & 1) if lpos >= 0 else 0
            right = ((xs >> rpos) & 1) if rpos >= 0 else 0
            # the pattern does not depend on bit i, so keep the bit-i=0 slice
            pat = (2 * left + right).astype(np.uint8).reshape(-1, 2, 1 << i)[:, 0, :].copy()
            m = {}
            for key in "VDE":
                mm = np.zeros(pat.shape, bool)
                for q in range(4):
                    if self.t[q] == key:
                        mm |= pat == q
                m[key] = mm if mm.any() else None
            self.masks[i] = m
        del xs

    def _site(self, psi, i, rng):
        v = psi.reshape(-1, 2, 1 << i)
        a, b = v[:, 0, :], v[:, 1, :]
        m = self.masks[i]
        vm, dm, em = m["V"], m["D"], m["E"]
        if vm is not None:
            a0, b0 = a[vm], b[vm]
            a[vm] = (a0 + b0) * INV_SQRT2
            b[vm] = (a0 - b0) * INV_SQRT2
        if dm is None and em is None:
            return psi
        pj = 0.0
        if dm is not None:
            pj += float(np.vdot(b[dm], b[dm]).real)
        if em is not None:
            pj += float(np.vdot(a[em], a[em]).real)
        if rng.random() < pj:
            na, nb = np.zeros_like(a), np.zeros_like(b)
            if dm is not None:
                na[dm] = b[dm]
            if em is not None:
                nb[em] = a[em]
            v[:, 0, :], v[:, 1, :] = na, nb
            psi /= np.sqrt(pj)
        else:
            if dm is not None:
                b[dm] = 0.0
            if em is not None:
                a[em] = 0.0
            psi /= np.linalg.norm(psi)
        return psi

    def step(self, psi, rng):
        for layer in self.layers:
            for i in layer:
                psi = self._site(psi, i, rng)
        return psi

    def jump_budget(self, psi):
        """(max jump probability, max norm removed by the projector) over one
        period -- both zero exactly iff the state sits in a DFS."""
        psi = psi.copy()
        maxpj = maxdn = 0.0
        for layer in self.layers:
            for i in layer:
                v = psi.reshape(-1, 2, 1 << i)
                a, b = v[:, 0, :], v[:, 1, :]
                m = self.masks[i]
                vm, dm, em = m["V"], m["D"], m["E"]
                if vm is not None:
                    a0, b0 = a[vm], b[vm]
                    a[vm] = (a0 + b0) * INV_SQRT2
                    b[vm] = (a0 - b0) * INV_SQRT2
                if dm is None and em is None:
                    continue
                pj = 0.0
                if dm is not None:
                    pj += float(np.vdot(b[dm], b[dm]).real)
                if em is not None:
                    pj += float(np.vdot(a[em], a[em]).real)
                maxpj = max(maxpj, pj)
                if dm is not None:
                    b[dm] = 0.0
                if em is not None:
                    a[em] = 0.0
                nrm = np.linalg.norm(psi)
                maxdn = max(maxdn, abs(1.0 - nrm))
                psi /= nrm
        return maxpj, maxdn

    def occupation(self, psi):
        p = psi ** 2 if psi.dtype.kind == "f" else np.abs(psi) ** 2
        return np.array([p.reshape(-1, 2, 1 << i)[:, 1, :].sum()
                         for i in range(self.N)])

    def entropy(self, psi, cut):
        s = np.linalg.svd(psi.reshape(1 << (self.N - cut), 1 << cut),
                          compute_uv=False)
        s = s[s > 1e-12] ** 2
        return float(-(s * np.log(s)).sum())


def basis_state(N: int, x: int) -> np.ndarray:
    psi = np.zeros(1 << N)
    psi[x] = 1.0
    return psi


# ------------------------------------------------------------------ the pass

def exact_pass(rules=ALL_RULES, sizes=range(6, 15), bc: str = BC) -> Dict:
    """Everything that can be settled by enumerating the whole 2^N space."""
    rows = []
    for rule in rules:
        for N in sizes:
            d = decompose(rule, N, bc)
            S, members, term = d["S"], d["members"], d["terminal"]
            tc = [c for c in range(len(members)) if term[c]]
            nontrivial = [c for c in tc if len(members[c]) > 1]
            words = {c: attractor_word(members[c], N) for c in tc}
            dens = []
            for c in nontrivial:
                m = set(members[c])
                dens.append(sum(len(m & set(S[x])) for x in m) / len(m) ** 2)
            byw = defaultdict(set)
            for c in tc:
                byw[d["wcc"][members[c][0]]].add(c)
            row = dict(
                rule=rule, N=N, bc=bc,
                n_wcc=len(set(d["wcc"])), n_terminal=len(tc),
                n_nontrivial=len(nontrivial),
                dim_max=max(len(members[c]) for c in tc),
                subcubes=sum(is_subcube(members[c], words[c]) for c in tc),
                density_min=min(dens) if dens else None,
                density_max=max(dens) if dens else None,
                wcc_with_multiple=sum(1 for s in byw.values() if len(s) > 1),
                max_terminal_per_wcc=max(len(s) for s in byw.values()),
            )
            if rule in FAMILY_OF:
                row["grammar_match"] = (set(words.values())
                                        == predicted_words(rule, N))
            rows.append(row)
    return {"exact": rows}


def branch_pass(rules=(29, 71), sizes=range(8, 13), bc: str = BC) -> Dict:
    """The 29/71 branch: how many coins, how fair, how long undecided."""
    rows = []
    for rule in rules:
        for N in sizes:
            d, A, tc, lab, P = absorption(rule, N, bc)
            split = [x for x in range(1 << N) if (A[x] > 1e-12).sum() >= 2]
            mult = defaultdict(int)
            probs = defaultdict(int)
            for x in split:
                nz = A[x][A[x] > 1e-12]
                mult[int(len(nz))] += 1
                probs[tuple(sorted(round(float(v), 6) for v in nz))] += 1
            undec = set(split)
            depth = 0
            while undec:
                nxt = {x for x in undec if any(y in undec for y in P[x])}
                depth += 1
                if nxt == undec:
                    depth = -1  # never resolves; would be a real surprise
                    break
                undec = nxt
            rows.append(dict(
                rule=rule, N=N, n_wcc=len(set(d["wcc"])), n_terminal=len(tc),
                n_split=len(split), split_fraction=len(split) / (1 << N),
                multiplicities={str(k): v for k, v in sorted(mult.items())},
                probability_vectors={str(list(k)): v for k, v in probs.items()},
                periods_to_decide=depth))
    return {"branch": rows}


def dfs_pass(rules=ALL_RULES, N: int = 11, ntop: int = 3, nrand: int = 4,
             periods: int = 25, bc: str = BC) -> Dict:
    """Is each terminal SCC a decoherence-free subspace?  Evolve random states
    supported on it and watch the jump probability and the projector."""
    rows = []
    for rule in rules:
        d = decompose(rule, N, bc)
        tc = sorted((c for c in range(len(d["members"])) if d["terminal"][c]),
                    key=lambda c: -len(d["members"][c]))
        tr = Trajectory(rule, N, bc)
        for c in tc[:ntop]:
            m = d["members"][c]
            if len(m) < 2:
                continue
            pj = dn = 0.0
            inside = True
            for k in range(nrand):
                rng = np.random.default_rng(k)
                psi = np.zeros(1 << N)
                v = rng.normal(size=len(m))
                psi[list(m)] = v / np.linalg.norm(v)
                for _ in range(periods):
                    p, q = tr.jump_budget(psi)
                    pj, dn = max(pj, p), max(dn, q)
                    psi = tr.step(psi, np.random.default_rng(0))
                    sup = set(np.nonzero(np.abs(psi) > 1e-11)[0].tolist())
                    inside &= sup <= set(m)
            rows.append(dict(rule=rule, N=N, dim=len(m), max_p_jump=pj,
                             max_norm_loss=dn, support_invariant=bool(inside)))
    return {"dfs": rows}


def traj_pass(rules=ALL_RULES, sizes=(9, 11, 13, 15), nsamp: int = 24,
              ens: int = 200, bc: str = BC) -> Dict:
    """Entanglement inside the attractor, and the period-2 signal single-shot
    against ensemble-averaged."""
    rows = []
    for rule in rules:
        for N in sizes:
            tr = Trajectory(rule, N, bc)
            T = 6 * N
            ent, p2 = [], []
            for k in range(nsamp):
                rng = np.random.default_rng(900 + k)
                psi = basis_state(N, int(rng.integers(1 << N)))
                for _ in range(T):
                    psi = tr.step(psi, rng)
                v, o = [], []
                for _ in range(8):
                    v.append(tr.entropy(psi, N // 2))
                    o.append(tr.occupation(psi))
                    psi = tr.step(psi, rng)
                ent.append(float(np.mean(v)))
                o = np.array(o)
                p2.append(float(np.abs(o[0::2].mean(0) - o[1::2].mean(0)).max()))
            acc = np.zeros((9, N))
            for k in range(ens):
                rng = np.random.default_rng(5000 + k)
                psi = basis_state(N, int(rng.integers(1 << N)))
                for _ in range(T):
                    psi = tr.step(psi, rng)
                for j in range(9):
                    acc[j] += tr.occupation(psi)
                    psi = tr.step(psi, rng)
            acc /= ens
            rows.append(dict(
                rule=rule, N=N, S_mid=float(np.mean(ent)),
                S_mid_sem=float(np.std(ent) / np.sqrt(nsamp)),
                period2_single=float(np.mean(p2)),
                period2_ensemble=float(np.abs(acc[0::2].mean(0)
                                              - acc[1::2].mean(0)).max())))
    return {"trajectory": rows}


def rebuild(bc: str = BC, quick: bool = False) -> Dict:
    data: Dict = {"bc": bc}
    data.update(exact_pass(sizes=range(6, 13 if quick else 15), bc=bc))
    data.update(branch_pass(sizes=range(8, 11 if quick else 13), bc=bc))
    data.update(dfs_pass(bc=bc))
    data.update(traj_pass(sizes=(9, 11, 13) if quick else (9, 11, 13, 15),
                          nsamp=12 if quick else 24,
                          ens=60 if quick else 200, bc=bc))
    os.makedirs(sectors.ANALYTICS, exist_ok=True)
    with open(OUT_JSON.format(bc=bc), "w") as f:
        json.dump(data, f, indent=1)
    return data


def load(bc: str = BC) -> Dict:
    with open(OUT_JSON.format(bc=bc)) as f:
        return json.load(f)


def summarise(data: Dict) -> None:
    print("=== attractor structure (obc0) ===")
    print(f"{'rule':>5} {'N':>3} {'n_wcc':>6} {'n_att':>6} {'dim_max':>8} "
          f"{'subcube':>9} {'density':>14} {'grammar':>8} {'multi-WCC':>10}")
    for r in data["exact"]:
        if r["N"] not in (10, 14):
            continue
        dens = ("--" if r["density_min"] is None
                else f"{r['density_min']:.2f}-{r['density_max']:.2f}")
        print(f"{r['rule']:>5} {r['N']:>3} {r['n_wcc']:>6} {r['n_terminal']:>6} "
              f"{r['dim_max']:>8} {r['subcubes']:>4}/{r['n_terminal']:<4} "
              f"{dens:>14} {str(r.get('grammar_match','n/a')):>8} "
              f"{r['wcc_with_multiple']:>10}")
    print("\n=== 29/71 branch ===")
    for r in data["branch"]:
        print(f"  R{r['rule']} N={r['N']}: {r['n_split']} split states, "
              f"multiplicities {r['multiplicities']}, "
              f"decided in {r['periods_to_decide']} period(s), "
              f"probabilities {list(r['probability_vectors'])}")
    print("\n=== decoherence-free test ===")
    worst = max(r["max_p_jump"] for r in data["dfs"])
    allin = all(r["support_invariant"] for r in data["dfs"])
    print(f"  {len(data['dfs'])} attractors: max P(jump) = {worst:.2e}, "
          f"support invariant everywhere: {allin}")
    print("\n=== trajectory ===")
    print(f"{'rule':>5} {'N':>3} {'S_mid':>10} {'period2 single':>15} "
          f"{'period2 ensemble':>17}")
    for r in data["trajectory"]:
        print(f"{r['rule']:>5} {r['N']:>3} {r['S_mid']:>10.4f} "
              f"{r['period2_single']:>15.3f} {r['period2_ensemble']:>17.3f}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bc", default=BC, choices=["obc0", "pbc"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="smaller sizes and fewer samples, for the test suite")
    a = ap.parse_args(argv)
    data = rebuild(a.bc, quick=a.quick) if a.rebuild else load(a.bc)
    summarise(data)


if __name__ == "__main__":
    main()
