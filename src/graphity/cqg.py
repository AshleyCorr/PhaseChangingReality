"""2D combinatorial quantum gravity (CQG): Monte Carlo on incompressible graphs.

Model, as we read it from the sources (keys in REFERENCES.bib). Q1-Q3 were
checked against the text of [T25] and [KTB19] on 2026-09-19; see ASSUMPTIONS.md.

* Configuration space: 4-regular graphs on N vertices in which short cycles
  (triangles, squares, pentagons) share at most one edge ("incompressible" /
  hard-core condition) [T25 Def. 2 and Fig. 1; T22; T24 Eq. (1)].
* Energy: total Ollivier-Ricci curvature, H = -2D sum_i sum_{j~i} kappa(ij)
  [T25 Eq. (21)], with edge curvature
  kappa = T/2D - [1-(2+T+S)/2D]_+ - [1-(2+T+S+P)/2D]_+  [T25 Eq. (8)].
* Simplification used in the published simulations: bipartite graphs only, so
  no triangles or pentagons can occur [T25; T24; KTB19 Sec. 4].
* Weight exp(-H/g) with g the dimensionless coupling, playing the role of
  temperature [T25 Eq. (20)].

ASSUMPTION Q1 (energy; [T25 Eqs. (8), (21)] at D = 2 with T = P = 0):
    kappa(e) = -(2 - S_e)_+ / 2,
    H = 4 sum_e (2 - S_e)_+  =  16 (N - S) + 4 X,     X = sum_e (S_e - 2)_+ .
  S is the number of squares, S_e the number on edge e, and X counts the
  surplus squares on edges that carry more than two. 16 (N - S) is the "global"
  term, 4 X the "local" one [T25 Eq. (22)]. Checks: a random graph (S ~ 0)
  has H = 16N; the flat torus (S = N, X = 0) has H = 0; the 4-cube (N = 16,
  S = 24, every S_e = 3, so X = 32) has H = -128 + 128 = 0, the degeneracy
  stated in [T25].

THE LAMBDA FAMILY (VISION, first knob; the interpolation is ours):
    H = 16 (N - S) + 4 lam X .
  lam = 0   global term only. Published: first order, hypercubes [T25; GV21].
  lam = 1   the full Ollivier curvature [T25 Eq. (22)].
  cap = 2   edges with S_e > 2 forbidden outright, which is lam -> infinity.
            This is the model simulated in [KTB19 Sec. 4] (Q3). X is then
            always zero and H = 16 (N - S) whatever lam is.

ASSUMPTION Q2 (hard-core rule): in a bipartite graph two distinct squares can
  share two edges only if the edges are adjacent, which makes a K_{2,3}. So
  "incompressible" is equivalent to "no two vertices have more than two common
  neighbours". Confirmed: K_{2,3} is excluded subgraph (b) of [T25 Fig. 1].
  Consequence used below: an edge carries at most 2D - 1 = 3 squares. (A square
  u-v-a-b on edge (u,v) uses a common neighbour b of u and a other than v, and
  there is at most one.) Hence "no cap" and "cap = 3" are the same thing.

ASSUMPTION Q3 (the cap is optional): cap = CAP = 2 gives the [KTB19 Sec. 4]
  model, cap = NO_CAP the space of [T25]. Without the cap the 4x4 torus, which
  is the 4-cube, is a valid state, so torus() allows a side of 4 there.

ASSUMPTION Q4 (move and acceptance): bipartite edge switch
  (u1,v1),(u2,v2) -> (u1,v2),(u2,v1). [KTB19 Sec. 4] says "edge switches";
  details not given there. The proposal is symmetric: two edges are chosen
  uniformly, and the reverse move picks the two new edges with the same
  probability. Moves leading outside the configuration space are rejected.
  Two acceptance rules are offered, with a = exp(-dH / g):
      Metropolis   p = min(1, a)                    [NB99]
      Glauber      p = 1 / (1 + 1/a)                [T25 Eq. (28)]
  Both satisfy p(dH) / p(-dH) = a, which with a symmetric proposal is detailed
  balance, so both sample exp(-H/g); they differ only in how fast. Ergodicity
  inside the constrained space is proved by exhaustive listing for N <= 18 and
  unproven beyond (Q9; small_graphs.py). Connectedness is not enforced.

  How dH is obtained exactly. dS: squares lost are those through the removed
  edges, squares gained those through the added ones, and no square contains
  both removed or both added edges (it would need the other pair as well). dX:
  S_e can change only on an edge of a lost or a gained square. Those edges are
  listed, X is summed over the list in the new graph, the move is switched
  back, X is summed over the same list in the old graph, and the move is
  switched forward again. Edges absent from a graph contribute nothing.

ASSUMPTION Q5 (start state): an lx x ly periodic square lattice, both sides
  even and >= 6 under the cap (>= 4 without it), melted by running at infinite
  temperature. Ours.
"""
import numpy as np
from numba import njit

from graphity.connectivity import connectivity
from graphity.squares import has_edge, squares_on_edge      # moved there so connectivity.py can share them

CAP = 2                    # squares allowed per edge in the capped model, 2D - 2      (Q3)
NO_CAP = 3                 # the hard-core rule alone already stops at 2D - 1 = 3      (Q2)
MAX_CODEGREE = 2           # no K_{2,3}                                                 (Q2)
ENERGY_PER_SQUARE = 16.0   # global term 16 (N - S)                                     (Q1)
ENERGY_PER_SURPLUS = 4.0   # local term 4 lam X                                         (Q1)


def torus(lx: int, ly: int = None, cap: int = CAP):
    """Adjacency (N,4) and bipartition (0/1 per vertex) of the lx x ly torus.

    torus(L) is the L x L torus. Both sides must be even, so that the lattice is
    bipartite. Under the cap they must also be >= 6: a side of 4 closes a 4-cycle
    around the torus, which puts three squares on half the edges (checked by brute
    force: 4 x 6 has S = 1.25 N). Without the cap a side of 4 is allowed; the
    4 x 4 torus is then the 4-cube of Q1.
    """
    if ly is None:
        ly = lx
    shortest = 6 if cap < NO_CAP else 4
    if lx % 2 or ly % 2 or min(lx, ly) < shortest:
        raise ValueError(f"both sides must be even and >= {shortest} (under the cap a side "
                         "of 4 wraps into extra squares; the 4x4 torus is the 4-cube, see Q3)")
    n = lx * ly
    adj = np.empty((n, 4), dtype=np.int64)
    part = np.empty(n, dtype=np.int64)
    for x in range(lx):
        for y in range(ly):
            i = x * ly + y
            adj[i] = [((x + 1) % lx) * ly + y, ((x - 1) % lx) * ly + y,
                      x * ly + (y + 1) % ly, x * ly + (y - 1) % ly]
            part[i] = (x + y) % 2
    return adj, part


@njit(cache=True)
def codegree(adj, x, y):
    c = 0
    for i in range(4):
        a = adj[x, i]
        if a >= 0 and has_edge(adj, y, a):
            c += 1
    return c


@njit(cache=True)
def total_squares(adj):
    s = 0
    for u in range(adj.shape[0]):
        for k in range(4):
            v = adj[u, k]
            if v > u:
                s += squares_on_edge(adj, u, v)
    return s // 4          # every square is seen from each of its 4 edges


@njit(cache=True)
def surplus(adj):
    """X = sum over edges of (S_e - 2)_+, computed from scratch."""
    x = 0
    for u in range(adj.shape[0]):
        for k in range(4):
            v = adj[u, k]
            if v > u:
                s_e = squares_on_edge(adj, u, v)
                if s_e > 2:
                    x += s_e - 2
    return x


@njit(cache=True)
def hamiltonian(adj, lam):
    """H = 16 (N - S) + 4 lam X, computed from scratch (Q1)."""
    return (ENERGY_PER_SQUARE * (adj.shape[0] - total_squares(adj))
            + ENERGY_PER_SURPLUS * lam * surplus(adj))


@njit(cache=True)
def is_valid(adj, cap=CAP):
    """4-regular, simple, codegree <= 2 everywhere, squares per edge <= cap."""
    n = adj.shape[0]
    for u in range(n):
        for k in range(4):
            v = adj[u, k]
            if v < 0 or v == u or not has_edge(adj, v, u):
                return False
            for m in range(k + 1, 4):
                if adj[u, m] == v:
                    return False
            if squares_on_edge(adj, u, v) > cap:
                return False
            for m in range(4):          # pairs at distance two through v
                w = adj[v, m]
                if w != u and codegree(adj, u, w) > MAX_CODEGREE:
                    return False
    return True


@njit(cache=True)
def _replace(adj, u, old, new):
    for k in range(4):
        if adj[u, k] == old:
            adj[u, k] = new
            return


@njit(cache=True)
def _switch(adj, u1, v1, u2, v2):
    """(u1,v1),(u2,v2) -> (u1,v2),(u2,v1); each new neighbour takes the old one's slot.

    _switch(adj, u1, v2, u2, v1) is the exact reverse, slots included.
    """
    _replace(adj, u1, v1, v2)
    _replace(adj, v1, u1, u2)
    _replace(adj, u2, v2, v1)
    _replace(adj, v2, u2, u1)


@njit(cache=True)
def _new_edge_ok(adj, p, q, cap):
    """Constraints that adding edge (p, q) could have broken."""
    for k in range(4):
        x = adj[q, k]
        if x != p and codegree(adj, p, x) > MAX_CODEGREE:
            return False
        y = adj[p, k]
        if y != q and codegree(adj, q, y) > MAX_CODEGREE:
            return False
    if squares_on_edge(adj, p, q) > cap:
        return False
    for i in range(4):                  # other edges of every square through (p, q)
        a = adj[q, i]
        if a == p:
            continue
        for j in range(4):
            b = adj[p, j]
            if b == q:
                continue
            if has_edge(adj, a, b):
                if (squares_on_edge(adj, q, a) > cap or squares_on_edge(adj, a, b) > cap
                        or squares_on_edge(adj, b, p) > cap):
                    return False
    return True


@njit(cache=True)
def _note_edge(edges, n, a, b):
    """Add edge (a, b) to the first n rows of `edges` unless it is there; return the new n."""
    lo, hi = (a, b) if a < b else (b, a)
    for i in range(n):
        if edges[i, 0] == lo and edges[i, 1] == hi:
            return n
    edges[n, 0] = lo
    edges[n, 1] = hi
    return n + 1


@njit(cache=True)
def _note_square_edges(adj, p, q, edges, n):
    """Note (p, q) and the other three edges of every square through it."""
    n = _note_edge(edges, n, p, q)
    for i in range(4):
        a = adj[q, i]
        if a < 0 or a == p:
            continue
        for j in range(4):
            b = adj[p, j]
            if b < 0 or b == q:
                continue
            if has_edge(adj, a, b):
                n = _note_edge(edges, n, q, a)
                n = _note_edge(edges, n, a, b)
                n = _note_edge(edges, n, b, p)
    return n


@njit(cache=True)
def _surplus_on(adj, edges, n):
    """Sum of (S_e - 2)_+ over those of the noted edges that exist in adj."""
    x = 0
    for i in range(n):
        a, b = edges[i, 0], edges[i, 1]
        if has_edge(adj, a, b):
            s_e = squares_on_edge(adj, a, b)
            if s_e > 2:
                x += s_e - 2
    return x


@njit(cache=True)
def run_chain(adj, side_u, inv_g, n_equil, n_meas, seed, lam=1.0, cap=CAP, glauber=False, conn=None, moves=None):
    """Markov chain at coupling g = 1 / inv_g. Modifies adj in place.

    side_u  : array of the vertices in one half of the bipartition.
    lam     : strength of the local term. Irrelevant under the cap, where X = 0.
    cap     : CAP (2) or NO_CAP.
    glauber : False for Metropolis, True for the rule of [T25 Eq. (28)].
    conn    : optional (n_meas, 4) integer array. If given, row i receives, after
              measurement sweep i, the four numbers of connectivity.connectivity:
              pieces, size of the largest, vertices in baby universes, 4-cubes.
              Looking uses no random numbers, so the chain is the same either way.
    moves   : optional (N,) integer array (T101, option A of VISION Update 156's clock addendum). If given,
              every accepted switch adds one to each of its four points, over every sweep of the call,
              equilibration included. A point's count is its own clock beside the sweep count. Counting
              uses no random numbers, so the chain is the same either way.
    seed    : seeds the random stream. **seed < 0 carries on the stream from the
              previous call instead.** Anything that calls this repeatedly in short
              blocks must use that, because re-seeding often makes the chain worse:
              measured on a flat-histogram walk with exactly known weights, the
              histogram went from 1.5 % uneven in one call to 24 % uneven in a
              thousand re-seeded blocks of the same total length (ASSUMPTIONS Q14).
              A single long call is unaffected, which is what every script here does.
    Returns (S after each measurement sweep, X after each, acceptance rate).
    One sweep = 2N attempted switches (one per edge).
    """
    if seed >= 0:
        np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    track_x = cap > CAP                 # under the cap X is identically zero
    edges = np.empty((64, 2), dtype=np.int64)   # at most 4 edges x (1 + 3 squares x 3) = 40 rows
    s = total_squares(adj)
    x = surplus(adj)
    out_s = np.zeros(n_meas, dtype=np.int64)
    out_x = np.zeros(n_meas, dtype=np.int64)
    attempted = 0
    accepted = 0
    for sweep in range(n_equil + n_meas):
        for _ in range(2 * n):
            attempted += 1
            u1 = side_u[np.random.randint(0, nu)]
            v1 = adj[u1, np.random.randint(0, 4)]
            u2 = side_u[np.random.randint(0, nu)]
            v2 = adj[u2, np.random.randint(0, 4)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            n_edges = 0
            if track_x:                 # edges of the squares about to be lost
                n_edges = _note_square_edges(adj, u1, v1, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v2, edges, n_edges)
            lost = squares_on_edge(adj, u1, v1)
            _replace(adj, u1, v1, -1)
            _replace(adj, v1, u1, -1)
            lost += squares_on_edge(adj, u2, v2)
            _replace(adj, u2, v2, -1)
            _replace(adj, v2, u2, -1)
            _replace(adj, u1, -1, v2)
            _replace(adj, v2, -1, u1)
            gained = squares_on_edge(adj, u1, v2)
            _replace(adj, u2, -1, v1)
            _replace(adj, v1, -1, u2)
            gained += squares_on_edge(adj, u2, v1)
            d_s = gained - lost
            d_x = 0
            ok = _new_edge_ok(adj, u1, v2, cap) and _new_edge_ok(adj, u2, v1, cap)
            if ok and track_x:          # the new graph is valid, so the list stays short
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                after = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v2, u2, v1)            # back to the old graph
                before = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v1, u2, v2)            # forward again
                d_x = after - before
            if ok:
                d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
                if glauber:
                    if np.random.random() >= 1.0 / (1.0 + np.exp(inv_g * d_h)):
                        ok = False
                elif d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                    ok = False
            if ok:
                s += d_s
                x += d_x
                accepted += 1
                if moves is not None:
                    moves[u1] += 1
                    moves[v1] += 1
                    moves[u2] += 1
                    moves[v2] += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        if sweep >= n_equil:
            out_s[sweep - n_equil] = s
            out_x[sweep - n_equil] = x
            if conn is not None:
                pieces, largest, in_babies, cubes = connectivity(adj)
                conn[sweep - n_equil, 0] = pieces
                conn[sweep - n_equil, 1] = largest
                conn[sweep - n_equil, 2] = in_babies
                conn[sweep - n_equil, 3] = cubes
    return out_s, out_x, accepted / max(attempted, 1)
