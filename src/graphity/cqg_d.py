"""The model graph family at any dimension D: Monte Carlo on 2D-regular bipartite graphs, with the knob lambda.

Naming (the model's author's condition, 24 September 2026): only lambda = 1 is combinatorial quantum gravity (CQG).
Every other lambda is our family around it, and results there are never written up as CQG.

The same model and the same chain as cqg.py, with the number of links per point read from the array instead of
fixed at 4. Written for the six-link (D = 3) model of VISION Update 22. cqg.py is left untouched, so every 2D
result on the record is safe by construction; tests/test_cqg_d.py checks that this module, given four links, makes
exactly the same chain as cqg.run_chain without the cap, draw for draw.

Model at dimension D (ASSUMPTIONS Q21; Q1 and Q2 at D = 2):
* Graphs: 2D-regular and bipartite, obeying the hard-core rule ([KTB19] Sec. 2.2 and [T22] Sec. II: the same rule
  at every D), which on bipartite graphs is "no two points share more than two neighbours". Q2's argument does not
  use the degree (ours): two squares u-v-a-b and u-v-a'-b on edge (u, v) that share a second edge share (v, a) or
  (u, b), and then u and a (or v and b) have three common neighbours; conversely three common neighbours p, q, r of
  x and y give the squares x-p-y-q and x-p-y-r, which share two edges. An edge then carries at most 2D - 1 squares.
* Energy (ours, derived from [T25] Eqs. (8), (21), (22) exactly as Q1):
      H = 16 (D(D-1)/2 N - S) + 4 lam X,   X = sum_e (S_e - (2D - 2))_+ .
  The flat D-torus (S = D(D-1)/2 N, every S_e = 2D - 2) has H = 0 at every lam.
* No cap. The cap of [KTB19] Sec. 4 is disputed by the model's author (VISION Update 18) and is not offered here.

Move and acceptance: cqg.py's, unchanged (Q4). A bipartite edge switch (u1,v1),(u2,v2) -> (u1,v2),(u2,v1), with
u1, u2 drawn from one side and v1, v2 from their neighbour slots; the proposal is symmetric; Metropolis or Glauber
acceptance, both detailed-balanced for exp(-H/g). One sweep is 2N attempts, as in cqg.py. dS and dX are obtained
exactly as in cqg.py: squares lost through the removed edges, gained through the added ones, and X re-summed over
the edges of every lost or gained square, before and after.
"""
import numpy as np
from numba import njit

MAX_CODEGREE = 2           # no K_{2,3}                                   (Q2)
ENERGY_PER_SQUARE = 16.0   # global term 16 (D(D-1)/2 N - S)              (Q1, Q21)
ENERGY_PER_SURPLUS = 4.0   # local term 4 lam X                           (Q1, Q21)


def torus(dims):
    """Adjacency (N, 2D) and bipartition of the torus C_{dims[0]} x ... x C_{dims[D-1]}.

    Every side even (bipartite) and at least 4. A side of 4 is a curled direction: its wrap-around 4-cycles put
    extra squares on the edges along it. At D = 2 the vertex numbering and the neighbour slots are cqg.torus's.
    """
    dims = tuple(int(x) for x in dims)
    if any(x % 2 or x < 4 for x in dims):
        raise ValueError("every side must be even and at least 4")
    d = len(dims)
    coords = np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims)).T
    strides = np.array([int(np.prod(dims[a + 1:])) for a in range(d)], dtype=np.int64)
    adj = np.empty((len(coords), 2 * d), dtype=np.int64)
    for i, c in enumerate(coords):
        k = 0
        for axis in range(d):
            for step in (1, -1):
                nb = c.copy()
                nb[axis] = (nb[axis] + step) % dims[axis]
                adj[i, k] = int(np.dot(nb, strides))
                k += 1
    part = (coords.sum(axis=1) % 2).astype(np.int64)
    return adj, part


def circulant(m, offsets):
    """A 2D-regular bipartite graph on 2m points: u_i ~ v_{(i + s) mod m} for each offset s.

    With offsets whose pairwise differences are all distinct mod m, no two points share more than one neighbour,
    so the graph is valid and has no squares at all: a start with nothing ordered in it, at any even N.
    """
    offsets = [int(s) % m for s in offsets]
    deg = len(offsets)
    adj = np.empty((2 * m, deg), dtype=np.int64)
    for i in range(m):
        for k, s in enumerate(offsets):
            adj[i, k] = m + (i + s) % m
    for j in range(m):
        for k, s in enumerate(offsets):
            adj[m + j, k] = (j - s) % m
    part = np.array([0] * m + [1] * m, dtype=np.int64)
    return adj, part


@njit(cache=True)
def has_edge(adj, a, b):
    for k in range(adj.shape[1]):
        if adj[a, k] == b:
            return True
    return False


@njit(cache=True)
def squares_on_edge(adj, u, v):
    """Number of 4-cycles u - v - a - b - u containing the edge (u, v)."""
    deg = adj.shape[1]
    count = 0
    for i in range(deg):
        a = adj[v, i]
        if a < 0 or a == u:
            continue
        for j in range(deg):
            b = adj[u, j]
            if b < 0 or b == v:
                continue
            if has_edge(adj, a, b):
                count += 1
    return count


@njit(cache=True)
def codegree(adj, x, y):
    c = 0
    for i in range(adj.shape[1]):
        a = adj[x, i]
        if a >= 0 and has_edge(adj, y, a):
            c += 1
    return c


@njit(cache=True)
def total_squares(adj):
    s = 0
    for u in range(adj.shape[0]):
        for k in range(adj.shape[1]):
            v = adj[u, k]
            if v > u:
                s += squares_on_edge(adj, u, v)
    return s // 4


@njit(cache=True)
def surplus(adj):
    """X = sum over edges of (S_e - (2D - 2))_+, computed from scratch."""
    sat = adj.shape[1] - 2
    x = 0
    for u in range(adj.shape[0]):
        for k in range(adj.shape[1]):
            v = adj[u, k]
            if v > u:
                s_e = squares_on_edge(adj, u, v)
                if s_e > sat:
                    x += s_e - sat
    return x


@njit(cache=True)
def hamiltonian(adj, lam):
    """H = 16 (D(D-1)/2 N - S) + 4 lam X, computed from scratch (Q21)."""
    dim = adj.shape[1] // 2
    return (ENERGY_PER_SQUARE * (dim * (dim - 1) / 2.0 * adj.shape[0] - total_squares(adj))
            + ENERGY_PER_SURPLUS * lam * surplus(adj))


@njit(cache=True)
def is_valid(adj):
    """Regular, simple, symmetric, and no two points share more than two neighbours."""
    n, deg = adj.shape
    for u in range(n):
        for k in range(deg):
            v = adj[u, k]
            if v < 0 or v == u or not has_edge(adj, v, u):
                return False
            for m in range(k + 1, deg):
                if adj[u, m] == v:
                    return False
            for m in range(deg):        # pairs at distance two through v
                w = adj[v, m]
                if w != u and codegree(adj, u, w) > MAX_CODEGREE:
                    return False
    return True


@njit(cache=True)
def _replace(adj, u, old, new):
    for k in range(adj.shape[1]):
        if adj[u, k] == old:
            adj[u, k] = new
            return


@njit(cache=True)
def _switch(adj, u1, v1, u2, v2):
    """(u1,v1),(u2,v2) -> (u1,v2),(u2,v1); _switch(adj, u1, v2, u2, v1) is the exact reverse, slots included."""
    _replace(adj, u1, v1, v2)
    _replace(adj, v1, u1, u2)
    _replace(adj, u2, v2, v1)
    _replace(adj, v2, u2, u1)


@njit(cache=True)
def _new_edge_ok(adj, p, q):
    """The hard-core rule, for the pairs that adding edge (p, q) could have broken."""
    for k in range(adj.shape[1]):
        x = adj[q, k]
        if x != p and codegree(adj, p, x) > MAX_CODEGREE:
            return False
        y = adj[p, k]
        if y != q and codegree(adj, q, y) > MAX_CODEGREE:
            return False
    return True


@njit(cache=True)
def _note_edge(edges, n, a, b):
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
    deg = adj.shape[1]
    n = _note_edge(edges, n, p, q)
    for i in range(deg):
        a = adj[q, i]
        if a < 0 or a == p:
            continue
        for j in range(deg):
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
    sat = adj.shape[1] - 2
    x = 0
    for i in range(n):
        a, b = edges[i, 0], edges[i, 1]
        if has_edge(adj, a, b):
            s_e = squares_on_edge(adj, a, b)
            if s_e > sat:
                x += s_e - sat
    return x


@njit(cache=True)
def run_chain(adj, side_u, inv_g, n_equil, n_meas, seed, lam=1.0, glauber=False):
    """Markov chain at coupling g = 1 / inv_g, at the dimension of adj (adj.shape[1] = 2D). Modifies adj in place.

    seed < 0 carries on the stream from the previous call (as cqg.run_chain; ASSUMPTIONS Q14).
    Returns (S after each measurement sweep, X after each, acceptance rate). One sweep = 2N attempted switches.
    """
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    # at most 4 edges, each with (2D - 1) squares of 3 further edges: 4 (1 + 3 (2D - 1)) rows
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
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
            v1 = adj[u1, np.random.randint(0, deg)]
            u2 = side_u[np.random.randint(0, nu)]
            v2 = adj[u2, np.random.randint(0, deg)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            n_edges = 0
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
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                after = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v2, u2, v1)            # back to the old graph
                before = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v1, u2, v2)            # forward again
                d_x = after - before
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
            else:
                _switch(adj, u1, v2, u2, v1)
        if sweep >= n_equil:
            out_s[sweep - n_equil] = s
            out_x[sweep - n_equil] = x
    return out_s, out_x, accepted / max(attempted, 1)
