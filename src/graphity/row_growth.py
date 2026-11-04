"""Space that grows by whole rows: the owner's growth rule for PREREGISTRATION T76.

Naming. The energy is the toy's, H = 16 (N - S) + 4 lam X (cqg.py, Q1), four links per point, no cap. T76 runs it at
lam = 1.25: at lam != 1 this is our family around the published model, never combinatorial quantum gravity (VISION
Update 22).

THE STATE. A four-link graph whose points carry grid labels: point p has coordinates (x, y) on an Lx x Ly torus,
and every coordinate pair belongs to exactly one point (`at[x, y]` is that point). The labels are given once, by the
flat torus the run starts from (cqg.torus numbering, point i at divmod(i, Ly)), and they stay with the points while
the chain rewires the links: a label says where a point sat in the grid, not what it is linked to now. Rows are the
sets of points sharing one coordinate: along axis 0, row i is the Ly points with x = i; along axis 1, the Lx points
with y = i. W is the length of a row (Ly along axis 0, Lx along axis 1). The two sides of the bipartition are
part = (x + y) % 2, and an insertion keeps that true.

THE GROWTH RULE (T76, "Design", the owner's words of VISION Update 75; the precise readings below are ours). A pair of
rows may go between rows i and i + 1 (mod L) along an axis when
  (i)  the grid is intact across that boundary: every point of row i is still linked to its partner in row i + 1,
       the point with the same other coordinate; and
  (ii) the corridor is clear: no link with an endpoint within `reach` (3) steps of a point of row i or row i + 1,
       distance counted along the links of the current graph, sits on more than two squares (S_e > 2, a "curled"
       link). *Ours:* "a link within three steps" is read as a link with at least one endpoint at graph distance at
       most 3; the stricter reading (both endpoints) would allow a few more insertions.
The insertion cuts the W links across the boundary and puts two new rows a, b between: row i - a - b - row i + 1,
with straight links across (p - a_y, a_y - b_y, b_y - q) and links along each new row (a_y - a_{y+1}, b_y - b_{y+1},
round the torus). Every old point keeps four links (it trades its partner across the boundary for a new point), every
new point has four. The new rows' sides alternate, so the graph stays two-sided; the coordinates along the axis
beyond i move up by 2 and the length L along that axis grows by 2. Valid whenever W >= 6: an old pair of points
gains no common neighbour, and a new point's common neighbours with any point are at most two (worked through in
the tests, which check cqg.is_valid without the cap). The new points have the numbers N .. N + 2W - 1.

THE GROWTH STEP. Every boundary along both axes that exists when the step begins is offered once, in an order drawn
from the step's numpy Generator. A boundary is remembered by one point of its lower row, so that insertions made
earlier in the same step, which renumber coordinates, do not change which pair of rows it names; boundaries created
by the step's own insertions are not offered until the next step. *Exact consequence:* on a clean sheet every one of
an axis's L boundaries takes a pair, so one step can take L to 3L along both axes and N to up to 9N. Each candidate is checked against the graph as it
stands at the moment it is offered (the curled links are recomputed after every insertion). With `stop_at` given, the
step offers nothing more once N >= stop_at (T76's stop, "when N reaches 8 times its starting value", read as
stopping at once, mid-step if need be: ours).

THE DRIVER (run_growth). Settling: n_settle fair sweeps of the chain with growth off. Growth: then each fair sweep is
one block of the chain followed by one growth step, until N >= cap_factor * N0 or n_growth fair sweeps have passed.
A reading is taken before any attempt and after every fair sweep: N, Lx, Ly, S, X, D = sum_e max(0, 2 - S_e) (the
missing squares), rho = D / N, the share of links with S_e > 2, the rows added in that fair sweep, and the block's
acceptance.

The fair clock (ASSUMPTIONS O90). One fair sweep is N / n_ref sweeps of 2N attempts, 2 N^2 / n_ref attempts, with N
the number of points when the block starts. The remainder is carried from block to block as an exact integer, so at
a fixed N the attempts after k fair sweeps are exactly floor(2 N^2 k / n_ref), as in coupled.fair_sweep_schedule; when
N changes the carry goes on with the new N.

The chain. `_attempts` is cqg.run_chain's inner loop, line for line (Metropolis; the same four integer draws per
attempt and a uniform draw only when dH > 0; cqg's helpers for the move and for dS, dX), run for a given number of
attempts rather than whole sweeps of 2N, so that a fair sweep need not be a whole number of sweeps. Its random stream
is numba's, seeded once at the start of the run and carried on through every block (seed < 0, as in run_chain:
re-seeding in short blocks is what Q14 warns against). With growth off and a whole number of chain sweeps in total
it is therefore cqg.run_chain on the same seed, bit for bit (tested). The chain's detailed balance is cqg's (Q4).
The growth step uses a separate numpy Generator; both are drawn from one SeedSequence (`seed`), so a run is fixed by
its seed.

NO DETAILED BALANCE IS CLAIMED FOR THE GROWTH STEP. It only adds points and never removes them; the run is not an
equilibrium sampler of anything once growth starts, and is not meant to be. What T76 reads is how the density of
missing squares evolves under this rule, chain and growth together.
"""
import numpy as np
from numba import njit

from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, _new_edge_ok, _note_square_edges, _replace,
                          _surplus_on, _switch, surplus, torus, total_squares)
from graphity.squares import has_edge, squares_on_edge

N_REF = 96          # the fair clock's reference size (O90)
REACH = 3           # T76: the corridor reaches three steps either side


# ---------------------------------------------------------------------------------------------- the chain

@njit(cache=True)
def _attempts(adj, side_u, inv_g, n_attempts, seed, lam, cap):
    """n_attempts switches of cqg.run_chain's chain (Metropolis), in place. seed < 0 carries the stream on.
    Returns the number accepted. The body is run_chain's, line for line."""
    if seed >= 0:
        np.random.seed(seed)
    nu = side_u.shape[0]
    track_x = cap > 2
    edges = np.empty((64, 2), dtype=np.int64)
    accepted = 0
    for _ in range(n_attempts):
        u1 = side_u[np.random.randint(0, nu)]
        v1 = adj[u1, np.random.randint(0, 4)]
        u2 = side_u[np.random.randint(0, nu)]
        v2 = adj[u2, np.random.randint(0, 4)]
        if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
            continue
        n_edges = 0
        if track_x:
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
        if ok and track_x:
            n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
            n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
            after = _surplus_on(adj, edges, n_edges)
            _switch(adj, u1, v2, u2, v1)
            before = _surplus_on(adj, edges, n_edges)
            _switch(adj, u1, v1, u2, v2)
            d_x = after - before
        if ok:
            d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
            if d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                ok = False
        if ok:
            accepted += 1
        else:
            _switch(adj, u1, v2, u2, v1)
    return accepted


# ---------------------------------------------------------------------------------------------- readers

@njit(cache=True)
def edge_census(adj):
    """(D, curled links, links): D = sum over links of max(0, 2 - S_e), curled = links with S_e > 2."""
    d = 0
    curled = 0
    links = 0
    for u in range(adj.shape[0]):
        for k in range(4):
            v = adj[u, k]
            if v > u:
                links += 1
                s_e = squares_on_edge(adj, u, v)
                if s_e < 2:
                    d += 2 - s_e
                elif s_e > 2:
                    curled += 1
    return d, curled, links


@njit(cache=True)
def curled_points(adj):
    """True for every point that is an endpoint of a link with S_e > 2."""
    n = adj.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for u in range(n):
        for k in range(4):
            v = adj[u, k]
            if v > u and squares_on_edge(adj, u, v) > 2:
                out[u] = True
                out[v] = True
    return out


@njit(cache=True)
def _corridor_clear(adj, sources, flagged, reach, dist, queue):
    """True when no point within `reach` steps of `sources` is flagged. dist (all -1 on entry) and queue are
    scratch arrays of length N; dist is reset to -1 before returning."""
    head = 0
    tail = 0
    clear = True
    for s in sources:
        if dist[s] < 0:
            dist[s] = 0
            queue[tail] = s
            tail += 1
    while head < tail:
        p = queue[head]
        head += 1
        if flagged[p]:
            clear = False
            break
        if dist[p] < reach:
            for k in range(4):
                q = adj[p, k]
                if dist[q] < 0:
                    dist[q] = dist[p] + 1
                    queue[tail] = q
                    tail += 1
    for j in range(tail):
        dist[queue[j]] = -1
    return clear


# ---------------------------------------------------------------------------------------------- the state

class GrowthState:
    """A four-link graph with grid labels: adj (N, 4), part (N,), coords (N, 2), at (Lx, Ly)."""

    def __init__(self, adj, part, coords):
        self.adj = np.ascontiguousarray(adj, dtype=np.int64)
        self.part = np.ascontiguousarray(part, dtype=np.int64)
        self.coords = np.ascontiguousarray(coords, dtype=np.int64)
        self._rebuild_at()

    @classmethod
    def from_torus(cls, lx, ly=None):
        """The flat lx x ly torus of cqg.torus (no cap), point i at (x, y) = divmod(i, ly)."""
        ly = lx if ly is None else ly
        adj, part = torus(lx, ly, cap=NO_CAP)
        i = np.arange(lx * ly)
        return cls(adj, part, np.stack([i // ly, i % ly], axis=1))

    def _rebuild_at(self):
        lx, ly = int(self.coords[:, 0].max()) + 1, int(self.coords[:, 1].max()) + 1
        at = np.full((lx, ly), -1, dtype=np.int64)
        at[self.coords[:, 0], self.coords[:, 1]] = np.arange(self.n)
        if (at < 0).any() or lx * ly != self.n:
            raise ValueError("the grid labels must give every coordinate of an Lx x Ly torus exactly one point")
        self.at = at

    @property
    def n(self):
        return self.adj.shape[0]

    @property
    def lx(self):
        return self.at.shape[0]

    @property
    def ly(self):
        return self.at.shape[1]

    def side_u(self):
        return np.flatnonzero(self.part == 0).astype(np.int64)

    def copy(self):
        return GrowthState(self.adj.copy(), self.part.copy(), self.coords.copy())


def row_points(state, axis, i):
    """The points of row i along `axis` (0: x = i; 1: y = i), in order of the other coordinate."""
    return np.ascontiguousarray(state.at.take(i % state.at.shape[axis], axis=axis))


def grid_intact(state, axis, i):
    """Condition (i): every point of row i is still linked to its partner in row i + 1."""
    p = row_points(state, axis, i)
    q = row_points(state, axis, i + 1)
    return bool((state.adj[p] == q[:, None]).any(axis=1).all())


def corridor_clear(state, axis, i, reach=REACH, flagged=None, _scratch=None):
    """Condition (ii): no link with an endpoint within `reach` steps of rows i and i + 1 has S_e > 2."""
    if flagged is None:
        flagged = curled_points(state.adj)
    if _scratch is None:
        _scratch = (np.full(state.n, -1, dtype=np.int64), np.empty(state.n, dtype=np.int64))
    src = np.concatenate([row_points(state, axis, i), row_points(state, axis, i + 1)])
    return bool(_corridor_clear(state.adj, src, flagged, int(reach), _scratch[0], _scratch[1]))


def can_insert(state, axis, i, reach=REACH, flagged=None):
    """(allowed, reason): reason is "ok", "broken" (condition (i) fails) or "curled" (condition (ii) fails)."""
    if not grid_intact(state, axis, i):
        return False, "broken"
    if not corridor_clear(state, axis, i, reach, flagged):
        return False, "curled"
    return True, "ok"


def insert_rows(state, axis, i):
    """Put two flat rows between rows i and i + 1 along `axis`, in place. Condition (i) must hold (it is checked);
    condition (ii) is the caller's (can_insert). Returns the numbers of the 2W new points."""
    length = state.at.shape[axis]
    width = state.at.shape[1 - axis]
    if width < 6:
        raise ValueError("rows shorter than 6 would close squares round the torus")
    i = i % length
    p = row_points(state, axis, i)
    q = row_points(state, axis, i + 1)
    hit = state.adj[p] == q[:, None]
    if not hit.any(axis=1).all():
        raise ValueError("the grid is broken across this boundary (condition (i))")
    n = state.n
    a = n + np.arange(width)
    b = n + width + np.arange(width)
    adj = np.empty((n + 2 * width, 4), dtype=np.int64)
    adj[:n] = state.adj
    rows, slots = np.nonzero(hit)                      # exactly one slot per row: links are simple
    adj[p[rows], slots] = a[rows]
    back = adj[q] == p[:, None]
    rows_b, slots_b = np.nonzero(back)
    adj[q[rows_b], slots_b] = b[rows_b]
    adj[a] = np.stack([b, p, np.roll(a, -1), np.roll(a, 1)], axis=1)
    adj[b] = np.stack([q, a, np.roll(b, -1), np.roll(b, 1)], axis=1)
    part = np.concatenate([state.part, 1 - state.part[p], state.part[p]])
    coords = state.coords.copy()
    coords[coords[:, axis] > i, axis] += 2
    other = np.arange(width)
    new = np.empty((2 * width, 2), dtype=np.int64)
    new[:width, axis], new[:width, 1 - axis] = i + 1, other
    new[width:, axis], new[width:, 1 - axis] = i + 2, other
    state.adj = np.ascontiguousarray(adj)
    state.part = part
    state.coords = np.concatenate([coords, new])
    state._rebuild_at()
    return np.concatenate([a, b])


def growth_step(state, rng, reach=REACH, stop_at=None, q=1.0):
    """Offer every boundary along both axes once, in an order drawn from `rng`, inserting a pair of rows wherever
    can_insert allows at the moment of offering. With q < 1 (T76 Amendment 2, the owner's choice (b)) each offer is
    taken up only with chance q, drawn from `rng` before the boundary is checked; at q = 1 no number is drawn, so the
    stream is that of the registered rate. Returns the number of pairs inserted."""
    candidates = [(axis, int(row_points(state, axis, i)[0]))
                  for axis in (0, 1) for i in range(state.at.shape[axis])]
    order = rng.permutation(len(candidates))
    flagged = curled_points(state.adj)
    pairs = 0
    for c in order:
        if stop_at is not None and state.n >= stop_at:
            break
        axis, rep = candidates[c]
        if q < 1.0 and rng.random() >= q:
            continue
        i = int(state.coords[rep, axis])
        ok, _ = can_insert(state, axis, i, reach, flagged)
        if ok:
            insert_rows(state, axis, i)
            pairs += 1
            flagged = curled_points(state.adj)
    return pairs


# ---------------------------------------------------------------------------------------------- the driver

def _reading(state, fair_sweep, phase, attempts, accepted, rows_added):
    d, curled, links = edge_census(state.adj)
    return dict(fair_sweep=fair_sweep, phase=phase, attempts=attempts,
                acceptance=(accepted / attempts if attempts else None),
                N=state.n, Lx=state.lx, Ly=state.ly, S=int(total_squares(state.adj)), X=int(surplus(state.adj)),
                D=int(d), rho=d / state.n, curled_share=curled / links, rows_added=rows_added)


def stream_seeds(seed):
    """(numba chain seed, numpy Generator for growth), both from SeedSequence(seed)."""
    chain, grow = np.random.SeedSequence(seed).spawn(2)
    return int(chain.generate_state(1)[0]), np.random.default_rng(grow)


def run_growth(state, g, lam, n_settle, n_growth, cap_factor, seed, n_ref=N_REF, reach=REACH, cap=NO_CAP, q=1.0):
    """T76's protocol on `state`, in place (see the module docstring). Returns the list of readings, one before any
    attempt (phase "start") and one after every fair sweep ("settle", then "grow")."""
    chain_seed, rng = stream_seeds(seed)
    inv_g = 1.0 / float(g)
    n0 = state.n
    stop_at = cap_factor * n0
    carry = 0
    first = True
    out = [_reading(state, 0, "start", 0, 0, 0)]
    for t in range(1, n_settle + n_growth + 1):
        growing = t > n_settle
        num = carry + 2 * state.n * state.n
        k, carry = num // n_ref, num % n_ref
        acc = _attempts(state.adj, state.side_u(), inv_g, int(k), chain_seed if first else -1, float(lam), int(cap))
        first = False
        pairs = growth_step(state, rng, reach, stop_at, q) if growing else 0
        out.append(_reading(state, t, "grow" if growing else "settle", int(k), int(acc), 2 * pairs))
        if growing and state.n >= stop_at:
            break
    return out


def settle_graph(adj, part, g, lam, n_settle, seed, n_ref=N_REF, cap=NO_CAP):
    """The settling stage of run_growth on any four-link two-sided graph, in place, with no grid labels (for a start
    such as a burp's product, which is not a labelled torus). The same chain, fair clock and stream as run_growth's
    settling (on a labelled torus, the same readings, tested); Lx and Ly are reported as 0."""
    adj = np.ascontiguousarray(adj, dtype=np.int64)
    side_u = np.flatnonzero(np.asarray(part) == 0).astype(np.int64)
    chain_seed, _ = stream_seeds(seed)
    inv_g = 1.0 / float(g)
    n = adj.shape[0]

    def reading(t, k, acc):
        d, curled, links = edge_census(adj)
        return dict(fair_sweep=t, phase="start" if t == 0 else "settle", attempts=k,
                    acceptance=(acc / k if k else None), N=n, Lx=0, Ly=0, S=int(total_squares(adj)),
                    X=int(surplus(adj)), D=int(d), rho=d / n, curled_share=curled / links, rows_added=0)

    carry = 0
    out = [reading(0, 0, 0)]
    for t in range(1, n_settle + 1):
        k, carry = divmod(carry + 2 * n * n, n_ref)
        acc = _attempts(adj, side_u, inv_g, int(k), chain_seed if t == 1 else -1, float(lam), int(cap))
        out.append(reading(t, int(k), int(acc)))
    return out
