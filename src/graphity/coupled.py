"""Two copies of the four-link chain driven by the same random numbers (PREREGISTRATION T75).

What it is for. T75 asks whether a disturbance in the toy's open space spreads, and how fast. The standard way
to watch that ("damage spreading", general knowledge, to verify) is to run two copies of one chain side by side,
disturb one of them, feed both the same random numbers, and follow the set of places where they differ. This
module is the core of that: the coupled kernel, the damage set, the radius readers and the flaw. Which space the
disturbance sits in and what the second kind of disturbance is are still the owner's to choose (T75 Amendment 1),
so neither the knot nor the seam is built here.

Naming. The energy is the toy's, H = 16 (N - S) + 4 lam X, as in cqg.py (Q1). At lam != 1 this is our family
around the published model, never combinatorial quantum gravity (VISION Update 22).

THE COUPLING (fixed by T75's design before any run). At every attempt exactly five numbers are drawn, whatever
the state of either copy:

    i1 in {0..nu-1}, k1 in {0..3}, i2 in {0..nu-1}, k2 in {0..3}, r uniform in [0, 1).

Both copies read the same five numbers. In each copy u1 = side_u[i1], v1 = the k1-th smallest neighbour of u1,
u2 = side_u[i2], v2 = the k2-th smallest neighbour of u2, and the switch (u1,v1),(u2,v2) -> (u1,v2),(u2,v1) is
accepted when it is valid and r < exp(-dH / g) (Metropolis; dH <= 0 is always accepted). Nothing is skipped: an
invalid proposal still uses up all five numbers, so the two copies never fall out of step.

  Why "k-th smallest neighbour" and not "the neighbour in slot k" as cqg.run_chain does (ours). The order of a
  point's slots records its history: _switch puts each new neighbour in the old one's slot, so two copies whose
  wiring has come back into agreement by different routes can hold the same neighbours in a different order. Read
  by slot, they would then receive different proposals at a point where they agree, and damage would be made by the
  bookkeeping. Read by rank, a point's proposal depends only on its set of neighbours, so the copies are identical
  wherever their wiring agrees, which is what the damage set measures.

DETAILED BALANCE, for each copy on its own. (1) The proposal. An ordered proposal (u1, v1, u2, v2) has
probability (1/nu)(1/4)(1/nu)(1/4): u1 and u2 are uniform on side U, and a uniform rank k picks each of the four
distinct neighbours with probability 1/4, whatever order the slots are in. That is exactly cqg's proposal
distribution (Q4). After the switch, u1 has neighbour v2 and u2 has neighbour v1, so the reverse switch
(u1,v2),(u2,v1) is proposed with the same probability: the proposal is symmetric. Proposals leaving the
configuration space (a double edge, the hard-core rule, the cap) are rejected, which keeps symmetry on the
restricted space. (2) The acceptance. r is uniform and independent of the state and of everything drawn before,
so the chance of accepting is min(1, exp(-dH/g)), and p(dH)/p(-dH) = exp(-dH/g). With (1) that is detailed balance
for exp(-H/g). (3) Drawing r and the other numbers when they are not needed changes nothing for either copy taken
alone: unused numbers are independent of the move. So each copy, read by itself, is a Markov chain with the same
transition probabilities as cqg.run_chain (not the same random stream, the same distribution), and the pair is a
coupling of two such chains. The joint chain of the pair is not claimed to be reversible and need not be.
Ergodicity is that of cqg (exhaustive for N <= 18, Q9; unproven beyond). The tests check one copy against the
exact averages at N = 16 and 18.

CAUTION. How damage spreads depends on how the copies share random numbers, not only on the energy: a different
coupling of the same chain (another way of reading the five numbers, or a heat-bath rule in place of Metropolis)
can heal or spread a disturbance differently. The coupling above is the one T75 fixed, and results from it speak
for this coupling.

THE DAMAGE. A point is damaged when its set of neighbours differs between the copies. A damaged point keeps its
place on the starting graph whatever the wiring has become. T75 as amended (Amendment 2, item 4) counts distance in
steps along the links of the shared starting graph (graph_distances, diameter, joined_region, read_records): R is
the distance of the farthest damaged point from the nearest of the flaw's four points, D the starting graph's
diameter, and the region joined to the flaw is the damaged points reachable from the flaw through damaged points at
most 2 steps apart; a record with damage but none joined to the flaw is a jump. The torus readers (coordinates,
lattice_distance, damage_radius, connected_region) are the same ideas on a flat lx x ly torus, point i at
(x, y) = divmod(i, ly) as cqg.torus numbers it; on a flat torus the two distances agree.

THE FLAW. T75 Amendment 3, item 1 (in force): the flaw at a chosen point is the cheapest valid switch of an edge at
that point with an edge within 3 steps of it, priced exactly at lam = 1.25 as the energy difference against the
unflawed state, ties broken in a fixed sorted order (cheapest_switch_at). Beside a knot (Amendment 4) it is the
cheapest among the switches that touch no knot point (cheapest_switch_at's `avoid`). Under Amendment 5 only switches
that cost strictly more than zero count (`above`=0), so a cost-free rewiring is never a flaw. It replaced Amendment 2's flaw, the switch
of the two outer parallel edges of two squares sharing an edge at a cost of exactly 32 (flaw_candidates, flaw_at,
kept for the record), because in the saved products of a burp that switch costs 31 where it is valid and the cheapest
way out costs 20. On a plain flat torus every cheapest switch costs 32 (O22); flaw() builds one there,
(p, p + e) and (p + 2f, p + 2f + e). No valid switch of "two edges that share a square" exists in flat space (two
edges of one square either share a point, or are opposite sides of it, and then each new edge is already present).
"""
import numpy as np
from numba import njit

from graphity.cqg import (CAP, ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, _new_edge_ok, _note_square_edges,
                          _replace, _surplus_on, _switch, hamiltonian, is_valid, surplus, total_squares)
from graphity.squares import has_edge, squares_on_edge

N_REF = 96                 # the fair clock's reference size (O90)


@njit(cache=True)
def _ranked_neighbour(adj, u, k):
    """The k-th smallest neighbour of u (k = 0..3). Neighbours are distinct, so the ranks are."""
    for j in range(4):
        w = adj[u, j]
        below = 0
        for m in range(4):
            if adj[u, m] < w:
                below += 1
        if below == k:
            return w
    return -1


@njit(cache=True)
def _attempt(adj, u1, v1, u2, v2, r, inv_g, lam, cap, track_x, edges):
    """One proposed switch in one copy, with the uniform r already drawn. Returns (accepted, dS, dX).

    The body is cqg.run_chain's, line for line, except that r is given rather than drawn.
    """
    if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
        return False, 0, 0
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
        if d_h > 0.0 and r >= np.exp(-inv_g * d_h):
            ok = False
    if not ok:
        _switch(adj, u1, v2, u2, v1)
        return False, 0, 0
    return True, d_s, d_x


@njit(cache=True)
def damage_mask(adj_a, adj_b):
    """True at every point whose set of neighbours differs between the two graphs."""
    n = adj_a.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for u in range(n):
        for k in range(4):
            if not has_edge(adj_b, u, adj_a[u, k]):
                out[u] = True
                break
    return out


@njit(cache=True)
def _run_coupled(adj_a, adj_b, side_u, inv_g, schedule, seed, lam, cap):
    if seed >= 0:
        np.random.seed(seed)
    n = adj_a.shape[0]
    nu = side_u.shape[0]
    track_x = cap > CAP
    edges_a = np.empty((64, 2), dtype=np.int64)
    edges_b = np.empty((64, 2), dtype=np.int64)
    s_a, x_a = total_squares(adj_a), surplus(adj_a)
    s_b, x_b = total_squares(adj_b), surplus(adj_b)
    n_records = schedule.shape[0]
    count = np.zeros(n_records, dtype=np.int64)
    masks = np.zeros((n_records, n), dtype=np.bool_)
    out = np.zeros((4, n_records), dtype=np.int64)
    acc_a = 0
    acc_b = 0
    for rec in range(n_records):
        for _ in range(schedule[rec]):
            i1 = np.random.randint(0, nu)              # the same five numbers at every attempt, whatever the state
            k1 = np.random.randint(0, 4)
            i2 = np.random.randint(0, nu)
            k2 = np.random.randint(0, 4)
            r = np.random.random()
            u1 = side_u[i1]
            u2 = side_u[i2]
            ok, d_s, d_x = _attempt(adj_a, u1, _ranked_neighbour(adj_a, u1, k1), u2,
                                    _ranked_neighbour(adj_a, u2, k2), r, inv_g, lam, cap, track_x, edges_a)
            if ok:
                s_a += d_s
                x_a += d_x
                acc_a += 1
            ok, d_s, d_x = _attempt(adj_b, u1, _ranked_neighbour(adj_b, u1, k1), u2,
                                    _ranked_neighbour(adj_b, u2, k2), r, inv_g, lam, cap, track_x, edges_b)
            if ok:
                s_b += d_s
                x_b += d_x
                acc_b += 1
        m = damage_mask(adj_a, adj_b)
        masks[rec] = m
        count[rec] = m.sum()
        out[0, rec] = s_a
        out[1, rec] = x_a
        out[2, rec] = s_b
        out[3, rec] = x_b
    total = max(schedule.sum(), 1)
    return count, masks, out, acc_a / total, acc_b / total


def run_coupled(adj_a, adj_b, side_u, inv_g, n_records, attempts_per_record, seed, lam=1.0, cap=NO_CAP):
    """Advance copies A and B together, in place, with shared draws (the coupling of the docstring).

    side_u              : the points of one side, shared by both copies (an index into it is what is drawn).
    inv_g               : 1 / g.
    n_records           : how many times the damage is read.
    attempts_per_record : attempts before each reading: one number for every block, or an array of n_records
                          numbers (fair_sweep_schedule gives the fair clock with its remainder carried). A sweep is 2N
                          attempts, as in cqg. Zero is allowed and reads the damage as it stands.
    seed                : seeds the stream; seed < 0 carries it on from the previous call, as in cqg.run_chain.
    Returns a dict: 'damaged' (n_records,) the number of damaged points after each block; 'mask' (n_records, N)
    which points; 'squares_a', 'surplus_a', 'squares_b', 'surplus_b' after each block; 'acceptance_a',
    'acceptance_b'.
    """
    adj_a = _check_graph(adj_a, cap, "A")
    adj_b = _check_graph(adj_b, cap, "B")
    if adj_a.shape != adj_b.shape:
        raise ValueError("the two copies must have the same number of points")
    side_u = np.ascontiguousarray(side_u, dtype=np.int64)
    in_u = np.zeros(adj_a.shape[0], dtype=bool)
    in_u[side_u] = True
    for adj, name in ((adj_a, "A"), (adj_b, "B")):
        if not np.all(in_u[:, None] != in_u[adj]):
            raise ValueError(f"copy {name}: every link must join side_u to the other side")
    schedule = np.ascontiguousarray(np.broadcast_to(np.asarray(attempts_per_record, dtype=np.int64),
                                                    (int(n_records),)))
    if n_records < 1 or (schedule < 0).any():
        raise ValueError("need n_records >= 1 and no negative number of attempts")
    count, masks, out, acc_a, acc_b = _run_coupled(adj_a, adj_b, side_u, float(inv_g), schedule, int(seed),
                                                   float(lam), int(cap))
    return dict(damaged=count, mask=masks, squares_a=out[0], surplus_a=out[1], squares_b=out[2],
                surplus_b=out[3], acceptance_a=acc_a, acceptance_b=acc_b)


def _check_graph(adj, cap, name):
    if not (isinstance(adj, np.ndarray) and adj.dtype == np.int64 and adj.ndim == 2 and adj.shape[1] == 4
            and adj.flags.c_contiguous):
        raise ValueError(f"copy {name} must be a C-contiguous (N, 4) int64 array (it is changed in place)")
    if not is_valid(adj, cap):
        raise ValueError(f"copy {name} is not a valid state")
    return adj


def fair_sweep_attempts(n, n_ref=N_REF):
    """Attempts in one fair sweep, N / n_ref sweeps of 2N attempts (O90), rounded to a whole number."""
    return int(round(2 * n * n / n_ref))


def fair_sweep_schedule(n, n_fair, record_every=1, n_ref=N_REF):
    """(attempts before each reading, fair sweeps at each reading) for readings every `record_every` fair sweeps up
    to n_fair, the remainder carried from record to record (T75 Amendment 2, item 4): after k fair sweeps exactly
    floor(2 N^2 k / n_ref) attempts have been made, so nothing is lost or gained by rounding. The first reading is
    before any attempt (0 attempts, fair sweep 0)."""
    fair = np.arange(0, n_fair + 1, record_every, dtype=np.int64)
    done = (2 * n * n * fair) // n_ref
    return np.concatenate([[0], np.diff(done)]).astype(np.int64), fair


# ---------------------------------------------------------------- reading the damage on the starting torus

def coordinates(lx, ly=None):
    """(N, 2) array of (x, y) for each point of cqg.torus(lx, ly): point i is at divmod(i, ly)."""
    ly = lx if ly is None else ly
    i = np.arange(lx * ly)
    return np.stack([i // ly, i % ly], axis=1)


def lattice_distance(points, origins, lx, ly=None):
    """Steps on the starting lx x ly torus from each of `points` to the nearest of `origins` (shorter way round)."""
    ly = lx if ly is None else ly
    xy = coordinates(lx, ly)
    p = xy[np.atleast_1d(np.asarray(points, dtype=np.int64))]
    o = xy[np.atleast_1d(np.asarray(origins, dtype=np.int64))]
    dx = np.abs(p[:, None, 0] - o[None, :, 0])
    dy = np.abs(p[:, None, 1] - o[None, :, 1])
    d = np.minimum(dx, lx - dx) + np.minimum(dy, ly - dy)
    return d.min(axis=1)


def damage_radius(mask, origins, lx, ly=None):
    """R: the largest lattice distance of a damaged point from the disturbance (the nearest of `origins`).
    -1 when nothing is damaged."""
    pts = np.flatnonzero(mask)
    if pts.size == 0:
        return -1
    return int(lattice_distance(pts, origins, lx, ly).max())


def connected_region(mask, origins, lx, ly=None, *, reach):
    """The damaged points joined to a damaged origin through damaged points, two damaged points counting as joined
    when they lie within `reach` lattice steps of each other on the starting torus. Returns a boolean mask (all False
    when no origin is damaged).

    `reach` has no default on purpose: T75 does not say what "connected" means for the JUMPS rule. With reach = 1 the
    flaw of `flaw()` is already two pieces (its two edges lie two steps apart); reach = 2 is the smallest that keeps
    it whole. Joining through the copies' own links is not offered: a far link would then join the far damage it
    carries to the near damage, which is what JUMPS is meant to catch.
    """
    ly = lx if ly is None else ly
    pts = np.flatnonzero(mask)
    region = np.zeros(len(mask), dtype=bool)
    if pts.size == 0:
        return region
    xy = coordinates(lx, ly)[pts]
    dx = np.abs(xy[:, None, 0] - xy[None, :, 0])
    dy = np.abs(xy[:, None, 1] - xy[None, :, 1])
    near = (np.minimum(dx, lx - dx) + np.minimum(dy, ly - dy)) <= reach
    seen = np.isin(pts, np.atleast_1d(origins))
    frontier = seen.copy()
    while frontier.any():
        new = near[frontier].any(axis=0) & ~seen
        seen |= new
        frontier = new
    region[pts[seen]] = True
    return region


def region_radius(mask, origins, lx, ly=None, *, reach):
    """Radius of the connected damaged region containing the disturbance (see connected_region); -1 if empty."""
    return damage_radius(connected_region(mask, origins, lx, ly, reach=reach), origins, lx, ly)


# ---------------------------------------------------------------- the flaw

def flaw(adj, point, lx, ly=None, axis=0, cap=NO_CAP):
    """A copy of the flat torus `adj` with one cheapest switch at `point` (see the docstring, THE FLAW).

    The edges switched are (p, p + e) and (p + 2f, p + 2f + e), e one step along `axis` (0: x, 1: y) and f one
    step along the other. Returns (flawed copy, (u1, v1, u2, v2)), the four points the switch touched. Raises
    ValueError if the result is not a valid state.
    """
    ly = lx if ly is None else ly
    x, y = divmod(int(point), ly)

    def at(a, b):
        return (a % lx) * ly + b % ly

    if axis == 0:
        u1, v1, u2, v2 = at(x, y), at(x + 1, y), at(x, y + 2), at(x + 1, y + 2)
    elif axis == 1:
        u1, v1, u2, v2 = at(x, y), at(x, y + 1), at(x + 2, y), at(x + 2, y + 1)
    else:
        raise ValueError("axis must be 0 or 1")
    if not (has_edge(adj, u1, v1) and has_edge(adj, u2, v2)):
        raise ValueError("adj is not the flat lx x ly torus of cqg.torus")
    out = adj.copy()
    if len({u1, v1, u2, v2}) < 4 or has_edge(out, u1, v2) or has_edge(out, u2, v1):
        raise ValueError("the switch is not possible here (the torus is too small)")
    _switch(out, u1, v1, u2, v2)
    if not is_valid(out, cap):
        raise ValueError("the switch breaks the hard-core rule or the cap here")
    return out, (u1, v1, u2, v2)


# ---------------------------------------------------------------- distances on the shared starting graph (Amendment 2)

@njit(cache=True)
def graph_distances(adj, sources):
    """Steps along the links of `adj` from the nearest of `sources` to every point; -1 where unreachable."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for s in sources:
        if dist[s] < 0:
            dist[s] = 0
            queue[tail] = s
            tail += 1
    while head < tail:
        x = queue[head]
        head += 1
        for k in range(4):
            y = adj[x, k]
            if y >= 0 and dist[y] < 0:
                dist[y] = dist[x] + 1
                queue[tail] = y
                tail += 1
    return dist


@njit(cache=True)
def diameter(adj):
    """D: the largest distance between two points of the graph; -1 if it is not connected."""
    n = adj.shape[0]
    src = np.empty(1, dtype=np.int64)
    best = 0
    for s in range(n):
        src[0] = s
        d = graph_distances(adj, src)
        if d.min() < 0:
            return -1
        if d.max() > best:
            best = d.max()
    return best


@njit(cache=True)
def joined_region(adj, mask, anchors, step):
    """The damaged points joined to the flaw: reachable from `anchors` (the flaw's four points) through damaged
    points, each within `step` steps of the one before on `adj` (the shared starting graph). An anchor belongs to the
    region only if it is damaged itself. Returns a boolean mask."""
    n = adj.shape[0]
    region = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n + anchors.shape[0], dtype=np.int64)
    head = 0
    tail = 0
    for a in anchors:
        queue[tail] = a
        tail += 1
        if mask[a] and not region[a]:
            region[a] = True
    stamp = np.full(n, -1, dtype=np.int64)       # the local walk of radius `step` around each popped point
    local = np.empty(n, dtype=np.int64)
    depth = np.empty(n, dtype=np.int64)
    walk = 0
    while head < tail:
        x = queue[head]
        head += 1
        lh = 0
        lt = 1
        local[0] = x
        depth[0] = 0
        stamp[x] = walk
        while lh < lt:
            y = local[lh]
            dy = depth[lh]
            lh += 1
            if dy == step:
                continue
            for k in range(4):
                z = adj[y, k]
                if z >= 0 and stamp[z] != walk:
                    stamp[z] = walk
                    local[lt] = z
                    depth[lt] = dy + 1
                    lt += 1
                    if mask[z] and not region[z]:
                        region[z] = True
                        queue[tail] = z
                        tail += 1
        walk += 1
    return region


@njit(cache=True)
def read_records(adj_start, masks, anchors, step):
    """For each record (a row of `masks`): the number damaged; R, the distance on the starting graph of the farthest
    damaged point from the nearest of `anchors` (-1 if none is damaged); the radius of the region joined to the flaw
    (joined_region; -1 if empty); and whether the record is a jump (Amendment 2, item 4): damage exists and either
    none of it is joined to the flaw or some lies farther than twice the joined region's radius. Points the starting
    graph cannot reach from the flaw count at distance N (farther than anything reachable)."""
    n_rec, n = masks.shape
    dist = graph_distances(adj_start, anchors)
    for v in range(n):
        if dist[v] < 0:
            dist[v] = n
    damaged = np.zeros(n_rec, dtype=np.int64)
    radius = np.full(n_rec, -1, dtype=np.int64)
    joined = np.full(n_rec, -1, dtype=np.int64)
    jump = np.zeros(n_rec, dtype=np.bool_)
    for r in range(n_rec):
        m = masks[r]
        c = 0
        big = -1
        for v in range(n):
            if m[v]:
                c += 1
                if dist[v] > big:
                    big = dist[v]
        damaged[r] = c
        radius[r] = big
        if c == 0:
            continue
        reg = joined_region(adj_start, m, anchors, step)
        rr = -1
        for v in range(n):
            if reg[v] and dist[v] > rr:
                rr = dist[v]
        joined[r] = rr
        jump[r] = rr < 0 or big > 2 * rr
    return damaged, radius, joined, jump


# ---------------------------------------------------------------- the flaw on any starting graph

def flaw_candidates(adj, point):
    """Every switch of the two outer, parallel edges of two squares that share an edge, with `point` among its four
    points, in a fixed order. A switch is (b, a, b2, a2): the squares are w - z - a - b and w - z - a2 - b2 on the
    shared edge (w, z), with b, b2 neighbours of w and a, a2 neighbours of z; the switch replaces (b, a), (b2, a2) by
    (b, a2), (b2, a). The order is that of the two removed edges written as sorted pairs, the pair of pairs sorted."""
    point = int(point)
    found = {}
    for w in adj[point]:
        for z in adj[w]:
            w, z = int(w), int(z)
            if z == point:
                continue
            for edge in ((w, z), (z, w)):              # the shared edge, from either end
                p, q = edge
                squares = []                            # p - q - a - b - p: a ~ q, b ~ p, a ~ b
                for a in adj[q]:
                    if a == p:
                        continue
                    for b in adj[p]:
                        if b != q and has_edge(adj, int(a), int(b)):
                            squares.append((int(b), int(a)))
                for i in range(len(squares)):
                    for j in range(i + 1, len(squares)):
                        (b, a), (b2, a2) = squares[i], squares[j]
                        if len({b, a, b2, a2}) < 4 or point not in (b, a, b2, a2):
                            continue
                        if has_edge(adj, b, a2) or has_edge(adj, b2, a):
                            continue
                        key = tuple(sorted([tuple(sorted((b, a))), tuple(sorted((b2, a2)))]))
                        found.setdefault(key, (b, a, b2, a2))
    return [found[k] for k in sorted(found)]


def flaw_at(adj, point, lam=1.25, cost=32.0, cap=NO_CAP):
    """The first switch of flaw_candidates(adj, point) that leaves a valid state and costs exactly `cost` at `lam`
    (Amendment 2, item 2: 32 at lam = 1.25). cost=None takes the first valid candidate whatever it costs. Returns
    (flawed copy, (b, a, b2, a2), its cost); raises ValueError if no candidate qualifies.

    Measured on T51's saved end states (2026-10-20, read only, nothing recorded): there the space is flat point by
    point (d = 2, two squares on every link) but the valid switches of this kind cost 31 almost everywhere, and the
    cheapest valid switch of any kind costs 20, so "every cheapest move out of flat space costs 32" (O22, measured on
    a plain flat torus) does not carry over to those products."""
    base = hamiltonian(adj, lam)
    for b, a, b2, a2 in flaw_candidates(adj, point):
        out = adj.copy()
        _switch(out, b, a, b2, a2)
        if not is_valid(out, cap):
            continue
        dh = hamiltonian(out, lam) - base
        if cost is None or abs(dh - cost) < 1e-9:
            return out, (b, a, b2, a2), float(dh)
    raise ValueError("no switch at point %d is valid with cost %s" % (int(point), cost))


def cheapest_switch_at(adj, point, lam=1.25, reach=3, cap=NO_CAP, avoid=None, above=None):
    """T75 Amendment 3, item 1: the cheapest valid switch of an edge at `point` with an edge within `reach` steps.
    With `avoid` (an array of points; Amendment 4, the knot points for a flaw beside a knot), only switches whose
    four points include none of them are considered. With `above` (Amendment 5: 0), only switches costing strictly
    more than it are considered, so a cost-free rewiring is never taken.

    An edge counts as within `reach` steps when its nearer end is at most `reach` steps from `point` on `adj`. For an
    edge (point, q) and such an edge (x, y), x the end on point's side of the bipartition (the end at an even
    distance), the switch replaces them by (point, y) and (x, q). Each switch is priced exactly, as
    hamiltonian(switched, lam) - hamiltonian(adj, lam), and the cheapest valid one is kept; among equal costs the one
    whose two removed edges, written as sorted pairs and the pair of pairs sorted, come first. Returns (switched copy,
    (point, q, x, y), cost); raises ValueError if no valid switch exists."""
    point = int(point)
    banned = set() if avoid is None else {int(v) for v in avoid}
    dist = graph_distances(adj, np.array([point], dtype=np.int64))
    base = hamiltonian(adj, lam)
    edges = set()
    for w in np.flatnonzero((dist >= 0) & (dist <= reach)):
        for z in adj[w]:
            w_, z_ = int(w), int(z)
            x, y = (w_, z_) if dist[w_] % 2 == 0 else (z_, w_)
            edges.add((x, y))
    best = None
    for q in sorted(int(v) for v in adj[point]):
        for x, y in sorted(edges):
            if x == point or y == q or has_edge(adj, point, y) or has_edge(adj, x, q):
                continue
            if banned and banned & {point, q, x, y}:
                continue
            out = adj.copy()
            _switch(out, point, q, x, y)
            if not is_valid(out, cap):
                continue
            cost = float(hamiltonian(out, lam) - base)
            if above is not None and cost <= above + 1e-9:
                continue
            key = (round(cost, 9), tuple(sorted([tuple(sorted((point, q))), tuple(sorted((x, y)))])))
            if best is None or key < best[0]:
                best = (key, out, (point, q, x, y), cost)
    if best is None:
        raise ValueError("no valid switch at point %d" % point)
    return best[1], best[2], best[3]
