"""The clean two-switch move (VISION Update 62; confirmed as part of the layer form's move set, Update 65 addendum).

Naming: at lambda != 1 this is our family around the published model, never combinatorial quantum gravity.

The author's picture of 18 October 2026: energy arrives at every point of the slab; a point is closed (four around,
d = 2) or open (six around, d = 3) and nothing else; breaking is not in the picture. In the six-link toy every single
switch out of the slab leaves four points broken (d > 3), so her picture needs a move the toy's chain lacks. This
module is that move, ported from the sandbox look of the same morning (`sandbox/clean_feed.py`) with rule-6 tests:

* a move is TWO switches of the kernel's kind (`cqg_d._switch`) made at once, the second anchored at a side-0 point
  within two steps of one of the first switch's four ends; it is accepted only if the hard-core rule holds after each
  switch and, after both, NO point is left broken (every point touched has d <= 3). Started from a clean state the
  chain stays clean, so breaking is excluded by the move set and not priced. A second switch that exactly undoes the
  first is a null move and is refused, not counted (the sandbox script counted such moves as accepted; its census
  numbers were unaffected). With `allow_break=True` the clean check
  is skipped and the same compound move is the CONTROL (the single-switch chain's behaviour, two switches at a time).
* its cost dH (the toy's H = 16 (3N - S) + 4 lam X, computed on the points within one step of each switch and exact,
  tests) is paid from the POOLED stores of the moved points (the up to eight ends of the two switches), each paying in
  proportion to what it holds; a release is credited to them evenly; a move the pool cannot pay is refused. No
  temperature. Energy is conserved exactly: H + sum(stores) - fed = H0 (tests, to 1e-9).
* `run_clean_feed` is the driven protocol of T67/T68 (the feed's "all" place): each sweep adds `rate` / N to every
  store, then makes N attempts. It is not an equilibrium chain and claims no detailed balance; its invariant is
  conservation. The single switch is kept only as a control (Update 65 addendum).

What a move changes, and why dH is local: a switch replaces (u1,v1),(u2,v2) by (u1,v2),(u2,v1); every square whose
count changes passes through one of those edges, so every edge whose square count changes has both ends among the four
points and their neighbours. `_local_energy` sums H over exactly those edges before and after. The tests check it
against `cqg_d.hamiltonian` on the whole graph for every accepted move.
"""
import numpy as np
from numba import njit

from .cqg_d import MAX_CODEGREE, _switch, hamiltonian, has_edge, is_valid, squares_on_edge, torus
from .dimension import _share_d, local_dimension_d, pieces_of

SAT = 4   # squares per link in flat six-link space; surplus counts above it (ASSUMPTIONS Q21; scripts/exact_lowest_path.py)


@njit(cache=True)
def d_at(adj, v):
    """Local dimension at v: the pairs of links at v that close no square (dimension.local_dimension_d, one point)."""
    deg = adj.shape[1]
    c = 0
    for i in range(deg):
        a = adj[v, i]
        for j in range(i + 1, deg):
            if not _share_d(adj, a, adj[v, j], v):
                c += 1
    return c


@njit(cache=True)
def _codeg(adj, x, y):
    c = 0
    for i in range(adj.shape[1]):
        a = adj[x, i]
        if a >= 0 and has_edge(adj, y, a):
            c += 1
    return c


@njit(cache=True)
def _ok_around(adj, pts):
    """Simple at pts, and the hard-core rule for every pair at distance two through a point of pts."""
    deg = adj.shape[1]
    for t in range(pts.shape[0]):
        p = pts[t]
        for i in range(deg):
            if adj[p, i] == p:
                return False
            for j in range(i + 1, deg):
                if adj[p, i] == adj[p, j]:
                    return False
        for i in range(deg):
            m = adj[p, i]
            for j in range(deg):
                w = adj[m, j]
                if w != p and _codeg(adj, p, w) > MAX_CODEGREE:
                    return False
    return True


@njit(cache=True)
def _local_energy(adj, pts, npts, lam):
    """16 (-S) + 4 lam X summed over the edges with both ends among pts[:npts] (each edge once)."""
    deg = adj.shape[1]
    mark = np.zeros(adj.shape[0], dtype=np.uint8)
    for t in range(npts):
        mark[pts[t]] = 1
    s4 = 0.0
    x = 0.0
    for t in range(npts):
        p = pts[t]
        for i in range(deg):
            q = adj[p, i]
            if q > p and mark[q]:
                c = squares_on_edge(adj, p, q)
                s4 += c
                if c > SAT:
                    x += c - SAT
    return -16.0 * s4 / 4.0 + 4.0 * lam * x


@njit(cache=True)
def _patch_points(adj, u1, v1, u2, v2, out):
    """The four points and their neighbours (before the switch), without repeats; returns the count."""
    deg = adj.shape[1]
    n = 0
    for s in (u1, v1, u2, v2):
        n = _add(out, n, s)
    for s in (u1, v1, u2, v2):
        for i in range(deg):
            n = _add(out, n, adj[s, i])
    return n


@njit(cache=True)
def _add(out, n, x):
    for t in range(n):
        if out[t] == x:
            return n
    out[n] = x
    return n + 1


@njit(cache=True)
def _near_side0(adj, part, anchor):
    """A side-0 point within two steps of anchor (a neighbour if anchor is side 1; else a neighbour's neighbour)."""
    deg = adj.shape[1]
    if part[anchor] == 1:
        return adj[anchor, np.random.randint(0, deg)]
    a = adj[anchor, np.random.randint(0, deg)]
    return adj[a, np.random.randint(0, deg)]


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def attempt(adj, part, stores, side0, lam, allow_break, pts1, pts2, four, ends, allpts):
    """One compound move, in place. Returns (accepted 0/1, dH). Work arrays: pts1, pts2 of length 4 + 4 deg; allpts
    twice that; four of 4; ends of 8."""
    deg = adj.shape[1]
    nu = side0.shape[0]
    u1 = side0[np.random.randint(0, nu)]
    v1 = adj[u1, np.random.randint(0, deg)]
    u2 = _near_side0(adj, part, u1)
    v2 = adj[u2, np.random.randint(0, deg)]
    if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
        return 0, 0.0
    n1 = _patch_points(adj, u1, v1, u2, v2, pts1)
    before = _local_energy(adj, pts1, n1, lam)
    _switch(adj, u1, v1, u2, v2)
    four[0] = u1; four[1] = u2; four[2] = v1; four[3] = v2
    if not _ok_around(adj, four):
        _switch(adj, u1, v2, u2, v1)
        return 0, 0.0
    dh = _local_energy(adj, pts1, n1, lam) - before
    a1 = _near_side0(adj, part, four[np.random.randint(0, 4)])
    b1 = adj[a1, np.random.randint(0, deg)]
    a2 = _near_side0(adj, part, four[np.random.randint(0, 4)])
    b2 = adj[a2, np.random.randint(0, deg)]
    if a1 == a2 or b1 == b2 or has_edge(adj, a1, b2) or has_edge(adj, a2, b1):
        _switch(adj, u1, v2, u2, v1)
        return 0, 0.0
    # a second switch on exactly the two links the first made undoes it: a null move, refused rather than counted
    if ((a1 == u1 and b1 == v2) or (a1 == u2 and b1 == v1)) and ((a2 == u1 and b2 == v2) or (a2 == u2 and b2 == v1)):
        _switch(adj, u1, v2, u2, v1)
        return 0, 0.0
    n2 = _patch_points(adj, a1, b1, a2, b2, pts2)
    before2 = _local_energy(adj, pts2, n2, lam)
    _switch(adj, a1, b1, a2, b2)
    four2 = four.copy()
    four2[0] = a1; four2[1] = a2; four2[2] = b1; four2[3] = b2
    ok = _ok_around(adj, four2)
    if ok:
        dh += _local_energy(adj, pts2, n2, lam) - before2
        if not allow_break:
            na = 0
            for t in range(n1):
                na = _add(allpts, na, pts1[t])
            for t in range(n2):
                na = _add(allpts, na, pts2[t])
            for t in range(na):
                if d_at(adj, allpts[t]) > 3:
                    ok = False
                    break
    if ok:
        ne = 0
        for e in (u1, v1, u2, v2, a1, b1, a2, b2):
            ne = _add(ends, ne, e)
        pool = 0.0
        for t in range(ne):
            pool += stores[ends[t]]
        if dh > pool:
            ok = False
        elif dh >= 0.0:
            if pool > 0.0:
                for t in range(ne):
                    stores[ends[t]] -= dh * stores[ends[t]] / pool
        else:
            for t in range(ne):
                stores[ends[t]] -= dh / ne
    if not ok:
        _switch(adj, a1, b2, a2, b1)
        _switch(adj, u1, v2, u2, v1)
        return 0, 0.0
    return 1, dh


@njit(cache=True)
def attempt_until_accepted(adj, part, stores, side0, lam, allow_break, cap, pts1, pts2, four, ends, allpts):
    """Attempts until one is accepted or `cap` attempts are made. Returns (accepted 0/1, dH, attempts made)."""
    for t in range(cap):
        a, dh = attempt(adj, part, stores, side0, lam, allow_break, pts1, pts2, four, ends, allpts)
        if a:
            return 1, dh, t + 1
    return 0, 0.0, cap


def work_arrays(deg):
    return (np.empty(4 + 4 * deg, dtype=np.int64), np.empty(4 + 4 * deg, dtype=np.int64),
            np.empty(4, dtype=np.int64), np.empty(8, dtype=np.int64), np.empty(2 * (4 + 4 * deg), dtype=np.int64))


@njit(cache=True)
def run_block(adj, part, stores, side0, lam, allow_break, n_sweeps, rate, pts1, pts2, four, ends, allpts):
    """n_sweeps of the driven protocol: feed rate / N into every store, then N attempts. Returns (accepted, sum dH)."""
    n = adj.shape[0]
    acc = 0
    dh_sum = 0.0
    for _ in range(n_sweeps):
        for i in range(n):
            stores[i] += rate / n
        for _ in range(n):
            a, dh = attempt(adj, part, stores, side0, lam, allow_break, pts1, pts2, four, ends, allpts)
            acc += a
            dh_sum += dh
    return acc, dh_sum


def census(adj):
    """Open (d = 3), curled (d = 2), broken (d > 3), low (d < 2) points; open regions and the largest."""
    d = local_dimension_d(adj)
    regions = pieces_of(adj, d == 3)
    return dict(open=int((d == 3).sum()), curled=int((d == 2).sum()), broken=int((d > 3).sum()),
                low=int((d < 2).sum()), regions=len(regions), largest=int(regions[0]) if len(regions) else 0)


def run_clean_feed(dims, lam, rate, seed, n_sweeps, every, allow_break=False, stop_open_share=None, adj=None,
                   part=None):
    """The slab (or `adj`, `part` if given) fed at every point under the clean two-switch move.

    Returns (adj, part, stores, rows): one row per block of `every` sweeps with the census, H, the stores' sum, the
    accepted moves in the block, and the conservation residual H + stores - fed - H0 (which the tests require to be
    zero to 1e-9). Stops early when the open share reaches `stop_open_share`, if given.
    """
    if adj is None:
        adj, part = torus(tuple(dims))
    adj = adj.copy()
    part = np.asarray(part).copy()
    n = adj.shape[0]
    side0 = np.flatnonzero(part == 0).astype(np.int64)
    stores = np.zeros(n)
    h0 = hamiltonian(adj, lam)
    np.random.seed(seed)
    seed_numba(seed)
    work = work_arrays(adj.shape[1])
    fed = 0.0
    rows = []
    for sweep in range(0, n_sweeps, every):
        block = min(every, n_sweeps - sweep)
        acc, _ = run_block(adj, part, stores, side0, lam, allow_break, block, rate, *work)
        fed += block * rate
        h = hamiltonian(adj, lam)
        row = dict(sweep=sweep + block, H=float(h), stores=float(stores.sum()), acc=int(acc),
                   residual=float(h + stores.sum() - fed - h0), **census(adj))
        rows.append(row)
        if stop_open_share is not None and row["open"] >= stop_open_share * n:
            break
    assert is_valid(adj)
    return adj, part, stores, rows
