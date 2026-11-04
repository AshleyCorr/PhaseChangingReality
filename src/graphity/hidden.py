"""The hidden count of a region: how many wirings of its inside look the same from outside. Exact.

THE DEFINITION. Take a valid state of the model (cqg.py: four links per vertex, every link joining side 0 to
side 1, the hard-core rule) and a region R, any set of its vertices. Remove every link that has both ends in R.
Each vertex of R is then short of

    need(v) = 4 - (its links to vertices outside R)

links. An INTERIOR WIRING is a set of links, each joining a side-0 vertex of R to a side-1 vertex of R and no
pair joined twice, that gives every vertex of R exactly its need back. Every link with an end outside R stays
as it was. An interior wiring is VALID when the rewired graph is again a state of the model, which is
`is_valid(adj2, NO_CAP)`: four links everywhere, no repeated link, and no two vertices with more than two
common neighbors. Write dH = H(rewired) - H(original), with H = 16 (N - S) + 4 lam X as in cqg.py. Then

    valid   the number of valid interior wirings, whatever their energy;
    levels  how many of them sit at each dH;
    count   the number with dH = 0 to within 1e-9: the HIDDEN COUNT.

The original wiring is one of them, so count >= 1. Nothing is sampled and nothing is rounded except dH, to 1e-9.

WIRINGS ARE COUNTED BY THE NAMES OF THEIR VERTICES, AND THAT ALONE MAKES THE COUNT LARGE. Call a vertex of R
SEALED when all four of its links stay inside R. Exchange the names of two sealed vertices of one side: every
link to the outside is untouched, the graph is the same graph with two names swapped, so it is valid and has
the same energy, and it is a different interior wiring (two vertices with the same four neighbors would break
the hard-core rule). So a region with k0 sealed vertices on side 0 and k1 on side 1 has a count that is a
multiple of k0! and of k1!, and at least k0! k1! divided by the symmetries of the graph that move only sealed
vertices. In flat space the 12 points within one step of a square have count 4 = 2! 2!: the original and the
three renamings of the square's own four points. The result gives k0 and k1 as `sealed`, so that this part of
the count can be seen, and `shapes` (below) is the count with such renamings taken out.

Why only the model's own wirings. `scripts/hidden_count.py`, the first version, required the degrees and the
energy only. It could therefore count wirings that link two vertices of one side, or that break the hard-core
rule, and neither is a state the model can be in (tests/test_hidden.py has a case where it counts two and
the model has one).

HOW IT IS COUNTED. The side-0 vertices of R are taken one after another in a fixed order (`_search_order`),
and each chooses its partners from a fixed list of the side-1 vertices of R, going down the list, one link at
a time (`_search`). Two prunings cut the tree, and neither can lose a valid wiring:

  1. The hard-core rule. Links are only ever added during the search, so the number of common neighbors of any
     two vertices can only grow. If a new link gives some pair more than two, every wiring below that point is
     invalid. A new link (u, w) changes the common neighbors of u with the other neighbors of w, and of w with
     the other neighbors of u, and of no other pair; those are the pairs `_hard_core_holds` looks at. The graph
     with R's inside removed satisfies the rule, being part of a valid graph, and every link added is checked,
     so a wiring that reaches the end is valid without a further test (tests: `is_valid` on every wiring).
  2. Counting ahead. A side-1 vertex that is passed over by one side-0 vertex can only be served by the ones
     still to come, one link each. If it needs more than that, the branch is dead.

HOW THE ENERGY IS READ. Not from scratch. Edge by edge, H = sum_e cost(S_e) with cost(s) = 4 [(2 - s) +
lam (s - 2)_+] (VISION Update 5; `test_energy_edge_by_edge`). S_e can differ between two interior wirings only
on an edge of a square that contains an interior link, and every vertex of such a square is in R or is an
outside neighbor of R (walk round the square from the interior link: each step is a link from a vertex of R or
to one). So it is enough to sum S_e and (S_e - 2)_+ over the LOCAL edges, those with both ends in R or beside
it: the fixed ones, listed once, and the interior links of the wiring at hand. The sum of S_e over all edges is
4 S, so the first sum moves in steps of four, which is checked on every result. tests/test_hidden.py compares
both sums with full recomputation on every wiring of several cases. The full search never sees lam: it tallies
wirings by (change in S, change in X), from which `levels` follows for any lam (`square_levels` in the result).

COST, AND THE SHORTER SEARCH. The number of valid wirings grows very fast with the region: 9,216 for three
columns of a 4 x L tube; for four, 18 million had been found when the first vertex was still on its first
choice of partners, and the run was stopped. The full search visits each, so `valid` and `levels` are for
regions of up to about twelve to fourteen points. `same_energy_only=True` finds `count` alone and prunes a
third way:

  3. The energy still in reach. While links are added, S_e on an edge never falls, and it never passes 3
     (cqg.py, Q2). An edge is SETTLED when its S_e can no longer change: both its ends have their four links,
     so that a new square through it could only come from a new link between a neighbor of one end and a
     neighbor of the other, and every such pair is linked already or has a vertex with no free slot
     (`_settled`). So the final local energy lies between the sum of the least and the sum of the most each
     local edge can still cost: cost(S_e) exactly if settled, otherwise the least and the most of cost over
     S_e .. 3, and over 0 .. 3 for a link not placed yet (`_can_tie`). If the original's local energy is
     outside that range, no wiring below this point ties with it. At a complete wiring every edge is settled,
     so the range is the energy itself and only ties are tallied.

  tests/test_hidden.py checks that the shorter search finds the same wirings as the full one, case by case.

FOLDING THE RENAMINGS (`fold_renamings=True`; the numbers do not change, the search is shorter). Renaming the
sealed side-1 vertices among themselves turns a valid wiring into a valid wiring with the same S and X, and
never into itself, since two of them would need the same four neighbors. So the valid wirings come in classes
of exactly k1! each, all in one cell of the tally. Give each sealed side-1 vertex the list of its four
neighbors, by their place in the search order. In every class exactly one wiring has these lists increasing,
in dictionary order, down the search's list of side-1 vertices; the folded search visits only those and
counts each k1! times (pruning 4). Two lists compare by the first side-0 vertex that is linked to one of the
two vertices and not to the other, so the order is checked each time a side-0 vertex is finished (`_in_order`).
`wirings` then holds one wiring of each class. The sealed side-0 vertices are not folded as well: the two
foldings together would be exact only if no renaming of both sides at once could leave a wiring unchanged,
which is not always so.

How long the shorter search takes depends strongly on the order in which the vertices are taken. Measured on
one 4 x 5 block of the flat torus, taking the vertices in the order of their names after renaming the graph
breadth-first from a corner of the block, from the middle of a side, or from inside: 0.5 s, 160 s and 14 s.
`_search_order` is a rule of thumb that starts at the region's edge. Nothing says it is the best order, and
because its ties go to the lower name, the same window under other names can take a very different time.
Measured with it, 2026-10-05, lam = 1.25, folded: that 4 x 5 block 0.2 s; four columns of the tube (16 points)
98 s; the 24 points within two steps of a square of the flat torus not finished in 25 minutes. (That last
window did finish once, in eleven minutes, in the order of the names after a breadth-first renaming from its
edge: count 518,400 = 6! 6!.)

SHAPES. Two same-energy wirings that differ only by a renaming of sealed vertices are one graph under two sets
of names. `shapes` counts the same-energy wirings up to such renamings: sealed vertices may exchange names
within their side, every vertex with a link out of the region keeps its own. So 1 <= shapes <= count, and
shapes = 1 says the region hides nothing but the names of its sealed points.

`_search` is a loop and not a function that calls itself because numba's on-disk cache crashed (an access
violation, numba 0.67, 2026-10-05) on reloading the self-calling version in a second process.

Nothing here interprets the count; this module only counts.
"""
import itertools
from collections import deque, namedtuple
from math import factorial

import numpy as np
from numba import njit

from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, MAX_CODEGREE, NO_CAP, _replace, codegree,
                          hamiltonian, is_valid)
from graphity.squares import has_edge, squares_on_edge

SAME_ENERGY = 1e-9         # |dH| at or below this is "the same energy"; dH is also rounded to this
EMPTY = -1                 # a slot whose link has been removed (the repository's convention)
SHAPES_KEEP = 5000         # hidden_count works out `shapes` when the search found at most this many ties
SHAPES_WORK = 1000000      # ... and `shapes` gives up (None) if it would have to look at more renamed wirings

# Everything the search reads and writes, in one bundle so that `_search` has one argument.
#   work       (N, 4) neighbor array, R's interior links removed; the search adds and removes links in it
#   side0      the side-0 vertices of R that need links, in the order of the search (_search_order)
#   owner      (M,) the links to place are numbered 0 .. M-1; link d belongs to side-0 vertex number owner[d]
#   side1      the side-1 vertices of R that need links, in their fixed order;  need1: how many each STILL needs
#   links      (M, 2) the links placed so far, as (side-0 vertex, side-1 vertex); full at a complete wiring
#   fixed      (F, 2) the local edges that are the same in every wiring (see the module docstring)
#   tally      tally[a, x] = number of wirings found whose local sums are a = sum S_e and x = sum (S_e - 2)_+
#   wanted     same shape, true where a wiring's links should be kept
#   kept       (capacity, M, 2) the kept wirings;  kept_sums: (capacity, 2) their (a, x);  n_kept: (1,) how many
#   same_only  true for the shorter search (pruning 3);  target: the original's local energy
#   cost       cost[s] of an edge with s squares;  low[s], high[s]: the least and most of cost over s .. 3
#   sealed1    positions in side1 of the sealed side-1 vertices if the renamings are folded (pruning 4), else empty
_Job = namedtuple("_Job", "work side0 owner side1 need1 links fixed tally wanted kept kept_sums n_kept "
                          "same_only target cost low high sealed1")


@njit(cache=True)
def _hard_core_holds(work, u, w):
    """After adding the link (u, w): does every pair it could have affected still share at most two neighbors?"""
    for k in range(4):
        x = work[w, k]                          # x and u now share w
        if x >= 0 and x != u and codegree(work, u, x) > MAX_CODEGREE:
            return False
        y = work[u, k]                          # y and w now share u
        if y >= 0 and y != w and codegree(work, w, y) > MAX_CODEGREE:
            return False
    return True


@njit(cache=True)
def _sums_over(work, edges):
    """(sum of S_e, sum of (S_e - 2)_+) over the listed edges."""
    total = 0
    over = 0
    for e in range(edges.shape[0]):
        s_e = squares_on_edge(work, edges[e, 0], edges[e, 1])
        total += s_e
        if s_e > 2:
            over += s_e - 2
    return total, over


@njit(cache=True)
def _full(work, v):
    """Does v have all four of its links?"""
    for k in range(4):
        if work[v, k] < 0:
            return False
    return True


@njit(cache=True)
def _settled(work, p, q):
    """Is the number of squares on the edge (p, q) final, whatever links are still added? (Pruning 3.)"""
    if not (_full(work, p) and _full(work, q)):
        return False                            # an end with a free slot can still gain a neighbor
    for i in range(4):
        a = work[q, i]
        if a == p or _full(work, a):
            continue
        for j in range(4):
            b = work[p, j]
            if b == q or _full(work, b) or has_edge(work, a, b):
                continue
            return False                        # the link (a, b) could still come, closing p - q - a - b
    return True


@njit(cache=True)
def _range_over(work, edges, n, cost, low, high):
    """(least, most) that the first n listed edges can cost in total at the end of the search."""
    least = 0.0
    most = 0.0
    for e in range(n):
        s_e = squares_on_edge(work, edges[e, 0], edges[e, 1])
        if _settled(work, edges[e, 0], edges[e, 1]):
            least += cost[s_e]
            most += cost[s_e]
        else:
            least += low[s_e]
            most += high[s_e]
    return least, most


@njit(cache=True)
def _can_tie(job, placed):
    """With `placed` links in, can the finished wiring still have the original's energy? (Pruning 3.)"""
    least_f, most_f = _range_over(job.work, job.fixed, job.fixed.shape[0], job.cost, job.low, job.high)
    least_l, most_l = _range_over(job.work, job.links, placed, job.cost, job.low, job.high)
    to_come = job.links.shape[0] - placed       # a link not placed yet can end with 0 to 3 squares
    least = least_f + least_l + to_come * job.low[0]
    most = most_f + most_l + to_come * job.high[0]
    return least <= job.target + SAME_ENERGY and most >= job.target - SAME_ENERGY


@njit(cache=True)
def _in_order(job, i, u, decided):
    """Side-0 vertex number i, which is u, has just got its partners: are the sealed side-1 vertices still in
    order? (Pruning 4.) decided[k] is the number of the side-0 vertex at which sealed vertices k and k + 1 first
    differed, the earlier one in job.side1 being the one linked; -1 while every vertex so far links both or neither."""
    for k in range(job.sealed1.shape[0] - 1):
        if decided[k] >= i:                     # written on a branch the search has since left
            decided[k] = -1
        if decided[k] < 0:
            lower = has_edge(job.work, u, job.side1[job.sealed1[k]])
            upper = has_edge(job.work, u, job.side1[job.sealed1[k + 1]])
            if upper and not lower:
                return False                    # the later vertex would get the smaller list
            if lower and not upper:
                decided[k] = i
    return True


@njit(cache=True)
def _record(job):
    """A complete wiring is in job.work: tally its two local sums, and keep its links if wanted and there is room."""
    a_fixed, x_fixed = _sums_over(job.work, job.fixed)
    a_links, x_links = _sums_over(job.work, job.links)
    a = a_fixed + a_links
    x = x_fixed + x_links
    job.tally[a, x] += 1
    n = job.n_kept[0]
    if job.wanted[a, x] and n < job.kept.shape[0]:
        job.kept[n] = job.links
        job.kept_sums[n, 0] = a
        job.kept_sums[n, 1] = x
        job.n_kept[0] = n + 1


@njit(cache=True)
def _search(job):
    """Visit every valid interior wiring once (every one that ties, if job.same_only) and `_record` it.

    The links to place are numbered d = 0 .. M-1: the first side-0 vertex's, then the second's, and so on.
    Link d tries its possible partners in the order of job.side1; pick[d] is the one it holds now, as a position
    in job.side1. A vertex's second link starts after its first one's partner, so each set of partners is tried
    once. The loop works like a mileage counter: it stands on link d, moves it on to its next partner, steps
    forward to link d + 1 if all is well so far, and steps back to link d - 1 when link d has no partner left.
    """
    n0 = job.side0.shape[0]
    n1 = job.side1.shape[0]
    m = job.links.shape[0]
    pick = np.empty(m, dtype=np.int64)
    decided = np.full(max(job.sealed1.shape[0] - 1, 1), -1, dtype=np.int64)     # see _in_order
    d = 0
    pick[0] = -1                                # "nothing tried yet"
    while d >= 0:
        i = job.owner[d]
        u = job.side0[i]
        later = n0 - 1 - i                      # side-0 vertices still to come after this one
        start = -1                              # link d tries the partners after `start`
        if d > 0 and job.owner[d - 1] == i:
            start = pick[d - 1]
        jj = pick[d]
        if jj > start:                          # link d sits on partner jj from the last round: take it out
            w = job.side1[jj]
            _replace(job.work, u, w, EMPTY)
            _replace(job.work, w, u, EMPTY)
            job.need1[jj] += 1
            if job.need1[jj] > later:           # passing over this partner would leave it short for good (2)
                d -= 1
                continue
        jj += 1
        while jj < n1 and job.need1[jj] == 0:   # the next partner that still needs a link
            jj += 1
        if jj == n1:                            # none left: step back
            d -= 1
            continue
        w = job.side1[jj]
        pick[d] = jj
        _replace(job.work, u, EMPTY, w)         # put the link (u, w) in
        _replace(job.work, w, EMPTY, u)
        job.need1[jj] -= 1
        job.links[d, 0] = u
        job.links[d, 1] = w
        if not _hard_core_holds(job.work, u, w):
            continue                            # (1); the next round takes the link out and moves on
        if d + 1 < m and job.owner[d + 1] == i:  # the same vertex needs another partner, after this one
            d += 1
            pick[d] = jj
            continue
        short = False                           # vertex i is done: the partners after jj are passed over too
        for other in range(jj + 1, n1):
            if job.need1[other] > later:
                short = True
        if short:
            continue                            # (2)
        if not _in_order(job, i, u, decided):
            continue                            # (4); always true when nothing is folded
        if job.same_only and not _can_tie(job, d + 1):
            continue                            # (3)
        if d + 1 == m:
            _record(job)                        # every link is placed and, by (2), no vertex is short
            continue
        d += 1                                  # on to the next vertex's first link
        pick[d] = -1


def _check_input(adj, part, region):
    """Return (adj, part, sorted region) as clean arrays, or say what is wrong with them."""
    adj = np.ascontiguousarray(adj, dtype=np.int64)
    part = np.asarray(part, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 4 or part.shape != (adj.shape[0],):
        raise ValueError("adj must be an (N, 4) neighbor array and part must give a side, 0 or 1, for each vertex")
    region = sorted({int(v) for v in region})
    if region and not 0 <= region[0] <= region[-1] < adj.shape[0]:
        raise ValueError("the region names a vertex the graph does not have")
    if not is_valid(adj, NO_CAP):
        raise ValueError("the graph is not a valid state of the model (is_valid with NO_CAP)")
    if not np.isin(part, (0, 1)).all() or (part[adj] == part[:, None]).any():
        raise ValueError("part must be 0 or 1 and every link must join side 0 to side 1")
    return adj, part, region


def _search_order(adj, part, region, need):
    """(side-0 vertices, side-1 vertices) of the region that need links, in the order the search takes them.

    Any order gives the same counts; this one usually gives them sooner than the order of the names (module
    docstring). It starts at the region's edge, with the side-0 vertex that has the fewest links inside, and
    walks breadth-first: after a vertex come the side-0 vertices two steps from it inside the region. The
    side-1 vertices are listed as the walk first meets them. Ties go to the lower name, so the order does not
    depend on how the rows of adj are arranged.
    """
    inside = set(region)
    order0, order1, seen = [], [], set()
    for start in sorted((v for v in region if part[v] == 0 and need[v] > 0), key=lambda v: (need[v], v)):
        if start in seen:
            continue                                        # reached already; a new start only for a separate piece
        seen.add(start)
        queue = deque([start])
        while queue:
            v = queue.popleft()
            order0.append(v)
            for w in sorted(int(w) for w in adj[v] if int(w) in inside):
                if w not in seen:
                    seen.add(w)
                    order1.append(w)
                for x in sorted(int(x) for x in adj[w] if int(x) in inside):
                    if x not in seen:
                        seen.add(x)
                        queue.append(x)
    return np.array(order0, dtype=np.int64), np.array(order1, dtype=np.int64)


def _prepare(adj, part, region):
    """Remove the region's interior links and list what the search needs.

    Returns (work, side0, need0, side1, need1, fixed, original, cut): `original` is the wiring that was removed,
    as increasing (side-0 vertex, side-1 vertex) pairs; `cut` the number of links with exactly one end in R.
    """
    inside = np.zeros(adj.shape[0], dtype=bool)
    inside[region] = True
    work = adj.copy()
    work[inside[:, None] & inside[adj]] = EMPTY             # a slot of a region vertex pointing into the region
    need = (work == EMPTY).sum(axis=1).astype(np.int64)
    cut = int((inside[:, None] & ~inside[adj]).sum())
    original = sorted((v, int(w)) for v in region if part[v] == 0 for w in adj[v] if inside[w])
    side0, side1 = _search_order(adj, part, region, need)
    beside = inside.copy()                                  # R and its outside neighbors
    beside[adj[region].ravel()] = True
    fixed = [(p, int(q)) for p in np.flatnonzero(beside) for q in work[p] if q > p and beside[q]]
    fixed = np.array(fixed, dtype=np.int64).reshape(-1, 2)
    return work, side0, need[side0], side1, need[side1], fixed, original, cut


def _edge_costs(lam):
    """(cost, low, high): an edge with s squares costs cost[s] = 4 [(2 - s) + lam (s - 2)_+], s = 0 .. 3;
    low[s] and high[s] are the least and the most of cost[s], ..., cost[3]."""
    cost = np.array([ENERGY_PER_SQUARE / 4 * (2 - s) + ENERGY_PER_SURPLUS * lam * max(0, s - 2) for s in range(4)])
    low = np.array([cost[s:].min() for s in range(4)])
    high = np.array([cost[s:].max() for s in range(4)])
    return cost, low, high


def _energy_change(d_squares, d_surplus, lam):
    """dH for a change of d_squares in S and d_surplus in X, with anything within SAME_ENERGY of zero set to zero."""
    d_h = -ENERGY_PER_SQUARE * d_squares + ENERGY_PER_SURPLUS * lam * d_surplus
    return np.where(np.abs(d_h) <= SAME_ENERGY, 0.0, np.round(d_h, 9))


def _run(adj, part, region, keep, lam, same_only, fold):
    """Do the search. Wirings are kept, up to `keep` of them, where dH = 0 at this lam, or everywhere if lam is None.

    Returns (square_levels, kept, original, cut): square_levels maps (change in S, change in X) to the number of
    wirings there, and kept is a list of (links, change in S, change in X). With `fold` the search visits one
    wiring of each class of renamings of the sealed side-1 vertices, and each stands for k1! in square_levels.
    """
    adj, part, region = _check_input(adj, part, region)
    work, side0, need0, side1, need1, fixed, original, cut = _prepare(adj, part, region)
    n_links = len(original)
    if n_links == 0:                                        # nothing inside to rewire: the one wiring is the empty one
        return {(0, 0): 1}, [((), 0, 0)][:keep], original, cut
    a_fixed, x_fixed = _sums_over(adj, fixed)               # the original's two local sums, read from adj itself
    a_links, x_links = _sums_over(adj, np.array(original, dtype=np.int64))
    a0, x0 = a_fixed + a_links, x_fixed + x_links
    n_local = len(fixed) + n_links                          # an edge carries at most 3 squares (cqg.py, Q2)
    d_squares = (np.arange(3 * n_local + 1)[:, None] - a0) / 4.0
    d_surplus = np.arange(n_local + 1)[None, :] - x0
    if lam is None:
        wanted = np.ones((3 * n_local + 1, n_local + 1), dtype=bool)
    else:
        wanted = _energy_change(d_squares, d_surplus, lam) == 0.0
    lam_or_zero = 0.0 if lam is None else float(lam)        # the full search does not use the three lines below
    cost, low, high = _edge_costs(lam_or_zero)
    target = float(ENERGY_PER_SQUARE / 4 * (2 * n_local - a0) + ENERGY_PER_SURPLUS * lam_or_zero * x0)
    keep = max(int(keep), 0)
    sealed1 = np.flatnonzero(need1 == 4) if fold else np.zeros(0, dtype=np.int64)
    job = _Job(work.copy(), side0, np.repeat(np.arange(len(side0), dtype=np.int64), need0), side1, need1.copy(),
               np.zeros((n_links, 2), dtype=np.int64), fixed, np.zeros(wanted.shape, dtype=np.int64), wanted,
               np.zeros((keep, n_links, 2), dtype=np.int64), np.zeros((keep, 2), dtype=np.int64),
               np.zeros(1, dtype=np.int64), bool(same_only), target, cost, low, high, sealed1.astype(np.int64))
    _search(job)
    assert (job.need1 == need1).all() and (job.work == work).all(), "the search did not clean up after itself"
    square_levels = {}
    for a, x in sorted(zip(*np.nonzero(job.tally))):
        assert (a - a0) % 4 == 0, "the sum of S_e over the local edges must move in steps of four"
        square_levels[(int(a - a0) // 4, int(x - x0))] = int(job.tally[a, x]) * factorial(len(sealed1))
    kept = [(tuple(sorted((int(u), int(w)) for u, w in job.kept[n])),
             int(job.kept_sums[n, 0] - a0) // 4, int(job.kept_sums[n, 1] - x0)) for n in range(job.n_kept[0])]
    return square_levels, kept, original, cut


def hidden_count(adj, part, region, lam, keep=10, same_energy_only=False, fold_renamings=False):
    """The hidden count of `region` (any iterable of vertices) in the valid graph `adj` with sides `part`.

    Returns a dict:
      count          valid interior wirings with the original's energy at this lam, the original included
      valid          valid interior wirings at any energy
      levels         {dH: how many valid wirings}, dH = H(rewired) - H(original) rounded to 1e-9, increasing
      square_levels  {(change in S, change in X): how many}; does not depend on lam
      points         vertices in the region
      cut            links with exactly one end in the region
      links          links with both ends in the region (the ones that are rewired)
      sealed         (k0, k1): region vertices of side 0 and of side 1 with all four links inside the region
      renamings      k0! k1!, the number of ways to rename the sealed vertices (see the module docstring)
      energy         H of the original graph at this lam
      shapes         same-energy wirings up to renaming of the sealed vertices (see `shapes`); None if the
                     search found more than SHAPES_KEEP ties or the renamings are too many to try
      wirings        the first `keep` same-energy wirings found, each a tuple of (side-0 vertex, side-1 vertex)
                     links in increasing order; `rewire` turns one back into a graph
    With same_energy_only=True the search is the shorter one of the module docstring: `count` and `wirings` are
    the same, `valid` is None, and `levels` and `square_levels` hold the same-energy wirings only.
    With fold_renamings=True every number is the same and is found sooner where several side-1 vertices are
    sealed; `wirings` then holds one wiring from each class of renamings of those vertices.
    The inputs are not changed, and the order of the neighbors within a row of adj makes no difference.
    """
    region = sorted({int(v) for v in region})
    keep = max(int(keep), 0)
    square_levels, kept, original, cut = _run(adj, part, region, max(keep, SHAPES_KEEP), lam, same_energy_only,
                                              fold_renamings)
    levels = {}
    for (d_squares, d_surplus), n in square_levels.items():
        d_h = float(_energy_change(d_squares, d_surplus, lam))
        levels[d_h] = levels.get(d_h, 0) + n
    part = np.asarray(part)
    links_inside = np.bincount(np.array(original, dtype=np.int64).ravel(), minlength=len(part))
    sealed = tuple(sum(1 for v in region if part[v] == side and links_inside[v] == 4) for side in (0, 1))
    count = levels.get(0.0, 0)
    wirings = [wiring for wiring, _, _ in kept]
    found = count // factorial(sealed[1]) if fold_renamings else count      # ties the search itself visited
    return {"count": count,
            "valid": None if same_energy_only else sum(square_levels.values()),
            "levels": dict(sorted(levels.items())),
            "square_levels": square_levels,
            "points": len(region),
            "cut": cut,
            "links": len(original),
            "sealed": sealed,
            "renamings": factorial(sealed[0]) * factorial(sealed[1]),
            "energy": float(hamiltonian(np.ascontiguousarray(adj, dtype=np.int64), lam)),
            "shapes": shapes(adj, part, region, wirings) if len(wirings) == found else None,
            "wirings": wirings[:keep]}


def _up_to_one_side(wiring, names, dealt, side):
    """`wiring` with its vertices renamed by `names`, written so that every renaming of the vertices in `dealt`
    (sealed, all on `side`) gives the same thing: (the links touching none of them, their neighbor lists sorted).
    Renaming those vertices only deals their four-neighbor lists out among them again."""
    rest, lists = [], {}
    for link in wiring:
        link = (names.get(link[0], link[0]), names.get(link[1], link[1]))
        if link[side] in dealt:
            lists.setdefault(link[side], []).append(link[1 - side])
        else:
            rest.append(link)
    return tuple(sorted(rest)), tuple(sorted(tuple(sorted(neighbors)) for neighbors in lists.values()))


def shapes(adj, part, region, wirings):
    """How many of `wirings` are different up to renaming the region's sealed vertices?

    A renaming exchanges names among the sealed vertices of side 0 and among those of side 1 (sealed: all four
    links inside the region) and leaves every other vertex alone. `wirings` is a list as `hidden_count` returns
    it. It should hold every same-energy wiring; a list holding one from each class of the folded search gives
    the same answer.

    How: the sealed vertices of the side that has more of them are dealt with by `_up_to_one_side`; those of
    the other side are renamed in every possible way, one renaming at a time. Each wiring not yet reached from
    an earlier one is a new shape, and everything it can be renamed into is marked as reached. Returns None if
    that could mean looking at more than SHAPES_WORK renamed wirings.
    """
    inside = {int(v) for v in region}
    sealed = [[v for v in sorted(inside) if part[v] == side and all(int(w) in inside for w in adj[v])]
              for side in (0, 1)]
    tried, dealt_side = (0, 1) if len(sealed[0]) <= len(sealed[1]) else (1, 0)
    if factorial(len(sealed[tried])) * len(wirings) > SHAPES_WORK:
        return None
    dealt = set(sealed[dealt_side])
    reached, n_shapes = set(), 0
    for wiring in wirings:
        if _up_to_one_side(wiring, {}, dealt, dealt_side) in reached:
            continue
        n_shapes += 1
        for new in itertools.permutations(sealed[tried]):
            reached.add(_up_to_one_side(wiring, dict(zip(sealed[tried], new)), dealt, dealt_side))
    return n_shapes


def valid_wirings(adj, part, region, limit=100000):
    """The valid interior wirings themselves, at any energy, up to `limit` of them, in the order they are found.

    Each is (links, change in S, change in X), with links as in `hidden_count`. For looking at them and for tests.
    """
    return _run(adj, part, sorted({int(v) for v in region}), limit, None, False, False)[1]


def rewire(adj, region, wiring):
    """A copy of adj with the region's interior links replaced by `wiring`, a list of (u, w) pairs inside the region.

    Nothing is checked beyond there being a free slot for every link; ask `is_valid` about the result.
    """
    adj = np.array(adj, dtype=np.int64)
    inside = np.zeros(adj.shape[0], dtype=bool)
    inside[[int(v) for v in region]] = True
    adj[inside[:, None] & inside[adj]] = EMPTY
    for u, w in wiring:
        if not (inside[u] and inside[w]) or EMPTY not in adj[u] or EMPTY not in adj[w]:
            raise ValueError(f"the link ({u}, {w}) is not inside the region or has no free slot")
        _replace(adj, u, EMPTY, w)
        _replace(adj, w, EMPTY, u)
    return adj
