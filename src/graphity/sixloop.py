"""A price on loops of six: the toy's own rule one loop-length up. OURS; this is not the published model.

Decided by the owner on 11 October 2026 (VISION Update 56; ASSUMPTIONS O124). The form was fixed there before any
registered run:

    H6 = 16 (N - S) + 4 lam X + mu E6,        E6 = sum over links e of (C_e - 6)_+ ,

where C_e is the number of closed loops of six distinct links through link e. S, X and lam are as in cqg.py; the
moves, the hard-core rule and the bipartition are cqg.py's, unchanged. With mu = 0 this is cqg.py's model exactly,
and `run_chain6` then makes cqg.run_chain's chain draw for draw (tested), while still keeping E6 as a number to
read.

Why 6. In a flat sheet wide enough that nothing wraps, the only loops of six are the outlines of two neighbouring
squares, and every link lies on exactly six of them. So E6 = 0 there: a wide flat torus still costs nothing. A
flat torus only six around has loops of six that go round it, its links lie on more than six, and it pays
mu E6 > 0. That is the square rule's (S_e - 2)_+ again, one loop-length up: a link "over-full at the next scale".
Having FEWER than six earns nothing, so breaking squares is not rewarded.

Counting C_e (bipartite graphs only). A loop of six through link (u, v) is u - v - p1 - p2 - p3 - p4 - u with all
six points distinct. `loops6` walks v -> p1 -> p2 -> p3 -> p4 and asks whether p4 is linked to u. Bipartiteness
puts p1, p3 on u's side and p2, p4 on v's, so the only coincidences to rule out are p1 = u, p3 = u, p3 = p1,
p2 = v, p4 = v, p4 = p2. Each loop through the link is met exactly once (the walk leaves v away from u, which
fixes the direction). Checked against networkx by brute force.

Detailed balance. The proposal is cqg.py's bipartite edge switch, which is symmetric (ASSUMPTIONS Q4). H6 is a
function of the graph alone. Acceptance is Metropolis, min(1, exp(-dH6 / g)), so with the symmetric proposal the
chain samples exp(-H6 / g) on the same space of valid graphs. An early refusal is used to save work and does not
change the rule: the uniform number r is drawn first when a lower bound on dH6 is positive, and a move with
r >= exp(-bound / g) would be refused by the exact test as well, because dH6 >= bound.

How dE6 is obtained exactly. A switch removes links e1 = (u1, v1), e2 = (u2, v2) and adds f1 = (u1, v2),
f2 = (u2, v1). A loop of six is destroyed exactly if it contains e1 or e2, and created exactly if it contains f1
or f2. `counts[a, k]` holds C_e for the link in slot k of the U-side point a. In the old graph every loop through
e1, and every loop through e2 that does not contain e1, takes one off the count of each of its other links. In
the new graph every loop through f1, and every loop through f2 that does not contain f1, adds one. The new links
take the slots of the old ones (cqg._switch keeps slots) and get their counts by direct counting. dE6 is the sum
of the changes in (C_e - 6)_+ over the links touched, plus the new links' excess, minus the old links'. A refused
move undoes every step in reverse. Checked against full recomputation after long random walks.

The lower bound used for early refusal: a destroyed loop lowers the excess of at most its six links by one each,
and at most C_e1 + C_e2 loops are destroyed, so dE6 >= -6 (C_e1 + C_e2).
"""
import numpy as np
from numba import njit

from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, _new_edge_ok, _note_square_edges, _replace,
                          _surplus_on, _switch, surplus, total_squares)
from graphity.squares import has_edge, squares_on_edge

FLAT = 6          # loops of six through every link of a wide flat sheet


@njit(cache=True)
def loops6(adj, u, v):
    """The number of closed loops of six links through link (u, v). Bipartite graphs only."""
    count = 0
    for i1 in range(4):
        p1 = adj[v, i1]
        if p1 == u:
            continue
        for i2 in range(4):
            p2 = adj[p1, i2]
            if p2 == v:
                continue
            for i3 in range(4):
                p3 = adj[p2, i3]
                if p3 == p1 or p3 == u:
                    continue
                for i4 in range(4):
                    p4 = adj[p3, i4]
                    if p4 != p2 and p4 != v and has_edge(adj, p4, u):
                        count += 1
    return count


@njit(cache=True)
def loop_counts(adj, side_u):
    """(N, 4) array: for a U-side point a, entry [a, k] is C_e of the link in its slot k. Zero on the other side."""
    counts = np.zeros(adj.shape, dtype=np.int64)
    for i in range(side_u.shape[0]):
        a = side_u[i]
        for k in range(4):
            counts[a, k] = loops6(adj, a, adj[a, k])
    return counts


@njit(cache=True)
def excess6(adj, side_u):
    """E6: the sum over links of (loops of six through the link - 6), counted only where positive."""
    total = 0
    for i in range(side_u.shape[0]):
        a = side_u[i]
        for k in range(4):
            c = loops6(adj, a, adj[a, k])
            if c > FLAT:
                total += c - FLAT
    return total


@njit(cache=True)
def hamiltonian6(adj, side_u, lam, mu):
    n = adj.shape[0]
    return (ENERGY_PER_SQUARE * (n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
            + mu * excess6(adj, side_u))


@njit(cache=True)
def _same(p, q, a, b):
    return (p == a and q == b) or (p == b and q == a)


@njit(cache=True)
def _bump(adj, is_u, counts, x, y, step, log, n_log):
    """Change the count of link (x, y) by `step`, note the slot in `log`, and return the change in its excess."""
    a, b = (x, y) if is_u[x] else (y, x)
    k = 0
    while adj[a, k] != b:
        k += 1
    old = counts[a, k]
    counts[a, k] = old + step
    log[n_log, 0] = a
    log[n_log, 1] = k
    return max(old + step - FLAT, 0) - max(old - FLAT, 0)


@njit(cache=True)
def _sweep_link(adj, is_u, counts, u, v, xa, xb, sa, sb, step, log, n_log):
    """Every loop of six through link (u, v) that does not contain link (xa, xb): change the count of each of its
    other links by `step`, except link (sa, sb). Returns (change in excess, loops met, loops met that contain
    (sa, sb), new n_log). Pass -1 for a link that is not there."""
    d_e = 0
    met = 0
    with_s = 0
    for i1 in range(4):
        p1 = adj[v, i1]
        if p1 == u:
            continue
        for i2 in range(4):
            p2 = adj[p1, i2]
            if p2 == v:
                continue
            for i3 in range(4):
                p3 = adj[p2, i3]
                if p3 == p1 or p3 == u:
                    continue
                for i4 in range(4):
                    p4 = adj[p3, i4]
                    if p4 == p2 or p4 == v or not has_edge(adj, p4, u):
                        continue
                    if xa >= 0 and (_same(v, p1, xa, xb) or _same(p1, p2, xa, xb) or _same(p2, p3, xa, xb)
                                    or _same(p3, p4, xa, xb) or _same(p4, u, xa, xb)):
                        continue
                    met += 1
                    has_s = False
                    for j in range(5):
                        if j == 0:
                            x, y = v, p1
                        elif j == 1:
                            x, y = p1, p2
                        elif j == 2:
                            x, y = p2, p3
                        elif j == 3:
                            x, y = p3, p4
                        else:
                            x, y = p4, u
                        if sa >= 0 and _same(x, y, sa, sb):
                            has_s = True
                            continue
                        d_e += _bump(adj, is_u, counts, x, y, step, log, n_log)
                        n_log += 1
                    if has_s:
                        with_s += 1
    return d_e, met, with_s, n_log


@njit(cache=True)
def _undo(counts, log, start, stop, step):
    for i in range(start, stop):
        counts[log[i, 0], log[i, 1]] -= step


@njit(cache=True)
def run_chain6(adj, side_u, inv_g, n_equil, n_meas, seed, lam, mu):
    """Markov chain for H6 at coupling g = 1 / inv_g, without the cap. Modifies adj in place.

    seed < 0 carries on the random stream from the previous call (as cqg.run_chain; ASSUMPTIONS Q14).
    Returns (S, X, E6 after each measurement sweep, acceptance rate). One sweep = 2N attempted switches.
    """
    if seed >= 0:
        np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    is_u = np.zeros(n, dtype=np.bool_)
    for i in range(nu):
        is_u[side_u[i]] = True
    edges = np.empty((64, 2), dtype=np.int64)
    log = np.empty((4096, 2), dtype=np.int64)
    counts = loop_counts(adj, side_u)
    s = total_squares(adj)
    x = surplus(adj)
    e6 = 0
    for i in range(nu):
        for k in range(4):
            if counts[side_u[i], k] > FLAT:
                e6 += counts[side_u[i], k] - FLAT
    out_s = np.zeros(n_meas, dtype=np.int64)
    out_x = np.zeros(n_meas, dtype=np.int64)
    out_e = np.zeros(n_meas, dtype=np.int64)
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
            k1 = 0
            while adj[u1, k1] != v1:
                k1 += 1
            k2 = 0
            while adj[u2, k2] != v2:
                k2 += 1
            c_e1 = counts[u1, k1]
            c_e2 = counts[u2, k2]
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
            if not (_new_edge_ok(adj, u1, v2, NO_CAP) and _new_edge_ok(adj, u2, v1, NO_CAP)):
                _switch(adj, u1, v2, u2, v1)
                continue
            n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
            n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
            after = _surplus_on(adj, edges, n_edges)
            _switch(adj, u1, v2, u2, v1)                # back to the old graph
            before = _surplus_on(adj, edges, n_edges)
            d_x = after - before
            cheap = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
            bound = cheap - mu * 6.0 * (c_e1 + c_e2)    # dE6 >= -6 (C_e1 + C_e2)
            r = -1.0
            if bound > 0.0:
                r = np.random.random()
                if r >= np.exp(-inv_g * bound):
                    continue                            # refused; the graph is the old one
            # the old graph: take the destroyed loops off their other links
            d_e = -(max(c_e1 - FLAT, 0) + max(c_e2 - FLAT, 0))
            n_log = 0
            step_e, _m, _w, n_log = _sweep_link(adj, is_u, counts, u1, v1, -1, -1, u2, v2, -1, log, n_log)
            d_e += step_e
            step_e, _m, _w, n_log = _sweep_link(adj, is_u, counts, u2, v2, u1, v1, -1, -1, -1, log, n_log)
            d_e += step_e
            n_removed = n_log
            _switch(adj, u1, v1, u2, v2)                # forward: the new graph
            step_e, c_f1, both, n_log = _sweep_link(adj, is_u, counts, u1, v2, -1, -1, u2, v1, 1, log, n_log)
            d_e += step_e
            step_e, only_f2, _w, n_log = _sweep_link(adj, is_u, counts, u2, v1, u1, v2, -1, -1, 1, log, n_log)
            d_e += step_e
            c_f2 = only_f2 + both
            d_e += max(c_f1 - FLAT, 0) + max(c_f2 - FLAT, 0)
            d_h = cheap + mu * d_e
            ok = True
            if d_h > 0.0:
                if r < 0.0:
                    r = np.random.random()
                if r >= np.exp(-inv_g * d_h):
                    ok = False
            if ok:
                accepted += 1
                s += d_s
                x += d_x
                e6 += d_e
                counts[u1, k1] = c_f1                   # the new links sit in the old links' slots
                counts[u2, k2] = c_f2
            else:
                _undo(counts, log, n_removed, n_log, 1)
                _switch(adj, u1, v2, u2, v1)
                _undo(counts, log, 0, n_removed, -1)
        if sweep >= n_equil:
            out_s[sweep - n_equil] = s
            out_x[sweep - n_equil] = x
            out_e[sweep - n_equil] = e6
    return out_s, out_x, out_e, accepted / max(attempted, 1)
