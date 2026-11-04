"""Sealed runs at any dimension D with the direction tie of VISION Update 30 in its "follow" form.

WHY. The owner decided (VISION Update 30) that the energy gets a term tying the large directions together, and the
working form, written before its exact test (ASSUMPTIONS O68), is the "follow" form

    H_tie = kappa * T,   T = sum over points v of f(d(v)),   f(d) = D - d for 1 <= d <= D, f(0) = 0,
                                                            f(d) = 0 for d > D (a broken point),

with d(v) = graphity.dimension.local_dimension_d (the pairs of links at v that close no square: D on flat space, 0 in
a cube). A fully curled point and a flat point cost nothing; a point with some directions open and some curled costs
kappa for each still curled. O68 priced every single move exactly; whether a run then opens all the directions
together is a question about the dynamics, which this module asks (PREREGISTRATION T44). At kappa != 0 (or lambda != 1)
this is our family around the model, never combinatorial quantum gravity (VISION Updates 22 and 30).

THE RULE. sealed_d.run_sealed_bath_d with dH = -16 dS + 4 lam dX + kappa dT, and nothing else changed: the same draws
in the same order, the same store rule. At kappa = 0 it is therefore run_sealed_bath_d draw for draw (tested).

WHICH POINTS CAN CHANGE d (ours, the argument the incremental update rests on; tested against full recomputation).
A switch removes the links (u1, v1), (u2, v2) and adds (u1, v2), (u2, v1). The neighbour lists of u1, v1, u2, v2
change, so their d may change. Any other point w keeps its links; the status of a pair (a, b) of them changes only if
some common neighbour c != w of a and b gains or loses a link to a or b, that is, a link a-c is one of the four
switched ones while w-a, w-b and b-c are present. Then w-a-c-b is a square through a switched link, lost if a-c was
removed and gained if it was added, and w is one of its corners. `_note_square_edges` lists every edge of every such
square (before the move for the removed links, after it for the added ones), so the points to re-read are the four
switch points and every endpoint of a noted edge.

DETAILED BALANCE. Unchanged from sealed.py (ASSUMPTIONS Q12, Q22): the proposal is symmetric and the store rule is
the demon rule for the total energy, which now includes kappa * T.
"""
import numpy as np
from numba import njit

from graphity.cqg_d import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _new_edge_ok, _note_square_edges, _replace,
                            _surplus_on, _switch, has_edge, squares_on_edge, surplus, total_squares)
from graphity.dimension import _share_d, local_dimension_d


@njit(cache=True)
def follow(d, dim):
    """f(d) of the follow form, per unit kappa."""
    if 1 <= d <= dim:
        return dim - d
    return 0


@njit(cache=True)
def _d_at(adj, v):
    """local_dimension_d at one point."""
    deg = adj.shape[1]
    open_pairs = 0
    for i in range(deg):
        a = adj[v, i]
        if a < 0:
            continue
        for j in range(i + 1, deg):
            b = adj[v, j]
            if b < 0:
                continue
            if not _share_d(adj, a, b, v):
                open_pairs += 1
    return open_pairs


@njit(cache=True)
def tie_total(adj):
    """T = sum over points of f(d)."""
    dim = adj.shape[1] // 2
    d = local_dimension_d(adj)
    t = 0
    for v in range(adj.shape[0]):
        t += follow(d[v], dim)
    return t


@njit(cache=True)
def _tie_on(adj, verts, n_verts):
    dim = adj.shape[1] // 2
    t = 0
    for i in range(n_verts):
        t += follow(_d_at(adj, verts[i]), dim)
    return t


@njit(cache=True)
def _add_vert(verts, n_verts, mark, stamp, v):
    if mark[v] != stamp:
        mark[v] = stamp
        verts[n_verts] = v
        n_verts += 1
    return n_verts


@njit(cache=True)
def run_sealed_bath_tie_d(adj, side_u, demons, n_sweeps, seed, lam=1.0, kappa=0.0, by_vertex=False):
    """Sealed run with the follow tie. Modifies adj and demons.

    Returns (S after each sweep, X after each sweep, T after each sweep, mean store energy, total store energy,
    acceptance rate). One sweep = 2N attempted switches. seed < 0 carries the random stream on (Q14).
    """
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    verts = np.empty(2 * edges.shape[0] + 4, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    stamp = 0
    s = total_squares(adj)
    x = surplus(adj)
    t = tie_total(adj)
    c = demons.shape[0]
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    out_t = np.zeros(n_sweeps, dtype=np.int64)
    out_mean = np.zeros(n_sweeps, dtype=np.float64)
    out_sum = np.zeros(n_sweeps, dtype=np.float64)
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
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
            d_t = 0
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                stamp += 1
                n_verts = 0
                n_verts = _add_vert(verts, n_verts, mark, stamp, u1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, u2)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v2)
                for i in range(n_edges):
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 0])
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 1])
                after = _surplus_on(adj, edges, n_edges)
                t_after = _tie_on(adj, verts, n_verts)
                _switch(adj, u1, v2, u2, v1)
                before = _surplus_on(adj, edges, n_edges)
                t_before = _tie_on(adj, verts, n_verts)
                _switch(adj, u1, v1, u2, v2)
                d_x = after - before
                d_t = t_after - t_before
                d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x + kappa * d_t
                if by_vertex:
                    k = u1
                else:
                    k = 0 if c == 1 else np.random.randint(0, c)
                if d_h > demons[k]:
                    ok = False
                else:
                    demons[k] -= d_h
            if ok:
                s += d_s
                x += d_x
                t += d_t
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        out_s[sweep] = s
        out_x[sweep] = x
        out_t[sweep] = t
        out_sum[sweep] = demons.sum()
        out_mean[sweep] = out_sum[sweep] / c
    return out_s, out_x, out_t, out_mean, out_sum, accepted / max(attempted, 1)


@njit(cache=True)
def table_total(adj, ftab):
    """T_f = sum over points of ftab[d] (0 for d beyond the table: a broken point)."""
    d = local_dimension_d(adj)
    t = 0.0
    for v in range(adj.shape[0]):
        if d[v] < ftab.shape[0]:
            t += ftab[d[v]]
    return t


@njit(cache=True)
def _table_on(adj, ftab, verts, n_verts):
    t = 0.0
    for i in range(n_verts):
        dv = _d_at(adj, verts[i])
        if dv < ftab.shape[0]:
            t += ftab[dv]
    return t


@njit(cache=True)
def run_sealed_bath_table_d(adj, side_u, demons, n_sweeps, seed, lam, ftab, by_vertex=False):
    """Sealed run with a tie of any shape: ftab[d] is the tie energy of a point with d open directions (length D + 1;
    a broken point, d > D, costs 0). PREREGISTRATION T45. The follow form at kappa is ftab = kappa (0, D - 1, ..., 1, 0);
    with that table and kappa a power of two this is run_sealed_bath_tie_d draw for draw (tested). The same affected
    set and the same draws as run_sealed_bath_tie_d; returns (S, X, T_f, mean store, total store, acceptance)."""
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    verts = np.empty(2 * edges.shape[0] + 4, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    stamp = 0
    s = total_squares(adj)
    x = surplus(adj)
    t = table_total(adj, ftab)
    c = demons.shape[0]
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    out_t = np.zeros(n_sweeps, dtype=np.float64)
    out_mean = np.zeros(n_sweeps, dtype=np.float64)
    out_sum = np.zeros(n_sweeps, dtype=np.float64)
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
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
            d_t = 0.0
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                stamp += 1
                n_verts = 0
                n_verts = _add_vert(verts, n_verts, mark, stamp, u1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, u2)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v2)
                for i in range(n_edges):
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 0])
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 1])
                after = _surplus_on(adj, edges, n_edges)
                t_after = _table_on(adj, ftab, verts, n_verts)
                _switch(adj, u1, v2, u2, v1)
                before = _surplus_on(adj, edges, n_edges)
                t_before = _table_on(adj, ftab, verts, n_verts)
                _switch(adj, u1, v1, u2, v2)
                d_x = after - before
                d_t = t_after - t_before
                d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x + d_t
                if by_vertex:
                    k = u1
                else:
                    k = 0 if c == 1 else np.random.randint(0, c)
                if d_h > demons[k]:
                    ok = False
                else:
                    demons[k] -= d_h
            if ok:
                s += d_s
                x += d_x
                t += d_t
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        out_s[sweep] = s
        out_x[sweep] = x
        out_t[sweep] = t
        out_sum[sweep] = demons.sum()
        out_mean[sweep] = out_sum[sweep] / c
    return out_s, out_x, out_t, out_mean, out_sum, accepted / max(attempted, 1)


@njit(cache=True)
def run_chain_table_d(adj, side_u, inv_g, n_sweeps, seed, lam, ftab):
    """The thermal chain (Metropolis at coupling g = 1 / inv_g) with a tie of any shape: cqg_d.run_chain with
    dH = -16 dS + 4 lam dX + dT_f and nothing else changed. Added 2026-10-09 for PREREGISTRATION T56 (the reservoir
    test under the owner's tie, VISION Update 47). The same draws in the same order as run_chain, and the acceptance
    draw taken in the same cases (dH > 0), so with ftab all zero it is run_chain draw for draw (tested); the affected
    set and the incremental dT_f are run_sealed_bath_table_d's (tested against full recomputation). Modifies adj in
    place; seed < 0 carries on the stream. Returns (S after each sweep, X after each, T_f after each, acceptance).

    DETAILED BALANCE: the proposal is symmetric (as run_chain) and the acceptance is Metropolis in the total energy
    H + T_f, so the chain samples exp(-(H + T_f) / g).
    """
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    verts = np.empty(2 * edges.shape[0] + 4, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    stamp = 0
    s = total_squares(adj)
    x = surplus(adj)
    t = table_total(adj, ftab)
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    out_t = np.zeros(n_sweeps, dtype=np.float64)
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
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
            d_t = 0.0
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                stamp += 1
                n_verts = 0
                n_verts = _add_vert(verts, n_verts, mark, stamp, u1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v1)
                n_verts = _add_vert(verts, n_verts, mark, stamp, u2)
                n_verts = _add_vert(verts, n_verts, mark, stamp, v2)
                for i in range(n_edges):
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 0])
                    n_verts = _add_vert(verts, n_verts, mark, stamp, edges[i, 1])
                after = _surplus_on(adj, edges, n_edges)
                t_after = _table_on(adj, ftab, verts, n_verts)
                _switch(adj, u1, v2, u2, v1)
                before = _surplus_on(adj, edges, n_edges)
                t_before = _table_on(adj, ftab, verts, n_verts)
                _switch(adj, u1, v1, u2, v2)
                d_x = after - before
                d_t = t_after - t_before
                d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x + d_t
                if d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                    ok = False
            if ok:
                s += d_s
                x += d_x
                t += d_t
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        out_s[sweep] = s
        out_x[sweep] = x
        out_t[sweep] = t
    return out_s, out_x, out_t, accepted / max(attempted, 1)


def energy_table_d(adj, lam, ftab):
    """H + T_f at the dimension of adj."""
    n, deg = adj.shape
    d = deg // 2
    h = ENERGY_PER_SQUARE * (d * (d - 1) // 2 * n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
    return h + table_total(adj, ftab)


def energy_tie_d(adj, lam, kappa):
    """H + kappa T at the dimension of adj."""
    n, deg = adj.shape
    d = deg // 2
    h = ENERGY_PER_SQUARE * (d * (d - 1) // 2 * n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
    return h + kappa * tie_total(adj)
