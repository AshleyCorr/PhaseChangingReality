"""Sealed and leaky runs: the energy given off stays in the system, or escapes at a chosen rate.

WHY. Every run so far has been at a fixed temperature, which means an unlimited bath:
the energy a conversion gives off is carried away at once and can do nothing. VISION
claims 5 and 6 are about what happens when it does not. The author put the three cases
plainly (design brief, "what happens to the released energy"):

    wide open      the energy is carried away              our other runs
    semi-permeable it leaks out at a limited rate          leak between 0 and 1 here
    sealed         it all stays                            leak = 0 here

THE METHOD [Creutz 1983, from general knowledge, not read by us; the standard
"demon"]. Beside the graph we keep one number, the DEMON, which holds energy and is
never allowed to go negative. A switch with energy change dH is accepted exactly when
the demon can pay for it, dH <= demon; then demon -= dH. A move that lowers the
graph's energy hands the difference to the demon, which can then spend it on later
moves. Nothing else changes: the same moves, the same configuration space, the same
validity rules as cqg.run_chain.

WHAT IS EXACT. H + demon + (whatever has leaked) is conserved algebraically; floating-point
arithmetic can introduce rounding error, which is checked in tests. The
run conserves energy by construction, which is the bookkeeping VISION claim 6 asks
for. With leak = 0 nothing escapes, so H + demon is fixed. Both are tested.

WHY A DEMON IS ENOUGH. At fixed total energy the graph alone is not a Markov chain:
without somewhere to put the energy, only moves with dH = 0 could ever be accepted.
The demon is the smallest thing that fixes this, and it is not a fudge: the standard
canonical approximation, if the extended system equilibrates and the graph acts as a
sufficiently large bath, is that demon levels have weights exp(-demon/g). This is an
assumption to check, not a consequence of conservation (`demon_temperature` below).

ASSUMPTION Q12 (ours: the leak). After each sweep the demon keeps a fraction
(1 - leak) of its energy and the rest is counted as gone. leak = 0 is sealed;
leak = 1 removes everything given off, as fast as it appears. In between, what
happens is a race between how fast energy is released and how fast it escapes, which
is the author's semi-permeable wall. The leak is a choice of protocol, not of model,
and must be declared before a run like any other knob (VISION S1).
"""
import numpy as np
from numba import njit

from graphity.connectivity import connectivity
from graphity.cqg import (CAP, ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _new_edge_ok, _note_square_edges, _replace,
                          _surplus_on, _switch, has_edge, squares_on_edge, surplus, total_squares)


@njit(cache=True)
def run_sealed(adj, side_u, demon, n_sweeps, seed, lam=1.0, cap=CAP, leak=0.0, conn=None):
    """Run at fixed total energy (leak = 0) or with a leak. Modifies adj in place.

    demon : the energy the demon starts with. It is never allowed to go negative.
    leak  : fraction of the demon's energy lost after each sweep (Q12).
    conn  : optional (n_sweeps, 4) array for the connectivity numbers, as in run_chain.
    Returns (S after each sweep, X after each sweep, demon after each sweep,
    energy lost after each sweep, acceptance rate). One sweep = 2N attempted switches.
    """
    np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    track_x = cap > CAP
    edges = np.empty((64, 2), dtype=np.int64)
    s = total_squares(adj)
    x = surplus(adj)
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    out_demon = np.zeros(n_sweeps, dtype=np.float64)
    out_lost = np.zeros(n_sweeps, dtype=np.float64)
    lost = 0.0
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
        for _ in range(2 * n):
            attempted += 1
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
            lost_squares = squares_on_edge(adj, u1, v1)
            _replace(adj, u1, v1, -1)
            _replace(adj, v1, u1, -1)
            lost_squares += squares_on_edge(adj, u2, v2)
            _replace(adj, u2, v2, -1)
            _replace(adj, v2, u2, -1)
            _replace(adj, u1, -1, v2)
            _replace(adj, v2, -1, u1)
            gained = squares_on_edge(adj, u1, v2)
            _replace(adj, u2, -1, v1)
            _replace(adj, v1, -1, u2)
            gained += squares_on_edge(adj, u2, v1)
            d_s = gained - lost_squares
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
                if d_h > demon:                      # the demon cannot pay for it
                    ok = False
                else:
                    demon -= d_h                     # it pays, or is paid
            if ok:
                s += d_s
                x += d_x
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        if leak > 0.0:
            gone = demon * leak
            demon -= gone
            lost += gone
        out_s[sweep] = s
        out_x[sweep] = x
        out_demon[sweep] = demon
        out_lost[sweep] = lost
        if conn is not None:
            pieces, largest, in_babies, cubes = connectivity(adj)
            conn[sweep, 0] = pieces
            conn[sweep, 1] = largest
            conn[sweep, 2] = in_babies
            conn[sweep, 3] = cubes
    return out_s, out_x, out_demon, out_lost, accepted / max(attempted, 1)


@njit(cache=True)
def run_sealed_bath(adj, side_u, demons, n_sweeps, seed, lam=1.0, cap=CAP, conn=None, by_vertex=False):
    """PREREGISTRATION T9: sealed, with a bath of C demons instead of one. Modifies adj in place.

    demons : float array of length C, the energy each demon starts with. Each attempted move
             picks one demon at random to pay for it or be paid by it; a move the chosen demon
             cannot afford is refused. H + sum(demons) is conserved exactly. The demons' mean
             energy is an observable, not a calibrated temperature; C sets the mean energy gain per unit
             of energy released -- which is the knob the experiment scans. With C = 1 no random
             choice is made and the run is bit-for-bit run_sealed with leak = 0 (tested).
    by_vertex : PREREGISTRATION T26 (the local spark; ASSUMPTIONS Q22). When True, `demons` has one
             store per vertex (length N) and a move pays from, or is paid into, the store of u1, the
             first vertex of the proposed switch, instead of a store chosen at random. Energy then
             stays where a move released it, and can only be spent by a move made from that vertex:
             a store is a local heat capacity, not a shared bath. Stores of side-1 vertices are never
             chosen (u1 is always in side 0) and stay as given. With by_vertex False nothing here
             changes, draw for draw (tested against a pinned run).
    seed   : seed >= 0 seeds the stream; seed < 0 carries on the previous call's stream, as
             cqg.run_chain does (ASSUMPTIONS Q14), so a run may be paused between blocks to read the
             graph or to drain the stores (a leak) without re-seeding.
    Returns (S after each sweep, X, mean demon energy, sum of demon energy, acceptance rate).
    """
    if seed >= 0:
        np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    track_x = cap > CAP
    edges = np.empty((64, 2), dtype=np.int64)
    s = total_squares(adj)
    x = surplus(adj)
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    c = demons.shape[0]
    out_mean = np.zeros(n_sweeps, dtype=np.float64)
    out_sum = np.zeros(n_sweeps, dtype=np.float64)
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
        for _ in range(2 * n):
            attempted += 1
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
            lost_squares = squares_on_edge(adj, u1, v1)
            _replace(adj, u1, v1, -1)
            _replace(adj, v1, u1, -1)
            lost_squares += squares_on_edge(adj, u2, v2)
            _replace(adj, u2, v2, -1)
            _replace(adj, v2, u2, -1)
            _replace(adj, u1, -1, v2)
            _replace(adj, v2, -1, u1)
            gained = squares_on_edge(adj, u1, v2)
            _replace(adj, u2, -1, v1)
            _replace(adj, v1, -1, u2)
            gained += squares_on_edge(adj, u2, v1)
            d_s = gained - lost_squares
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
                if by_vertex:
                    k = u1
                else:
                    k = 0 if c == 1 else np.random.randint(0, c)
                if d_h > demons[k]:                  # this demon cannot pay for it
                    ok = False
                else:
                    demons[k] -= d_h                 # it pays, or is paid
            if ok:
                s += d_s
                x += d_x
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        out_s[sweep] = s
        out_x[sweep] = x
        out_sum[sweep] = demons.sum()
        out_mean[sweep] = out_sum[sweep] / c
        if conn is not None:
            pieces, largest, in_babies, cubes = connectivity(adj)
            conn[sweep, 0] = pieces
            conn[sweep, 1] = largest
            conn[sweep, 2] = in_babies
            conn[sweep, 3] = cubes
    return out_s, out_x, out_mean, out_sum, accepted / max(attempted, 1)


def demon_temperature(demon_series, step=None):
    """Infer g from mean demon energy under a canonical discrete-level approximation.

    The demon settles into P(demon) proportional to exp(-demon / g) over the values it
    can take, assumed here to be 0, step, 2*step, ... . At lam=1.25 the coefficients
    in dH=-16*dS+5*dX are integral; valid costs 12 and 25 rule out a common step
    coarser than 1. A quarter-integer lambda does NOT imply a step of 4.
    For a geometric distribution the mean is step/(exp(step/g)-1), which this
    inverts. Supply a separately justified step: the legacy fallback infers the
    smallest observed gap, which need not be the actual accessible spacing.
    Returns NaN for nonpositive mean/step (including constant data with step=None).
    Conservation or a stationary mean alone does not establish this distribution.
    """
    mean = float(np.mean(demon_series))
    if step is None:
        values = np.unique(np.asarray(demon_series))
        step = float(np.min(np.diff(values))) if values.size > 1 else 0.0
    if mean <= 0 or step <= 0:
        return float("nan")
    return step / np.log1p(step / mean)
