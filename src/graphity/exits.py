"""Counting every exit from the curled torus, and every fall-back into it (PREREGISTRATION T22).

WHY. Eq. (2) of the curled-torus paper predicts the time to the *first* exit from the perfect 4 x L torus.
Near lambda = 1 the measured waits run longer than that, and one reading is that some exits fall back into
the torus instead of growing into a front. Checking it needs every exit seen, including one that heals within
a sweep, so this counts move by move rather than sweep by sweep.

THE MOVE. The kernel's own (cqg.run_chain): two points of side 0 each swap one partner, Metropolis at coupling
g, the hard-core rule enforced by `is_valid`. Energies are recomputed in full at every accepted candidate,
which is simple and exact (the arrangements here have N <= 96).

WHAT IS COUNTED. The torus state is S = 5N/4 and X = N exactly (moves that keep both, which exist, leave it a
torus). An **exit** is an accepted move out of that state; a **fall-back** an accepted move back into it. The run
stops when the conversion fraction (5N/4 - S) / (N/4) first reaches `stop_fraction` (the change has gone
through), or after `max_sweeps`. Returns (exits, fallbacks, attempts to the first exit, attempts to the end,
went_through).

TWO READERS OF THE SAME CHAIN (PREREGISTRATION T59). `first_exit_clocks` stops at the first exit that a look
every few sweeps would see, and returns the first exit on three clocks: by the move, by a look every sweep, by a
look every `look` sweeps. `offers_until_exit` stops at the first exit and tallies what the chain was offered while
it waited. Both make `run_until_through`'s draws in its order, so the first exit of a seed is the same attempt in
all three (tested), and neither draws a random number to look.
"""
import numpy as np
from numba import njit

from graphity.cqg import _switch, has_edge, is_valid, surplus, total_squares


@njit(cache=True)
def _energy(adj, lam):
    return 16.0 * (adj.shape[0] - total_squares(adj)) + 4.0 * lam * surplus(adj)


@njit(cache=True)
def run_until_through(adj, side_u, g, lam, cap, max_sweeps, seed, stop_fraction):
    np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    s_tube = 5 * n // 4
    x_tube = n
    s_stop = s_tube - stop_fraction * (n / 4.0)
    s = total_squares(adj)
    x = surplus(adj)
    h = 16.0 * (n - s) + 4.0 * lam * x
    in_tube = s == s_tube and x == x_tube
    exits = 0
    fallbacks = 0
    first_exit = -1
    attempt = 0
    trial = adj.copy()
    for _ in range(max_sweeps * 2 * n):
        attempt += 1
        u1 = side_u[np.random.randint(0, nu)]
        u2 = side_u[np.random.randint(0, nu)]
        v1 = adj[u1, np.random.randint(0, 4)]
        v2 = adj[u2, np.random.randint(0, 4)]
        if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
            continue
        trial[:] = adj
        _switch(trial, u1, v1, u2, v2)
        if not is_valid(trial, cap):
            continue
        s_new = total_squares(trial)
        x_new = surplus(trial)
        h_new = 16.0 * (n - s_new) + 4.0 * lam * x_new
        d_h = h_new - h
        if d_h > 0 and np.random.random() >= np.exp(-d_h / g):
            continue
        adj[:] = trial
        s, x, h = s_new, x_new, h_new
        now_tube = s == s_tube and x == x_tube
        if in_tube and not now_tube:
            exits += 1
            if first_exit < 0:
                first_exit = attempt
        elif now_tube and not in_tube:
            fallbacks += 1
        in_tube = now_tube
        if s <= s_stop:
            return exits, fallbacks, first_exit, attempt, True
    return exits, fallbacks, first_exit, attempt, False


@njit(cache=True)
def first_exit_clocks(adj, side_u, g, lam, cap, max_sweeps, seed, look):
    """The first exit from the curled torus on three clocks at once (PREREGISTRATION T59).

    The chain is `run_until_through`'s, draw for draw, so its first exit is the same attempt for the same seed.
    Between sweeps the arrangement is looked at, which draws no random number: is it the torus (S = 5N/4 and
    X = N)? The run stops at the first look falling on a multiple of `look` sweeps that finds it is not, or after
    `max_sweeps`. Returns

        first_exit   attempts to the first accepted move out of the torus (-1 if none)
        seen_1       the first sweep after which the arrangement was not the torus (-1 if none)
        seen_k       the first multiple of `look` sweeps after which it was not (-1 if none)
        exits_1      exits made up to and including the one a look every sweep saw
        exits_k      exits made up to and including the one a look every `look` sweeps saw
        fallbacks    returns into the torus before the run stopped

    An exit that came back before a look is one that look cannot see: exits_1 - 1 and exits_k - 1 count them.
    """
    np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    s_tube = 5 * n // 4
    x_tube = n
    s = total_squares(adj)
    x = surplus(adj)
    h = 16.0 * (n - s) + 4.0 * lam * x
    in_tube = s == s_tube and x == x_tube
    exits = 0
    fallbacks = 0
    first_exit = -1
    seen_1 = -1
    exits_1 = 0
    attempt = 0
    trial = adj.copy()
    for sweep in range(1, max_sweeps + 1):
        for _ in range(2 * n):
            attempt += 1
            u1 = side_u[np.random.randint(0, nu)]
            u2 = side_u[np.random.randint(0, nu)]
            v1 = adj[u1, np.random.randint(0, 4)]
            v2 = adj[u2, np.random.randint(0, 4)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            trial[:] = adj
            _switch(trial, u1, v1, u2, v2)
            if not is_valid(trial, cap):
                continue
            s_new = total_squares(trial)
            x_new = surplus(trial)
            h_new = 16.0 * (n - s_new) + 4.0 * lam * x_new
            d_h = h_new - h
            if d_h > 0 and np.random.random() >= np.exp(-d_h / g):
                continue
            adj[:] = trial
            s, x, h = s_new, x_new, h_new
            now_tube = s == s_tube and x == x_tube
            if in_tube and not now_tube:
                exits += 1
                if first_exit < 0:
                    first_exit = attempt
            elif now_tube and not in_tube:
                fallbacks += 1
            in_tube = now_tube
        if not in_tube:
            if seen_1 < 0:
                seen_1 = sweep
                exits_1 = exits
            if sweep % look == 0:
                return first_exit, seen_1, sweep, exits_1, exits, fallbacks
    return first_exit, seen_1, -1, exits_1, exits, fallbacks


@njit(cache=True)
def offers_until_exit(adj, side_u, g, lam, cap, max_sweeps, seed, stretch):
    """What the chain was offered while it waited in the curled torus (PREREGISTRATION T59, stage A).

    The same chain as `run_until_through`, draw for draw, stopped at its first exit. The wait is cut into
    stretches of `stretch` sweeps, and for each stretch the valid proposals are tallied by what they would do:
    kind A (lose 2 squares and 4 surplus squares, the cheapest way out), kind B (lose 4 and 10), any other change
    of (S, X), and the proposals that change neither. Beside them, the smallest acceptance draw made against an
    A proposal and against a B proposal: an exit is taken when that draw falls below exp(-cost / g). Returns
    (attempts to the first exit or -1, a (stretches, 5) integer array of [A, B, other, neutral, hard-core refusals],
    a (stretches, 2) array of the smallest draws against A and against B).
    """
    np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    s_tube = 5 * n // 4
    x_tube = n
    s = total_squares(adj)
    x = surplus(adj)
    h = 16.0 * (n - s) + 4.0 * lam * x
    per_stretch = stretch * 2 * n
    n_stretch = (max_sweeps + stretch - 1) // stretch
    offered = np.zeros((n_stretch, 5), dtype=np.int64)
    smallest = np.ones((n_stretch, 2))
    trial = adj.copy()
    for attempt in range(1, max_sweeps * 2 * n + 1):
        c = (attempt - 1) // per_stretch
        u1 = side_u[np.random.randint(0, nu)]
        u2 = side_u[np.random.randint(0, nu)]
        v1 = adj[u1, np.random.randint(0, 4)]
        v2 = adj[u2, np.random.randint(0, 4)]
        if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
            continue
        trial[:] = adj
        _switch(trial, u1, v1, u2, v2)
        if not is_valid(trial, cap):
            offered[c, 4] += 1
            continue
        s_new = total_squares(trial)
        x_new = surplus(trial)
        d_s = s_new - s
        d_x = x_new - x
        kind = 2
        if d_s == 0 and d_x == 0:
            kind = 3
        elif d_s == -2 and d_x == -4:
            kind = 0
        elif d_s == -4 and d_x == -10:
            kind = 1
        offered[c, kind] += 1
        h_new = 16.0 * (n - s_new) + 4.0 * lam * x_new
        d_h = h_new - h
        if d_h > 0:
            draw = np.random.random()
            if kind < 2 and draw < smallest[c, kind]:
                smallest[c, kind] = draw
            if draw >= np.exp(-d_h / g):
                continue
        adj[:] = trial
        s, x, h = s_new, x_new, h_new
        if not (s == s_tube and x == x_tube):
            return attempt, offered, smallest
    return -1, offered, smallest
