"""Every single move out of a curled or flat torus, priced exactly, in D = 2 or D = 3. Usage:

    python scripts/exact_exits_d.py

EXACT, exploratory (ASSUMPTIONS Q21 for the D = 3 definitions, O41 for the result). The first exact question of the
six-link model (VISION Update 22): which curled tori are stuck for now above lambda = 1, and how high are their
walls? Tori are C_{L1} x ... x C_{LD}: a side of 4 is a curled direction (its wrap-around 4-cycles put extra
squares on every edge along it), a side of 6 or more is open. Graphs are 2D-regular and bipartite; the move is the
kernel's (two points of side 0 each swap one partner); validity is the kernel's hard-core rule (no two points share
more than two neighbours; [KTB19] Sec. 2.2 and [T22] Sec. II: the same rule at every D).

Energy (Q21; ours, derived from [T25] Eqs. (8), (21), (22) exactly as Q1 at D = 2):
    H = 16 (D(D-1)/2 N - S) + 4 lam X,   X = sum_e (S_e - (2D - 2))_+,
so a move changing S by dS and X by dX costs -16 dS + 4 lam dX. A move offered c times among the (N/2)^2 (2D)^2
proposals, which number N^2 D^2, is offered 2c * 2N / (N^2 D^2) = 4c / (N D^2) times per sweep of 2N attempts
(each unordered switch is two ordered proposals). At D = 2 that is c / N, the count used in paper 1. A sweep is kept
at 2N attempts at D = 3 too; the D = 3 kernel must say so when it is built.

Checked against D = 2 before D = 3 is read (tests/test_exact_exits_d.py): the 16 x 4 torus gives paper 1's move A
(-2, -4; 3N moves) and move B (-4, -10; 2N moves), and a gas of 4-cubes gives O40.
"""
import sys
from collections import Counter

import numpy as np
from numba import njit


def torus(dims):
    """(adj, side) of the torus C_{dims[0]} x ... ; every side even and at least 4."""
    dims = tuple(int(x) for x in dims)
    assert all(x % 2 == 0 and x >= 4 for x in dims)
    d = len(dims)
    coords = np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims)).T
    index = {tuple(c): i for i, c in enumerate(coords)}
    adj = np.empty((len(coords), 2 * d), dtype=np.int64)
    for i, c in enumerate(coords):
        k = 0
        for axis in range(d):
            for step in (1, -1):
                nb = list(c)
                nb[axis] = (nb[axis] + step) % dims[axis]
                adj[i, k] = index[tuple(nb)]
                k += 1
    side = (coords.sum(axis=1) % 2).astype(np.int64)
    return adj, side


def disjoint(parts):
    """Disjoint union of several (adj, side)."""
    adjs, sides, off = [], [], 0
    for adj, side in parts:
        adjs.append(adj + off)
        sides.append(side)
        off += adj.shape[0]
    return np.concatenate(adjs), np.concatenate(sides)


@njit
def _has(adj, x, y):
    for k in range(adj.shape[1]):
        if adj[x, k] == y:
            return True
    return False


@njit
def _squares_on(adj, u, v):
    c = 0
    for i in range(adj.shape[1]):
        a = adj[u, i]
        if a == v:
            continue
        for j in range(adj.shape[1]):
            b = adj[v, j]
            if b == u:
                continue
            if _has(adj, a, b):
                c += 1
    return c


@njit
def s_and_x(adj, sat):
    """(S, X): squares, and surplus squares above sat on each edge."""
    n, deg = adj.shape
    tot = 0
    x = 0
    for u in range(n):
        for k in range(deg):
            v = adj[u, k]
            if u < v:
                se = _squares_on(adj, u, v)
                tot += se
                if se > sat:
                    x += se - sat
    return tot // 4, x


@njit
def _codegree(adj, x, y):
    c = 0
    for i in range(adj.shape[1]):
        for j in range(adj.shape[1]):
            if adj[x, i] == adj[y, j]:
                c += 1
    return c


@njit
def _locally_valid(adj, pts):
    """Simple at pts, and no pair (p, w) with p in pts sharing more than two neighbours."""
    deg = adj.shape[1]
    for p in pts:
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
                if w != p and _codegree(adj, p, w) > 2:
                    return False
    return True


@njit
def _replace(adj, x, old, new):
    for k in range(adj.shape[1]):
        if adj[x, k] == old:
            adj[x, k] = new
            return


@njit
def all_moves(adj, side_u, sat):
    """(dS, dX) of every valid switch the chain can propose, one entry per unordered switch."""
    s0, x0 = s_and_x(adj, sat)
    deg = adj.shape[1]
    out_s = []
    out_x = []
    pts = np.empty(4, dtype=np.int64)
    for i in range(side_u.shape[0]):
        u1 = side_u[i]
        for j in range(i + 1, side_u.shape[0]):
            u2 = side_u[j]
            for a in range(deg):
                v1 = adj[u1, a]
                for b in range(deg):
                    v2 = adj[u2, b]
                    if v1 == v2 or _has(adj, u1, v2) or _has(adj, u2, v1):
                        continue
                    adj[u1, a] = v2
                    adj[u2, b] = v1
                    _replace(adj, v1, u1, u2)
                    _replace(adj, v2, u2, u1)
                    pts[0] = u1; pts[1] = u2; pts[2] = v1; pts[3] = v2
                    if _locally_valid(adj, pts):
                        s1, x1 = s_and_x(adj, sat)
                        out_s.append(s1 - s0)
                        out_x.append(x1 - x0)
                    adj[u1, a] = v1
                    adj[u2, b] = v2
                    _replace(adj, v1, u2, u1)
                    _replace(adj, v2, u1, u2)
    return out_s, out_x


def census(adj, side, dim):
    """Counter of (dS, dX) -> number of unordered switches, and per-sweep offer rate factor."""
    sat = 2 * dim - 2
    ds, dx = all_moves(adj.copy(), np.flatnonzero(side == 0).astype(np.int64), sat)
    return Counter(zip(ds, dx)), 4.0 / (adj.shape[0] * dim * dim)


def energy_per_vertex(adj, dim):
    """H/N as (constant, coefficient of lam)."""
    n = adj.shape[0]
    s, x = s_and_x(adj, 2 * dim - 2)
    return 16.0 * (dim * (dim - 1) / 2 * n - s) / n, 4.0 * x / n


def stuck_window(counts):
    """The lam interval above 1 in which every move that changes (S, X) costs energy (costs -16 dS + 4 lam dX)."""
    lo, hi = 1.0, float("inf")
    for (ds, dx) in counts:
        if ds == 0 and dx == 0:
            continue
        a, b = -16.0 * ds, 4.0 * dx            # cost = a + b lam > 0
        if b > 0:
            lo = max(lo, -a / b)
        elif b < 0:
            hi = min(hi, -a / b)
        elif a <= 0:
            return None
    return (lo, hi) if lo < hi else None


def report(name, adj, side, dim, lam=1.25, show=6):
    c0, c1 = energy_per_vertex(adj, dim)
    counts, rate = census(adj, side, dim)
    print(f"{name}: N = {adj.shape[0]}, H/N = {c0:+.2f} {c1:+.2f} lam = {c0 + c1 * lam:+.3f} at lam = {lam}")
    rows = sorted(((-16 * ds + 4 * lam * dx, ds, dx, n) for (ds, dx), n in counts.items() if (ds, dx) != (0, 0)))
    for cost, ds, dx, n in rows[:show]:
        print(f"   dS {ds:+d} dX {dx:+d}  cost {-16 * ds:+d} {4 * dx:+d} lam = {cost:+7.2f}   {n:6d} moves, offered {n * rate:7.3f} per sweep")
    w = stuck_window(counts)
    print("   stuck for now (every move out costs) for lam in", "none above 1" if w is None else "(%.3f, %.3f)" % w)
    return counts


def main():
    print("== D = 2, checks against paper 1 and O40")
    report("curled torus 16 x 4", *torus((16, 4)), 2)
    report("gas of two 4-cubes", *disjoint([torus((4, 4)), torus((4, 4))]), 2)
    print("\n== D = 3")
    report("flat 6 x 6 x 6", *torus((6, 6, 6)), 3)
    report("one direction curled, 4 x 6 x 6", *torus((4, 6, 6)), 3)
    report("two directions curled, 4 x 4 x 6", *torus((4, 4, 6)), 3)
    report("all three curled, the 6-cube 4 x 4 x 4", *torus((4, 4, 4)), 3)
    report("gas of two 6-cubes", *disjoint([torus((4, 4, 4)), torus((4, 4, 4))]), 3)


if __name__ == "__main__":
    main(*sys.argv[1:])
