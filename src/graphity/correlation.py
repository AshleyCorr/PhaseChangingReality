"""The correlation length of [KTB19] Sec. 4, Eqs. (4.14)-(4.15) and Fig. 9 (task T16).

WHAT IS MEASURED, in the paper's own terms. Each edge uv carries the field
    phi_sq(uv) = S_uv / (d - 2)                                   [KTB19] Eq. (3.4)
(S_uv the squares on the edge, d = 4, so phi_sq = S_uv / 2). Its mean over the edges of one graph
is phi(omega) = S / N, our order parameter. Each vertex carries the average over its d edges,
    f(u) = sum_{v ~ u} phi_sq(uv) / d.
The correlation at graph distance r is
    C(r) = < (f(u) - phi)(f(v) - phi) >_{pairs at distance r} / < (phi_sq(e) - phi)^2 >_{edges},
with both averages taken over the one graph ("expectation with respect to the graph"). The
denominator is the variance of the EDGE field, as the paper writes it, not of the vertex field, so
C need not be 1 at r = 0. The paper then defines
    xi = - < r / log C(r) >
and divides it by the graph's diameter.

CHOICES THE PAPER LEAVES OPEN (ours; ASSUMPTIONS Q19, fixed before any run).
  1. The outer average in xi is over the distances r = 1 .. diameter at which 0 < C(r) < 1, each
     distance counted once. log C is undefined for C <= 0, and C >= 1 gives a negative or infinite
     length; those distances are skipped and counted.
  2. A graph whose edge field has zero variance (a perfect lattice, every S_e = 2) has no
     fluctuations to correlate; it is skipped and counted.
  3. Distances are shortest paths; pairs in different pieces are ignored.
"""
import numpy as np
from numba import njit

from graphity.squares import squares_on_edge

DEGREE = 4


@njit(cache=True)
def edge_squares(adj):
    """(N, 4) array: squares on the edge from u to its k-th neighbour."""
    n = adj.shape[0]
    out = np.zeros((n, DEGREE), dtype=np.int64)
    for u in range(n):
        for k in range(DEGREE):
            out[u, k] = squares_on_edge(adj, u, adj[u, k])
    return out


@njit(cache=True)
def correlation_by_distance(adj, max_r):
    """One graph: (num, count, edge_var, phi, diameter).

    num[r]   sum over ordered pairs (u, v) at distance r of (f(u) - phi)(f(v) - phi)
    count[r] number of such ordered pairs
    so that C(r) = num[r] / count[r] / edge_var. Distances beyond max_r are not recorded but still
    count towards the diameter.
    """
    n = adj.shape[0]
    se = edge_squares(adj)
    total = 0.0
    for u in range(n):
        for k in range(DEGREE):
            total += se[u, k]
    phi = total / (n * DEGREE) / (DEGREE - 2)    # mean of S_e/(d-2) over edges; each edge is seen twice
    edge_var = 0.0
    for u in range(n):
        for k in range(DEGREE):
            d = se[u, k] / (DEGREE - 2) - phi
            edge_var += d * d
    edge_var /= n * DEGREE                                      # each edge seen twice; mean unchanged
    f = np.zeros(n)
    for u in range(n):
        for k in range(DEGREE):
            f[u] += se[u, k] / (DEGREE - 2)
        f[u] = f[u] / DEGREE - phi
    num = np.zeros(max_r + 1)
    count = np.zeros(max_r + 1, dtype=np.int64)
    dist = np.empty(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    diameter = 0
    for s in range(n):
        dist[:] = -1
        dist[s] = 0
        head, tail = 0, 1
        queue[0] = s
        while head < tail:
            a = queue[head]
            head += 1
            for k in range(DEGREE):
                b = adj[a, k]
                if dist[b] < 0:
                    dist[b] = dist[a] + 1
                    queue[tail] = b
                    tail += 1
        for i in range(1, tail):
            v = queue[i]
            r = dist[v]
            if r > diameter:
                diameter = r
            if r <= max_r:
                num[r] += f[s] * f[v]
                count[r] += 1
    return num, count, edge_var, phi, diameter


def correlation_curve(num, count, edge_var):
    """C(r) for r = 0 .. len(num) - 1; NaN where no pair sits at r or edge_var is 0."""
    c = np.full(len(num), np.nan)
    if edge_var <= 0:
        return c
    ok = count > 0
    c[ok] = num[ok] / count[ok] / edge_var
    return c


def xi_literal(c, diameter):
    """[KTB19] Eq. (4.15) under choice 1: (xi, distances used, distances skipped)."""
    used, skipped, total = 0, 0, 0.0
    for r in range(1, min(diameter, len(c) - 1) + 1):
        if np.isfinite(c[r]) and 0.0 < c[r] < 1.0:
            total += -r / np.log(c[r])
            used += 1
        else:
            skipped += 1
    return (total / used if used else np.nan), used, skipped
