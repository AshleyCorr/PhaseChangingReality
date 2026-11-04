"""The two smallest readers of a CQG graph, shared by cqg.py and connectivity.py.

The graph is an (N, 4) array of neighbours; -1 marks a slot that is empty in the
middle of a move (see cqg.run_chain).
"""
from numba import njit


@njit(cache=True)
def has_edge(adj, a, b):
    for k in range(4):
        if adj[a, k] == b:
            return True
    return False


@njit(cache=True)
def squares_on_edge(adj, u, v):
    """Number of 4-cycles u - v - a - b - u containing the edge (u, v)."""
    count = 0
    for i in range(4):
        a = adj[v, i]
        if a < 0 or a == u:
            continue
        for j in range(4):
            b = adj[u, j]
            if b < 0 or b == v:
                continue
            if has_edge(adj, a, b):
                count += 1
    return count
