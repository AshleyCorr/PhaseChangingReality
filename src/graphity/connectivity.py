"""Connectivity of a CQG graph: how many pieces, how big, and which are ground-state pieces.

Why it is needed. With the global term only (lam = 0) the published result is that
the graph "decomposes into isolated, weakly interacting hypercubic complexes"
[T25, Cycle condensation]. [KTB19 Sec. 3.3.1] gives the definition: a BABY
UNIVERSE is a connected piece in which every edge carries the largest possible
number of squares, 2D - 1 = 3. Such pieces have the lowest energy the global term
allows, -8 per vertex (ASSUMPTION Q1), and [KTB19 Fig. 5] draws the ones it saw:
16-vertex pieces that look like 4-cubes. Gate A has to see that shattering, and
VISION S4 has to see its absence.

The 4-cube Q4 (16 vertices, 32 edges, 24 squares) is a baby universe but not the
only one: the 14-vertex graph of the seven points and seven blocks of the biplane
(the complement of the Fano plane) is another, with exactly the same energy per
vertex (ASSUMPTION Q8; ours, checked by computer). So baby universes are counted
by the published definition, and the 4-cubes among them separately.

The edge switch of Q4 does not protect connectedness, so pieces split off and
rejoin freely. Everything here reads the (N, 4) neighbour array and changes nothing.
"""
import numpy as np
from numba import njit

from graphity.squares import squares_on_edge

CUBE_SIZE = 16             # vertices of the 4-cube
MAX_SQUARES_ON_EDGE = 3    # 2D - 1: all the hard-core rule allows               (Q2)


@njit(cache=True)
def component_labels(adj):
    """Label every vertex with the number of its connected piece: 0, 1, 2, ...

    Pieces are numbered in the order of their lowest vertex. Plain flood fill.
    """
    n = adj.shape[0]
    label = np.full(n, -1, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)          # a vertex is pushed once, when it gets its label
    pieces = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        label[start] = pieces
        stack[0] = start
        top = 1
        while top > 0:
            top -= 1
            u = stack[top]
            for k in range(4):
                v = adj[u, k]
                if label[v] < 0:
                    label[v] = pieces
                    stack[top] = v
                    top += 1
        pieces += 1
    return label


@njit(cache=True)
def _find(members, count, v):
    for i in range(count):
        if members[i] == v:
            return i
    return -1


@njit(cache=True)
def is_four_cube(adj, root):
    """True if the connected piece containing `root` is the 4-cube Q4.

    Method: try to give every vertex of the piece a 4-bit address. The root gets
    0000 and its four neighbours 0001, 0010, 0100, 1000. Working outwards, a vertex
    takes the bitwise OR of the addresses of its neighbours one step nearer the
    root, which in a cube are its own address with one bit removed. The piece is Q4
    exactly when this ends with 16 vertices, 16 different addresses, and every edge
    joining two addresses that differ in one bit: the addresses then map the 32
    edges of the piece one-to-one onto the 32 edges of Q4. Exact, no tolerance.
    """
    members = np.empty(CUBE_SIZE, dtype=np.int64)
    address = np.zeros(CUBE_SIZE, dtype=np.int64)
    dist = np.zeros(CUBE_SIZE, dtype=np.int64)
    members[0] = root
    count = 1
    head = 0
    while head < count:                          # breadth first, so nearer vertices are finished first
        u = members[head]
        for k in range(4):
            v = adj[u, k]
            j = _find(members, count, v)
            if j < 0:
                if count == CUBE_SIZE:
                    return False                 # a 17th vertex
                members[count] = v
                dist[count] = dist[head] + 1
                address[count] = (1 << k) if head == 0 else address[head]
                count += 1
            elif dist[j] == dist[head] + 1:
                address[j] |= address[head]
        head += 1
    if count != CUBE_SIZE:
        return False
    seen = 0
    for i in range(CUBE_SIZE):
        seen |= 1 << address[i]
    if seen != (1 << CUBE_SIZE) - 1:
        return False                             # two vertices share an address
    for i in range(CUBE_SIZE):
        for k in range(4):
            j = _find(members, count, adj[members[i], k])
            diff = address[i] ^ address[j]
            if diff == 0 or diff & (diff - 1):   # not exactly one bit
                return False
    return True


@njit(cache=True)
def connectivity(adj):
    """(pieces, vertices in the largest piece, vertices in baby universes, number of 4-cubes).

    A piece is a baby universe when every one of its edges carries three squares
    [KTB19 Sec. 3.3.1]. Every 4-cube is one, so 16 * cubes <= vertices in baby universes.
    """
    n = adj.shape[0]
    label = component_labels(adj)
    pieces = label.max() + 1
    size = np.zeros(pieces, dtype=np.int64)
    first = np.full(pieces, -1, dtype=np.int64)
    full = np.ones(pieces, dtype=np.bool_)       # no edge of the piece below three squares, so far
    for v in range(n):
        p = label[v]
        if first[p] < 0:
            first[p] = v
        size[p] += 1
        for k in range(4):
            if full[p] and adj[v, k] > v and squares_on_edge(adj, v, adj[v, k]) < MAX_SQUARES_ON_EDGE:
                full[p] = False
    in_babies = 0
    cubes = 0
    for p in range(pieces):
        if full[p]:
            in_babies += size[p]
            if size[p] == CUBE_SIZE and is_four_cube(adj, first[p]):
                cubes += 1
    return pieces, size.max(), in_babies, cubes
