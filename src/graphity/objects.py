"""Two objects of nine squares in a four-link graph, counted exactly (PREREGISTRATION T78).

The curled ring: a square a-b-c-d each of whose four links carries three squares (the most the hard-core rule allows;
cqg.py Q2). It is the column of paper 2 in kind: on the 4 x L curled torus every ring round the short side is one. It
touches nine squares (its own and two more through each link) on twelve points and carries four surplus squares.

The flat patch: a 4 x 4 grid of sixteen distinct points with all 24 grid links present, so nine squares in a 3 x 3
block. Extra links among the sixteen points are allowed (the count is of grid monomorphisms, not induced subgraphs).
A grid is found cell by cell: the first row x[0][0..3] and the first column x[0..3][0] are walks along links, and
every other cell x[i][j] is the unique common neighbour of x[i-1][j] and x[i][j-1] other than x[i-1][j-1] (by the
hard-core rule two points have at most two common neighbours, so "the other one" is unique when it exists). Ordered
(corner, two directions) tuples are counted and divided by 8, the symmetries of a square grid acting on a 4 x 4 grid
with sixteen distinct points.

Both counts are checked in tests/test_objects.py against networkx (rings: 4-cycles whose links all carry three
squares; patches: subgraph monomorphisms of grid_2d_graph(4, 4)) and against the flat and curled tori (196 patches
and no ring on the flat 14 x 14 torus; 16 rings on the 16 x 4 curled torus).
"""
import numpy as np
from numba import njit

from graphity.squares import squares_on_edge

K = 4                                   # points per grid side: 4 x 4 points = 3 x 3 squares


@njit(cache=True)
def _common_other(adj, a, b, exclude):
    """The common neighbours of a and b other than `exclude`, as (count, first found)."""
    count = 0
    found = -1
    for i in range(4):
        x = adj[a, i]
        if x == exclude:
            continue
        for j in range(4):
            if adj[b, j] == x:
                count += 1
                if found < 0:
                    found = x
                break
    return count, found


@njit(cache=True)
def count_rings(adj):
    """Squares whose four links each carry three squares; each square counted once."""
    n = adj.shape[0]
    tuples = 0
    for a in range(n):
        for i in range(4):
            b = adj[a, i]
            if squares_on_edge(adj, a, b) < 3:
                continue
            for k in range(4):
                d = adj[a, k]
                if d == b or squares_on_edge(adj, a, d) < 3:
                    continue
                for j in range(4):              # c: a common neighbour of b and d other than a
                    c = adj[b, j]
                    if c == a:
                        continue
                    linked = False
                    for m in range(4):
                        if adj[d, m] == c:
                            linked = True
                            break
                    if not linked:
                        continue
                    if squares_on_edge(adj, b, c) == 3 and squares_on_edge(adj, c, d) == 3:
                        tuples += 1
    # each square is seen from each of its four corners a, with (b, d) in both orders
    return tuples // 8


@njit(cache=True)
def count_ring2(adj):
    """Squares whose four links each carry exactly two squares (the uncurled twin of the ring; T79), each once."""
    n = adj.shape[0]
    tuples = 0
    for a in range(n):
        for i in range(4):
            b = adj[a, i]
            if squares_on_edge(adj, a, b) != 2:
                continue
            for k in range(4):
                d = adj[a, k]
                if d == b or squares_on_edge(adj, a, d) != 2:
                    continue
                for j in range(4):
                    c = adj[b, j]
                    if c == a:
                        continue
                    linked = False
                    for m in range(4):
                        if adj[d, m] == c:
                            linked = True
                            break
                    if not linked:
                        continue
                    if squares_on_edge(adj, b, c) == 2 and squares_on_edge(adj, c, d) == 2:
                        tuples += 1
    return tuples // 8


@njit(cache=True)
def count_grid(adj, a, b, flat_only):
    """a x b grids of points (a rows, b columns, a, b >= 2) with all grid links present and, if flat_only, none of
    those links carrying three squares; sixteen or fewer distinct points. Ordered tuples / 8 (a == b) or / 4 (T79)."""
    n = adj.shape[0]
    grid = np.empty((a, b), dtype=np.int64)
    used = np.empty(a * b, dtype=np.int64)
    # the free cells in fill order: first row (0, 1..b-1), first column (1..a-1, 0); then the interior row by row
    nfree = (b - 1) + (a - 1)
    choice = np.zeros(nfree, dtype=np.int64)
    tuples = 0
    for x00 in range(n):
        grid[0, 0] = x00
        # depth-first over the free cells with an explicit stack of neighbour indices
        depth = 0
        for t in range(nfree):
            choice[t] = -1
        while depth >= 0:
            choice[depth] += 1
            if choice[depth] >= 4:
                choice[depth] = -1
                depth -= 1
                continue
            if depth < b - 1:
                r, c = 0, depth + 1
                prev = grid[0, c - 1]
            else:
                r, c = depth - (b - 1) + 1, 0
                prev = grid[r - 1, 0]
            v = adj[prev, choice[depth]]
            # distinct from every point placed so far (first row up to c-1, first column up to r-1)
            dup = False
            for cc in range(b):
                if (r == 0 and cc >= c) or (r > 0 and cc > b - 1):
                    break
                if grid[0, cc] == v:
                    dup = True
                    break
            if not dup and r > 0:
                for rr in range(1, r):
                    if grid[rr, 0] == v:
                        dup = True
                        break
            if dup:
                continue
            grid[r, c] = v
            if depth == nfree - 1:
                # fill the interior
                m = 0
                for rr in range(a):
                    used[m] = grid[rr, 0]
                    m += 1
                for cc in range(1, b):
                    used[m] = grid[0, cc]
                    m += 1
                ok = True
                for rr in range(1, a):
                    for cc in range(1, b):
                        cnt, w = _common_other(adj, grid[rr - 1, cc], grid[rr, cc - 1], grid[rr - 1, cc - 1])
                        if cnt != 1 or _in(used, m, w):
                            ok = False
                            break
                        grid[rr, cc] = w
                        used[m] = w
                        m += 1
                    if not ok:
                        break
                if ok and flat_only:
                    for rr in range(a):
                        for cc in range(b):
                            if cc + 1 < b and squares_on_edge(adj, grid[rr, cc], grid[rr, cc + 1]) > 2:
                                ok = False
                            if rr + 1 < a and squares_on_edge(adj, grid[rr, cc], grid[rr + 1, cc]) > 2:
                                ok = False
                if ok:
                    tuples += 1
            else:
                depth += 1
    return tuples // (8 if a == b else 4)


@njit(cache=True)
def _in(arr, m, v):
    for i in range(m):
        if arr[i] == v:
            return True
    return False


@njit(cache=True)
def count_patches(adj):
    """4 x 4 grids with all grid links present and sixteen distinct points: ordered tuples / 8."""
    n = adj.shape[0]
    grid = np.empty((K, K), dtype=np.int64)
    used = np.empty(K * K, dtype=np.int64)
    tuples = 0
    for x00 in range(n):
        grid[0, 0] = x00
        for i1 in range(4):
            x01 = adj[x00, i1]
            grid[0, 1] = x01
            for i2 in range(4):
                x02 = adj[x01, i2]
                if x02 == x00:
                    continue
                grid[0, 2] = x02
                for i3 in range(4):
                    x03 = adj[x02, i3]
                    if x03 == x00 or x03 == x01:
                        continue
                    grid[0, 3] = x03
                    for j1 in range(4):
                        x10 = adj[x00, j1]
                        if x10 == x01 or x10 == x02 or x10 == x03:
                            continue
                        grid[1, 0] = x10
                        for j2 in range(4):
                            x20 = adj[x10, j2]
                            if x20 == x00 or x20 == x01 or x20 == x02 or x20 == x03:
                                continue
                            grid[2, 0] = x20
                            for j3 in range(4):
                                x30 = adj[x20, j3]
                                if x30 == x00 or x30 == x01 or x30 == x02 or x30 == x03 or x30 == x10:
                                    continue
                                grid[3, 0] = x30
                                # interior, row by row: unique common neighbour of up and left, other than diagonal
                                m = 0
                                for r in range(K):
                                    used[m] = grid[r, 0]
                                    m += 1
                                for c in range(1, K):
                                    used[m] = grid[0, c]
                                    m += 1
                                ok = True
                                for r in range(1, K):
                                    for c in range(1, K):
                                        cnt, v = _common_other(adj, grid[r - 1, c], grid[r, c - 1], grid[r - 1, c - 1])
                                        if cnt != 1 or _in(used, m, v):
                                            ok = False
                                            break
                                        grid[r, c] = v
                                        used[m] = v
                                        m += 1
                                    if not ok:
                                        break
                                if ok:
                                    tuples += 1
    return tuples // 8
