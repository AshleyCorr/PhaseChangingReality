"""Graph construction: initial states, the reference ground state, and menus.

A "menu" is the set of edges that are ALLOWED to be switched on. The original
graphity models use the complete graph as the menu: any vertex may link to
any other [KMS06, Sec. II; K08, Sec. II]. Restricting the menu is the new
ingredient of this project (ASSUMPTION B1 in ASSUMPTIONS.md).
"""
import itertools

import networkx as nx
import numpy as np


# ----------------------------------------------------------------------------
# conversions
# ----------------------------------------------------------------------------
def to_adj(graph: nx.Graph, degree: int) -> np.ndarray:
    """(N, degree) neighbour array for a `degree`-regular graph on 0..N-1."""
    n = graph.number_of_nodes()
    adj = np.full((n, degree), -1, dtype=np.int64)
    for u in range(n):
        nbrs = sorted(graph[u])
        if len(nbrs) != degree:
            raise ValueError(f"vertex {u} has degree {len(nbrs)}, expected {degree}")
        adj[u, :] = nbrs
    return adj


def to_networkx(adj: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(adj.shape[0]))
    for u in range(adj.shape[0]):
        for v in adj[u]:
            if v >= 0:
                g.add_edge(u, int(v))
    return g


def menu_to_array(menu: nx.Graph) -> np.ndarray:
    """(N, Kmax) sorted neighbour array of the menu graph, padded with -1."""
    n = menu.number_of_nodes()
    kmax = max(d for _, d in menu.degree())
    arr = np.full((n, kmax), -1, dtype=np.int64)
    for u in range(n):
        nbrs = sorted(menu[u])
        arr[u, : len(nbrs)] = nbrs
    return arr


# ----------------------------------------------------------------------------
# initial states
# ----------------------------------------------------------------------------
def random_regular_connected(n: int, degree: int, seed: int) -> nx.Graph:
    """Connected random regular graph.

    [K08, Sec. IV A 1] starts simulations from random regular graphs, citing
    Steger & Wormald (1999) for generation; networkx implements that
    algorithm. The Monte Carlo move preserves connectedness, so the start
    must be connected. Random cubic graphs are connected with probability
    tending to 1 [W99], so the retry loop is almost never used.
    """
    rng = np.random.default_rng(seed)
    while True:
        g = nx.random_regular_graph(degree, n, seed=int(rng.integers(2**31 - 1)))
        if nx.is_connected(g):
            return g


def konopka_ground_state(n_units: int) -> nx.Graph:
    """Ring of `n_units` six-vertex units; each unit is K_{3,3} minus one edge.

    ASSUMPTION A4. [K08, Sec. IV A 1, Fig. 2(a)] describes the r = -2.5
    ground state at N = 36 as "six groups of six vertices connected in a
    chain" that maximises 4-cycles. The internal wiring of a group is shown
    only in a figure we could not read, so K_{3,3} minus an edge is our
    inference. It is supported by the fact that it reproduces the published
    epsilon_0 = -12.2 (we get -12.207).
    """
    g = nx.Graph()
    for u in range(n_units):
        base = 6 * u
        for i, j in itertools.product(range(3), range(3)):
            if not (i == 0 and j == 0):
                g.add_edge(base + i, base + 3 + j)
    for u in range(n_units):
        g.add_edge(6 * u + 3, 6 * ((u + 1) % n_units))
    return g


# ----------------------------------------------------------------------------
# menus (ASSUMPTION B1-B3)
# ----------------------------------------------------------------------------
def random_menu_with_planted_cubic(n: int, k: int, seed: int):
    """Random menu of degree k that contains a connected cubic start graph.

    Returns (menu, start). The menu is start + a random (k-3)-regular graph
    on the same vertices, made edge-disjoint from `start` by double-edge
    swaps. ASSUMPTION B2: this is not exactly uniform over k-regular graphs;
    planting is needed so that a valid 3-regular start state exists.
    """
    if k < 4:
        raise ValueError("menu degree must exceed the target valence 3")
    if (n * (k - 3)) % 2:
        raise ValueError("n * (k - 3) must be even")
    rng = np.random.default_rng(seed)
    start = random_regular_connected(n, 3, int(rng.integers(2**31 - 1)))
    extra = nx.random_regular_graph(k - 3, n, seed=int(rng.integers(2**31 - 1)))
    clashes = [e for e in extra.edges() if start.has_edge(*e)]
    guard = 0
    while clashes:
        guard += 1
        if guard > 100000:
            raise RuntimeError("could not make menu edge-disjoint")
        u, v = clashes.pop()
        if not extra.has_edge(u, v):
            continue
        edges = list(extra.edges())
        x, y = edges[int(rng.integers(len(edges)))]
        if len({u, v, x, y}) < 4:
            clashes.append((u, v))
            continue
        ok = all(
            not extra.has_edge(a, b) and not start.has_edge(a, b)
            for a, b in ((u, x), (v, y))
        )
        if not ok:
            clashes.append((u, v))
            continue
        extra.remove_edge(u, v)
        extra.remove_edge(x, y)
        extra.add_edge(u, x)
        extra.add_edge(v, y)
    menu = nx.compose(start, extra)
    return menu, start


def torus_menu_with_cubic_start(sides):
    """d-dimensional periodic lattice menu (degree 2d) and a cubic start graph.

    Start graph (ASSUMPTION B3, our construction): every site keeps both
    edges along axis 0. Its third edge pairs it with a neighbour along axis
    j = 1 + (x0 // 2) % (d-1), with the pairing offset alternating with x0
    so that the rings along axis 0 are tied into one connected graph. For
    d = 2 this is the brick-wall (honeycomb) lattice.

    Requires sides[0] divisible by 2(d-1) and all other sides even.
    """
    sides = tuple(int(s) for s in sides)
    d = len(sides)
    if d < 2:
        raise ValueError("need at least 2 dimensions")
    if sides[0] % (2 * (d - 1)) or any(s % 2 for s in sides[1:]) or min(sides) < 4:
        raise ValueError("sides[0] must be a multiple of 2(d-1); all sides even and >= 4")

    def index(x):
        i = 0
        for xi, s in zip(x, sides):
            i = i * s + (xi % s)
        return i

    menu, start = nx.Graph(), nx.Graph()
    for x in itertools.product(*[range(s) for s in sides]):
        here = index(x)
        for axis in range(d):
            y = list(x)
            y[axis] += 1
            menu.add_edge(here, index(y))
        y = list(x)
        y[0] += 1
        start.add_edge(here, index(y))
        axis = 1 + (x[0] // 2) % (d - 1)
        offset = x[0] % 2
        if (x[axis] + offset) % 2 == 0:
            y = list(x)
            y[axis] += 1
            start.add_edge(here, index(y))
    return menu, start
