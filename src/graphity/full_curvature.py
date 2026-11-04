"""The full Ollivier energy on graphs that need not be two-sided: triangles and pentagons included.

WHY. The kernel in cqg.py works on bipartite graphs, where triangles and pentagons
cannot occur, as the published simulations do [T25; KTB19 Sec. 4]. The published
energy itself has no such restriction. For a 4-regular graph in which any two short
cycles (length 3, 4, 5) share at most one edge ("quasi-convex", the hard-core rule in
its general form [T25 Def. 2, Fig. 1]), the curvature of an edge is [T25 Eq. (8)]

    kappa = T/4 - [1 - (2 + T + S)/4]_+ - [1 - (2 + T + S + P)/4]_+ ,

with T, S, P the triangles, squares and pentagons through the edge, and the energy
is H = -4 * (sum of kappa over both directions of every edge) [T25 Eq. (21)].
On a bipartite graph T = P = 0 and this is cqg.hamiltonian at lam = 1 (ASSUMPTION
Q1); a test checks that the two agree.

WHAT FOR. The owner asked whether a "brace" (a diagonal support across a block) could
stabilise an arrangement. A brace makes triangles, so it is already priced by this
formula. This module is a slow, plain reference for exact questions at small sizes:
what is the energy of a braced arrangement, and is it a DIP (every valid single move
raises H)? It is not a simulation kernel. networkx graphs, brute force throughout.

ASSUMPTION Q10 (ours). Moves: the general edge switch, (a,b),(c,d) -> (a,c),(b,d) or
(a,d),(b,c), which keeps every degree; the bipartite switch of Q4 is the special case
that keeps the two sides. A move is valid if the new graph is simple and quasi-convex.
"""
from itertools import combinations

import networkx as nx

DEGREE = 4


def short_cycles(g):
    """Every simple cycle of length 3, 4 or 5, each once, as a frozenset of edges (each edge a frozenset)."""
    found = set()

    def walk(path):
        first, last = path[0], path[-1]
        for nxt in g[last]:
            if nxt == first and len(path) >= 3:
                found.add(frozenset(frozenset(pair) for pair in zip(path, path[1:] + [first])))
            elif nxt not in path and nxt > first and len(path) < 5:      # a cycle is listed from its smallest vertex
                walk(path + [nxt])

    for v in g:
        walk([v])
    return found


def is_valid(g):
    """Simple, 4-regular, and no two short cycles sharing more than one edge."""
    if any(d != DEGREE for _, d in g.degree()) or nx.number_of_selfloops(g):
        return False
    return all(len(a & b) <= 1 for a, b in combinations(short_cycles(g), 2))


def energy(g):
    """H of the docstring. 16 per vertex on a graph with no short cycles, 0 on the flat square sheet."""
    through = {frozenset(e): [0, 0, 0] for e in g.edges}
    for cycle in short_cycles(g):
        for e in cycle:
            through[e][len(cycle) - 3] += 1
    total = 0.0
    for t, s, p in through.values():
        kappa = t / 4 - max(0.0, 1 - (2 + t + s) / 4) - max(0.0, 1 - (2 + t + s + p) / 4)
        total += -4 * 2 * kappa                                           # both directions of the edge
    return total


def moves(g):
    """Every valid graph one general edge switch away (Q10)."""
    out = []
    for (a, b), (c, d) in combinations(list(g.edges), 2):
        if len({a, b, c, d}) < 4:
            continue
        for new in (((a, c), (b, d)), ((a, d), (b, c))):
            if any(g.has_edge(*e) for e in new):
                continue
            h = g.copy()
            h.remove_edges_from([(a, b), (c, d)])
            h.add_edges_from(new)
            if is_valid(h):
                out.append(h)
    return out


def way_out(g):
    """(smallest change of H over all valid single moves, number of valid moves). Positive means g is a dip."""
    here = energy(g)
    around = moves(g)
    return min(energy(h) - here for h in around), len(around)


def square_sheet(lx, ly):
    """The flat lx x ly torus of cqg.torus as a networkx graph."""
    g = nx.Graph()
    for x in range(lx):
        for y in range(ly):
            g.add_edge((x, y), ((x + 1) % lx, y))
            g.add_edge((x, y), (x, (y + 1) % ly))
    return nx.convert_node_labels_to_integers(g)


def kagome_sheet(cells):
    """The braced sheet: corner-sharing triangles around hexagons, on a cells x cells torus (3 vertices a cell).

    Built as the graph whose vertices are the edges of a honeycomb, two of them joined when the honeycomb
    edges meet at a corner: three edges meet at each corner, which makes the triangles. Every vertex has
    degree 4, every edge lies in exactly one triangle, and there are no squares or pentagons.
    """
    honeycomb = nx.Graph()
    for x in range(cells):
        for y in range(cells):
            a = ("a", x, y)
            for b in (("b", x, y), ("b", (x - 1) % cells, y), ("b", x, (y - 1) % cells)):
                honeycomb.add_edge(a, b)
    return nx.convert_node_labels_to_integers(nx.line_graph(honeycomb))
