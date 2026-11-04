"""A Metropolis chain on general 4-regular graphs, where triangles and pentagons are allowed.

WHY. `cqg.py`, the fast kernel, works on two-sided (bipartite) graphs, as the published
simulations do. `full_curvature.py` holds the same published energy written for graphs that
need not be two-sided (ASSUMPTION Q10) and is exact but static: it can say what an arrangement
costs and whether it is a dip, and nothing about what a run does. This module adds the missing
half, a plain Metropolis chain over those graphs, so that questions of the form "left to itself,
what does this arrangement turn into?" can be asked where braces are allowed.

The question it was written for (the author's, 2026-09-22). At the published prices a closed
30-point piece with a triangle and a pentagon on every edge (the icosidodecahedron) has exactly
the energy of the flat sheet, and both are dips (Q10). So refolding space into a closed piece
costs nothing there. Does it then happen, and what has to be supplied for it to happen?

SPEED. Everything is networkx and brute force, as in `full_curvature`. One attempted move costs
about 1.3 ms at 30 vertices and 3.2 ms at 60, because the trial graph's short cycles are listed
from scratch; validity and energy are read off that one listing rather than two. That is slow
enough to fix the sizes this module can reach and fast enough for them.

THE MOVE, and why it satisfies detailed balance. Pick an unordered pair of distinct edges
uniformly out of the 2N the graph always has, then one of its two rewirings uniformly:
(a,b),(c,d) -> (a,c),(b,d) or (a,d),(b,c). This is the general edge switch of Q10, which keeps
every degree at 4. If the four endpoints are not distinct, or a proposed edge is already there,
or the result breaks the hard-core rule, the move is refused and the chain stays where it is.
The proposal is symmetric: the new graph has the same number of edges, so the pair is drawn with
the same probability there, and one of that pair's two rewirings is exactly the move back.
Refusing the invalid ones restricts the walk to the valid states without breaking that symmetry,
so Metropolis acceptance min(1, exp(-dH/g)) has the Boltzmann distribution at coupling g as its
stationary distribution. IRREDUCIBILITY IS NOT PROVED on this space (Q10 says so for the same
move set): a chain that does not reach an arrangement says nothing about that arrangement.

A sweep is 2N attempted moves, the same convention as `cqg.run_chain`, so sweep counts are
comparable with the rest of the repository.
"""
import networkx as nx
import numpy as np

from .full_curvature import DEGREE, short_cycles


def read(g):
    """(valid, H, triangles, squares, pentagons) from one listing of the short cycles.

    Same definitions as `full_curvature.is_valid` and `.energy`, computed together because
    each of them pays for that listing and a trial move needs both.
    """
    cycles = short_cycles(g)
    valid = all(len(a & b) <= 1 for a, b in _pairs(cycles))
    through = {frozenset(e): [0, 0, 0] for e in g.edges}
    counts = [0, 0, 0]
    for cycle in cycles:
        counts[len(cycle) - 3] += 1
        for e in cycle:
            through[e][len(cycle) - 3] += 1
    total = 0.0
    for t, s, p in through.values():
        kappa = t / 4 - max(0.0, 1 - (2 + t + s) / 4) - max(0.0, 1 - (2 + t + s + p) / 4)
        total += -4 * 2 * kappa
    return valid, total, counts[0], counts[1], counts[2]


def _pairs(items):
    items = list(items)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            yield a, b


def propose(g, edges, rng):
    """One general switch drawn as the docstring says, or None if it cannot be made."""
    i, j = rng.choice(len(edges), size=2, replace=False)
    (a, b), (c, d) = edges[i], edges[j]
    if len({a, b, c, d}) < 4:
        return None
    new = ((a, c), (b, d)) if rng.random() < 0.5 else ((a, d), (b, c))
    if any(g.has_edge(*e) for e in new):
        return None
    h = g.copy()
    h.remove_edges_from([(a, b), (c, d)])
    h.add_edges_from(new)
    return h


def run(g, coupling, sweeps, seed, every=1):
    """Run at fixed coupling from graph g. Returns (final graph, list of rows, acceptance).

    A row is written every `every` sweeps: sweep, H, the three cycle counts, the number of
    connected pieces and the share of vertices in the largest. The graph passed in is not
    changed. `coupling` is g of the rest of the repository: the acceptance is exp(-dH/g).
    """
    if not nx.is_regular(g) or any(d != DEGREE for _, d in g.degree()):
        raise ValueError("the chain only runs on 4-regular graphs")
    rng = np.random.default_rng(seed)
    cur = g.copy()
    valid, h_cur, t, s, p = read(cur)
    if not valid:
        raise ValueError("the starting graph breaks the hard-core rule")
    n = cur.number_of_nodes()
    rows, accepted, attempts = [], 0, 0
    for sweep in range(1, sweeps + 1):
        edges = list(cur.edges)
        for _ in range(2 * n):
            attempts += 1
            trial = propose(cur, edges, rng)
            if trial is None:
                continue
            ok, h_new, t_new, s_new, p_new = read(trial)
            if not ok:
                continue
            delta = h_new - h_cur
            if delta <= 0 or rng.random() < np.exp(-delta / coupling):
                cur, h_cur, t, s, p = trial, h_new, t_new, s_new, p_new
                edges = list(cur.edges)
                accepted += 1
        if sweep % every == 0:
            parts = sorted((len(c) for c in nx.connected_components(cur)), reverse=True)
            rows.append(dict(sweep=sweep, H=h_cur, h_per_vertex=h_cur / n, triangles=t,
                             squares=s, pentagons=p, pieces=len(parts),
                             largest_frac=parts[0] / n))
    return cur, rows, accepted / attempts


def melt(g, sweeps, seed):
    """The same walk with every valid move accepted: the infinite-coupling (hottest) chain."""
    return run(g, float("inf"), sweeps, seed)


def icosidodecahedron():
    """The closed 30-point piece with a triangle and a pentagon on every edge: H = 0, like the sheet."""
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.dodecahedral_graph()))
