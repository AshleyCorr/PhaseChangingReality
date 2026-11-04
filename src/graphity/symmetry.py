"""How many renamings leave every relationship intact: the fast count (ASSUMPTIONS Q15, addendum).

WHY. Treating points as interchangeable weights each arrangement by its number of symmetries A, the
renamings of the points that keep every edge (Q15). The project's first counter
(`small_graphs.log_automorphisms`, networkx isomorphism search) takes 1 to 45 seconds per count at
N = 64 to 160, which rules out using A inside a chain. This one calls igraph's `count_automorphisms`
(the bliss algorithm, from general knowledge; not read by us) and takes about a millisecond at the same
sizes, with the same answers (tests/test_symmetry.py checks it against the slow counter).

WHAT IS COUNTED. Renamings that map each side of the bipartition to itself, exactly as Q15 counts them
(the slow counter works on graphs renumbered by `sides_first` and keeps the sides apart); the sides are
passed to igraph as vertex colours.
"""
import numpy as np

try:
    import igraph as _ig
except ImportError:                                   # the slow counter still works without it
    _ig = None


def available():
    return _ig is not None


def count(adj, part):
    """Number of side-preserving automorphisms of the graph with neighbour array `adj`."""
    if _ig is None:
        raise ImportError("igraph is not installed; use small_graphs.log_automorphisms")
    n = adj.shape[0]
    edges = [(u, int(v)) for u in range(n) for v in adj[u] if u < v]
    g = _ig.Graph(n=n, edges=edges)
    return int(g.count_automorphisms(color=[int(c) for c in np.asarray(part)]))


def canonical_key(adj, part):
    """A hashable form that two graphs share exactly when a side-preserving renaming maps one onto the
    other (igraph's canonical labelling, the same bliss algorithm as `count`)."""
    if _ig is None:
        raise ImportError("igraph is not installed")
    n = adj.shape[0]
    colour = [int(c) for c in np.asarray(part)]
    edges = [(u, int(v)) for u in range(n) for v in adj[u] if u < v]
    g = _ig.Graph(n=n, edges=edges)
    g.vs["side"] = colour                              # carried through the permutation by igraph itself
    h = g.permute_vertices(g.canonical_permutation(color=colour))
    return tuple(h.vs["side"]), tuple(sorted(tuple(sorted(e)) for e in h.get_edgelist()))
