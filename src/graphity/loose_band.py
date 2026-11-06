"""The loose band: a 4 x L tube packed everywhere except one band of columns, loosened by random valid switches
(her picture of 5 November 2026: the burp's cycle leaves a lower-density remnant inside the denser layers; decided
6 November: the slab is a doughnut of squares, the toy's tube is that doughnut unrolled, and the loose band is the
inner rim where the last burp's remnant sits). Built as the sandbox look `sandbox/loose_band_look.py` on 5 November;
moved here for T107 (registered 7 November).

EXACT: a switch moves two links among four points; `loosen` accepts only switches whose four points all lie in the
band and whose result keeps the hard-core rule (no two points share more than two neighbours; the cap), so the
wiring stays bipartite and valid, and squares are lost only inside the band (a square needs all four of its points'
links, and only links inside the band change). k is the number of switches made; a switch that fails the rule is
undone and does not count.
"""
import numpy as np

from graphity.cqg import NO_CAP, _new_edge_ok, _switch, has_edge, torus


def band_mask(L, width, centre=None):
    """The points of the 4 x L tube whose column lies within width/2 of the centre column (circular)."""
    centre = L // 2 if centre is None else centre
    x_of = np.arange(4 * L) // 4
    return np.abs(((x_of - centre + L // 2) % L) - L // 2) < width / 2


def loosen(adj, part, rng, in_band, k, cap=NO_CAP):
    """k random valid switches among links with all four points in the band. Returns the number made."""
    side_u = np.flatnonzero((part == 0) & in_band)
    made, tries = 0, 0
    while made < k and tries < 100000:
        tries += 1
        u1, u2 = side_u[rng.integers(len(side_u))], side_u[rng.integers(len(side_u))]
        v1, v2 = adj[u1, rng.integers(4)], adj[u2, rng.integers(4)]
        if u1 == u2 or v1 == v2 or not (in_band[v1] and in_band[v2]):
            continue
        if has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
            continue
        _switch(adj, u1, v1, u2, v2)
        if _new_edge_ok(adj, u1, v2, cap) and _new_edge_ok(adj, u2, v1, cap):
            made += 1
        else:
            _switch(adj, u1, v2, u2, v1)
    return made


def loose_band_tube(L, k, rng, width=None, cap=NO_CAP):
    """The 4 x L tube with its band of `width` columns (default L/8) loosened by k switches.
    Returns adj, part, in_band, made."""
    width = L // 8 if width is None else width
    adj, part = torus(L, 4, cap)
    in_band = band_mask(L, width)
    made = loosen(adj, part, rng, in_band, k, cap)
    return adj, part, in_band, made
