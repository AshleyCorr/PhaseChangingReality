"""T94's matter window (scripts/carroll_matter_window.py), checked on cases with known answers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import carroll_matter_window as mw  # noqa: E402
import carroll_reader as cr  # noqa: E402
from graphity.cqg import NO_CAP, torus  # noqa: E402


def _flat():
    return cr.graph_of(torus(8, 8, NO_CAP)[0])


def test_stored_pairs_are_the_two_diagonals_of_every_square():
    g = _flat()
    pairs = mw.stored_pairs(g)
    assert len(pairs) == 2 * 64                       # 64 squares on the 8 x 8 torus, two diagonals each
    for a, b in pairs:
        assert not g.has_edge(a, b) and len(set(g[a]) & set(g[b])) == 2


def test_patch_is_eight_pairs_round_the_center():
    g = _flat()
    patch = mw.pairs_at_center(mw.stored_pairs(g), g, 0)
    assert len(patch) == 8
    assert all(0 in p or set(p) <= set(g[0]) for p in patch)


def test_alpha_two_reproduces_empty_space():
    g = _flat()
    pairs = mw.stored_pairs(g)
    _, d_empty = mw.window_weighted(g, mw.weighted_graph(g, [], 1.0), 0, 3)
    _, d_two = mw.window_weighted(g, mw.weighted_graph(g, pairs, 2.0), 0, 3)
    assert np.allclose(d_empty, d_two)
    _, d_plain = cr.window(g, 0, 3)
    assert np.allclose(d_empty, d_plain)


def test_a_shared_diagonal_is_one_step_and_the_window_keeps_its_points():
    g = _flat()
    pairs = mw.stored_pairs(g)
    a, b = pairs[0]
    pts, d = mw.window_weighted(g, mw.weighted_graph(g, pairs, 1.0), a, 2)
    assert set(pts) == set(cr.window(g, a, 2)[0])
    assert d[pts.index(a)][pts.index(b)] == 1.0
