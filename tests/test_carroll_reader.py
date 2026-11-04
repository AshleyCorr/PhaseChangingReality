"""T91's reader (scripts/carroll_reader.py): classical MDS on graph-distance windows, checked on cases with known answers."""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import carroll_reader as cr  # noqa: E402


def test_points_on_a_line_read_one():
    x = np.arange(7.0)
    w = cr.mds_spectrum(np.abs(x[:, None] - x[None, :]))
    assert w[0] > 0 and np.allclose(w[1:], 0, atol=1e-9)
    assert cr.d_mds(w) == 1 and cr.eps(w, 1) < 1e-12


def test_points_in_a_plane_read_two():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(20, 2))
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    w = cr.mds_spectrum(d)
    assert np.allclose(w[2:], 0, atol=1e-9) and cr.d_mds(w) == 2


def test_four_cycle_spectrum_is_two_two_zero_minus_one():
    d = nx.floyd_warshall_numpy(nx.cycle_graph(4))
    assert np.allclose(cr.mds_spectrum(d), [2, 2, 0, -1])


def test_window_uses_full_graph_distances():
    g = nx.cycle_graph(10)
    pts, dist = cr.window(g, 0, 2)
    assert pts == [0, 1, 2, 8, 9]
    assert dist[pts.index(2)][pts.index(8)] == 4          # round through 0, not the long way


def test_flat_torus_center_is_two_dimensional_locally_by_the_toy():
    from graphity.cqg import NO_CAP, torus
    adj, _ = torus(16, 16, NO_CAP)
    rows = cr.read_state(adj, "flat", (2,), centers=[0])
    assert rows[0]["d_local"] == 2 and rows[0]["points"] == 13
