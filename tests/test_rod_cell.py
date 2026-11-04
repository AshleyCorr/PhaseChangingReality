"""T93's builder and readers (scripts/run_rod_cell.py): exact checks on the built states."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_rod_cell as rc  # noqa: E402
from graphity.cqg_d import is_valid  # noqa: E402
from graphity.dimension import local_dimension_d  # noqa: E402


def test_the_cell_is_zero_dimensional_and_the_rod_one_dimensional_before_joining():
    from graphity.cqg_d import torus
    assert set(local_dimension_d(torus((4, 4, 4))[0])) == {0}
    assert set(local_dimension_d(torus((4, 4, 16))[0])) == {1}
    assert set(local_dimension_d(torus((4, 4, 8))[0])) == {1}


def test_joined_states_are_valid_six_regular_bipartite_and_connected():
    for kind in ("cell", "rodpiece"):
        adj, part, n_rod, mid = rc.build(kind, 16)
        assert is_valid(adj)
        assert adj.shape[1] == 6
        for u in range(len(adj)):
            for v in adj[u]:
                assert part[u] != part[v]
        comps = rc.open_regions(adj, np.ones(len(adj), dtype=bool))
        assert len(comps) == 1
        assert n_rod == 256 and mid == 8


def test_most_of_the_attached_cell_stays_zero_dimensional():
    adj, part, n_rod, mid = rc.build("cell", 16)
    d = local_dimension_d(adj)
    assert (d[n_rod:] == 0).sum() == 44            # exact: the two joins disturb 20 of the 64 cell points; 44 stay at d = 0


def test_circular_helpers():
    assert rc.circular_distance(1, 15, 16) == 2
    assert abs(rc.circular_mean_layer([15, 1], 16) - 0) < 1e-9 or abs(rc.circular_mean_layer([15, 1], 16) - 16) < 1e-9


def test_plain_rod_is_the_t88_rod():
    adj, part, n_rod, mid = rc.build("plain", 16)
    assert n_rod == len(adj) == 256 and set(local_dimension_d(adj)) == {1}
