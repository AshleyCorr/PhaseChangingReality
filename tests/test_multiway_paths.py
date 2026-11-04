"""T95's path count (scripts/multiway_paths.py), checked against T4's exact class space at N = 16 and a hand case."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multiway_paths as mp  # noqa: E402
from graphity.small_graphs import circulant, explore  # noqa: E402


def test_move_counts_match_the_class_graph_and_the_move_total_at_n8():
    classes, joined = explore(circulant(8))
    m = mp.move_counts(classes)
    assert len(classes.reps) == 5                       # T4: five classes at N = 16
    for k in range(5):
        assert {j for j in range(5) if m[k, j] > 0 and j != k} == joined[k]
    # moves into class j from class k, times the states of k, equal moves from j into k times the states of j (reversibility)
    a = np.array([classes.symmetries(j) for j in range(5)], dtype=float)
    states = 1 / a
    flow = m * states[:, None]
    assert np.allclose(flow, flow.T)


def test_path_shares_are_normalised_and_start_at_the_start():
    m = np.array([[0.0, 2.0], [1.0, 0.0]])
    shares = mp.path_shares(m, 0, 4)
    assert all(abs(s.sum() - 1) < 1e-12 for s in shares)
    assert shares[0][1] == 1.0                           # one step from class 0 lands in class 1


def test_long_path_share_is_the_labelled_stationary_law_on_a_hand_chain():
    # two classes: class 0 has 3 labelled states with 2 moves each (all into class 1); class 1 has 1 state with 6 moves into class 0
    m = np.array([[0.0, 2.0], [6.0, 0.0]])
    shares = mp.path_shares(m, 0, 40)
    long_share = (shares[-1] + shares[-2]) / 2
    labelled = np.array([3.0, 1.0])
    deg = m.sum(axis=1)
    target = labelled * deg / (labelled * deg).sum()
    assert mp.within(long_share, target)
