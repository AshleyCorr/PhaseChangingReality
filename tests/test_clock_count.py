"""The per-point move counter of run_chain (T101, option A): it changes nothing in the chain and counts every accepted
switch four times, once per point it touches."""
import numpy as np

from graphity.cqg import NO_CAP, run_chain, torus


def _tube():
    adj, part = torus(16, 4, NO_CAP)
    return adj, np.flatnonzero(part == 0)


def test_counting_leaves_the_chain_bit_for_bit_unchanged():
    adj_a, side = _tube()
    adj_b = adj_a.copy()
    s_a, x_a, acc_a = run_chain(adj_a, side, 1 / 1.5, 0, 200, 7, 1.25, NO_CAP, False)
    moves = np.zeros(adj_b.shape[0], dtype=np.int64)
    s_b, x_b, acc_b = run_chain(adj_b, side, 1 / 1.5, 0, 200, 7, 1.25, NO_CAP, False, None, moves)
    assert np.array_equal(s_a, s_b) and np.array_equal(x_a, x_b) and acc_a == acc_b
    assert np.array_equal(adj_a, adj_b)


def test_every_accepted_switch_counts_four_points():
    adj, side = _tube()
    moves = np.zeros(adj.shape[0], dtype=np.int64)
    sweeps = 300
    _, _, acc = run_chain(adj, side, 1 / 1.5, 0, sweeps, 11, 1.25, NO_CAP, False, None, moves)
    attempts = 2 * adj.shape[0] * sweeps
    assert moves.sum() == 4 * round(acc * attempts)
    assert moves.sum() > 0


def test_a_frozen_sheet_counts_nothing():
    adj, part = torus(8, 8, NO_CAP)
    moves = np.zeros(adj.shape[0], dtype=np.int64)
    run_chain(adj, np.flatnonzero(part == 0), 1 / 0.25, 0, 50, 3, 1.25, NO_CAP, False, None, moves)
    assert moves.sum() == 0
