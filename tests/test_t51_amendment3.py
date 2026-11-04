"""T51's Amendment 3 reading (scripts/analyse_t51_amendment3.py): the power-law fit, its inversion, and the
piece readings on hand-built graphs."""
import importlib.util
import math
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, torus

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a3 = _load("analyse_t51_amendment3")


def _rec(replica, c1, h1, n=1024, lam=1.25):
    return dict(replica=replica, N=n, lam=lam, c1=c1, h1=h1, melted=False)


def test_power_law_recovers_an_exact_exponent():
    ts = [10000, 30000, 100000, 300000]
    alpha, beta = a3.power_law(ts, [3.0 * t ** -0.5 for t in ts])
    assert abs(alpha + 0.5) < 1e-9 and abs(math.exp(beta) - 3.0) < 1e-9


def test_power_law_refuses_a_zero_and_a_single_point():
    assert all(math.isnan(x) for x in a3.power_law([1, 2], [1.0, 0.0]))
    assert all(math.isnan(x) for x in a3.power_law([1], [1.0]))


def test_fit_with_error_on_exact_data_has_the_exponent_and_no_spread():
    # every replica identical, so every resample gives the same cell means and the same alpha
    by_t = {t: [_rec(i, 2.0 * (t / 10000) ** -1.0, 0.04 * (t / 10000) ** -0.5) for i in range(8)]
            for t in (10000, 30000, 100000)}
    fit = a3.fit_with_error(by_t, lambda r: r["c1"])
    assert abs(fit["alpha"] + 1.0) < 1e-9 and fit["se"] < 1e-9 and fit["n_left_out"] == 0
    fit = a3.fit_with_error(by_t, a3.frozen_share)
    assert abs(fit["alpha"] + 0.5) < 1e-9
    assert abs(fit["means"][10000] - 0.04 / 1024) < 1e-12


def test_fit_leaves_out_resamples_whose_cell_mean_is_zero():
    by_t = {10000: [_rec(0, 1.0, 1.0), _rec(1, 0.0, 1.0)], 100000: [_rec(0, 0.5, 1.0), _rec(1, 0.0, 1.0)]}
    fit = a3.fit_with_error(by_t, lambda r: r["c1"])
    assert 0 < fit["n_left_out"] < a3.BOOT_N          # resamples drawing only replica 1 have a zero mean


def test_time_to_reach_inverts_the_fit_and_refuses_a_rising_one():
    fit = dict(alpha=-0.5, beta=math.log(0.04))
    t = a3.time_to_reach(fit, 7e-7)
    assert abs(0.04 * t ** -0.5 - 7e-7) < 1e-15
    assert math.isnan(a3.time_to_reach(dict(alpha=0.2, beta=0.0), 7e-7))


def test_piece_readings_on_a_flat_torus_and_a_tube():
    flat, _ = torus(8, 8, NO_CAP)
    assert a3.piece_sizes(flat) == ([], 0) and a3.whole_graph_pieces(flat) == 1
    tube, _ = torus(16, 4, NO_CAP)                               # every point at d = 1: one piece of 64, not a column
    assert a3.piece_sizes(tube) == ([64], 0) and a3.whole_graph_pieces(tube) == 1
    two = np.full((flat.shape[0] + tube.shape[0], 4), -1, dtype=np.int64)
    two[:flat.shape[0]] = flat
    two[flat.shape[0]:] = tube + flat.shape[0]
    assert a3.whole_graph_pieces(two) == 2


def test_size_histogram_counts_sizes_and_one_piece_graphs():
    recs = [(0, [4, 4, 9], 2, 1), (1, [4], 1, 2), (2, [], 0, 1)]
    hist, one = a3.size_histogram(recs)
    assert dict(hist) == {4: 3, 9: 1} and one == 2
