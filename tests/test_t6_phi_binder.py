"""T6 amendment 5 (a): the phi cumulant and criterion 3, checked on cases whose answer is known.

The cumulant decides a pre-registered verdict, so it is checked against the textbook two-spike
value, against a single narrow hump (which must sit just below 2/3), and the criterion against
gap sequences that do and do not shrink.
"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "analyse_t6_phi_binder",
    Path(__file__).resolve().parents[1] / "scripts" / "analyse_t6_phi_binder.py")
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)


def test_two_equal_spikes_give_the_textbook_value():
    n, a, b = 100, 0.4, 1.0
    u = ana.phi_cumulant([a * n, b * n], [1.0, 1.0], n)
    expected = 1 - 2 * (a ** 4 + b ** 4) / (3 * (a ** 2 + b ** 2) ** 2)
    assert abs(u - expected) < 1e-12
    assert u < 2 / 3 - 0.05


def test_one_narrow_hump_away_from_zero_sits_just_below_two_thirds():
    n = 100
    s = np.arange(40, 81)
    w = np.exp(-0.5 * ((s - 60) / 3.0) ** 2)
    u = ana.phi_cumulant(s, w, n)
    assert u < 2 / 3
    assert 2 / 3 - u < 0.01


def test_narrower_hump_is_closer_to_two_thirds():
    n = 100
    s = np.arange(0, 201)
    wide = np.exp(-0.5 * ((s - 100) / 10.0) ** 2)
    narrow = np.exp(-0.5 * ((s - 100) / 5.0) ** 2)
    assert ana.phi_cumulant(s, narrow, n) > ana.phi_cumulant(s, wide, n)


def test_criterion_needs_the_gap_to_shrink_at_every_step():
    t = 2 / 3
    assert ana.approaches_two_thirds(np.array([36, 64, 100]), np.array([t - 0.03, t - 0.02, t - 0.01]))
    assert not ana.approaches_two_thirds(np.array([36, 64, 100]), np.array([t - 0.03, t - 0.04, t - 0.01]))
    assert not ana.approaches_two_thirds(np.array([36, 64, 100]), np.array([t - 0.01, t - 0.02, t - 0.03]))
    # order of the input does not matter, only the sizes
    assert ana.approaches_two_thirds(np.array([100, 36, 64]), np.array([t - 0.01, t - 0.03, t - 0.02]))


def test_rung_minimum_is_no_larger_than_the_value_at_the_run_coupling():
    n, lam, g0 = 64, 1.25, 3.0
    s_bin = np.repeat(np.arange(20, 61), 3)
    x_bin = np.tile(np.array([0, 2, 4]), 41)
    counts = 1e4 * np.exp(-0.5 * ((s_bin - 40) / 5.0) ** 2) * np.exp(-0.3 * x_bin)
    d = dict(N=n, lam=lam, g=g0, s_bin=s_bin, x_bin=x_bin, sx_counts=counts)
    u_min, g_at = ana.rung_minimum(d)
    per_s = np.bincount(s_bin - 20, weights=counts)
    u_here = ana.phi_cumulant(np.arange(20, 61), per_s, n)
    assert u_min <= u_here + 1e-12
    assert g0 * 0.75 - 1e-9 <= g_at <= g0 * 1.25 + 1e-9
