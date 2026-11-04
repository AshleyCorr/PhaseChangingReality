"""The T6 analysis, checked against histograms whose answer is known by construction.

WHY THIS FILE EXISTS. The sampling is checked against exact counts elsewhere. The *analysis* is
not, and it is where a silent error would do the most damage: it decides where the transition
is, how far apart the two humps are and how deep the valley between them is, and every one of
those feeds the pre-registered verdict. So here the histograms are built by hand with the answer
written in, and the analysis has to find it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "analyse_t6_hist", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t6_hist.py")
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)


def two_gaussian_histogram(e_lo, e_hi, weight_hi, sigma, step=4.0, span=260.0, total=4e6):
    """A histogram measured at some coupling: a known mixture of two Gaussians.

    A measured histogram IS the distribution at the coupling it was measured at, so nothing is
    pre-divided here. The reweighter's job is to move it to OTHER couplings.
    """
    levels = np.arange(min(e_lo, e_hi) - span, max(e_lo, e_hi) + span + step, step)
    p = ((1 - weight_hi) * np.exp(-0.5 * ((levels - e_lo) / sigma) ** 2)
         + weight_hi * np.exp(-0.5 * ((levels - e_hi) / sigma) ** 2))
    return levels, p / p.sum() * total


def test_it_finds_the_gap_and_the_valley_it_was_given():
    n = 100
    e_lo, e_hi, sigma, g = 200.0, 560.0, 42.0, 6.0
    levels, counts = two_gaussian_histogram(e_lo, e_hi, 0.5, sigma)
    got = ana.observables(levels, counts, g, n)
    assert got is not None, "a clearly bimodal histogram was reported as having one hump"
    assert got["latent"] == pytest.approx((e_hi - e_lo) / n, rel=0.06)
    # Two equal Gaussians have a known valley depth. Both humps reach the midpoint between them,
    # so the floor there is twice one hump's tail and the depth is short by ln 2.
    expected = 0.5 * ((e_hi - e_lo) / (2 * sigma)) ** 2 - np.log(2.0)
    assert got["barrier"] == pytest.approx(expected, rel=0.03), (got["barrier"], expected)
    assert got["shift"] < 0.05, "a balanced histogram should not need reweighting to balance it"


def test_it_reweights_to_the_balance_point_rather_than_using_the_one_it_was_given():
    """Given a lopsided histogram, the analysis must move the coupling until the humps match."""
    n = 100
    levels, counts = two_gaussian_histogram(200.0, 560.0, 0.12, 42.0)
    got = ana.observables(levels, counts, 6.0, n)
    assert got is not None
    assert got["imbalance"] < 0.15, "did not reach a balanced pair of humps"
    assert got["shift"] > 0.005, "claims it was already balanced when it was 12:88"


def test_one_hump_is_reported_as_one_hump():
    n = 100
    levels = np.arange(0.0, 800.0, 4.0)
    counts = np.exp(-0.5 * ((levels - 400.0) / 60.0) ** 2) * 1e6
    assert ana.observables(levels, counts, 6.0, n) is None


def test_a_ripple_a_few_levels_wide_is_not_two_humps():
    """Kelly22's warning: a discrete spectrum can make one hump look like several."""
    n = 100
    levels = np.arange(0.0, 800.0, 4.0)
    smooth = np.exp(-0.5 * ((levels - 400.0) / 60.0) ** 2)
    ripple = smooth * (1.0 + 0.04 * np.cos(levels / 4.0 * np.pi))     # wobbles every 2 levels
    assert ana.observables(levels, ripple * 1e6, 6.0, n) is None


def test_a_reweight_beyond_the_window_is_refused():
    """A pair of humps that only balances far outside the window must say so, not pretend."""
    n = 100
    levels, counts = two_gaussian_histogram(200.0, 560.0, 1e-6, 42.0)
    got = ana.observables(levels, counts, 6.0, n)
    if got is not None:
        assert got["shift"] <= ana.MAX_SHIFT + 1e-9, "reweighted further than it admits"
        assert got["imbalance"] > 0.5, "claims humps are balanced when the window never got there"


def test_counting_noise_on_one_hump_is_not_a_barrier():
    """The commonest false positive: a lumpy histogram from finite sampling."""
    n, rng = 100, np.random.default_rng(20260921)
    levels = np.arange(0.0, 800.0, 4.0)
    shape = np.exp(-0.5 * ((levels - 400.0) / 60.0) ** 2)
    found = 0
    for _ in range(25):
        counts = rng.poisson(shape / shape.sum() * 2e5).astype(float)
        if ana.observables(levels, counts, 6.0, n) is not None:
            found += 1
    assert found == 0, "%d of 25 noisy single humps were reported as two" % found
