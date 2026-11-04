"""Checks on error bars and the autocorrelation time, on series with known answers."""
import math

import numpy as np

from graphity.analysis import autocorr_time, block_bootstrap, block_bootstrap_mean_var


def ar1(a, n, seed):
    """x[t] = a x[t-1] + noise. Known answer: tau = 1/2 + a / (1 - a)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + rng.normal()
    return x


def test_autocorr_time_of_independent_samples_is_one_half():
    x = np.random.default_rng(0).normal(size=20000)
    assert abs(autocorr_time(x) - 0.5) < 0.1


def test_autocorr_time_of_a_correlated_series():
    assert abs(autocorr_time(ar1(0.8, 50000, 1)) - 4.5) < 0.7     # 1/2 + 0.8/0.2


def test_autocorr_time_of_a_frozen_chain_is_nan():
    assert math.isnan(autocorr_time(np.full(100, 0.93)))


def test_bootstrap_error_of_independent_samples():
    x = np.random.default_rng(2).normal(loc=3.0, scale=2.0, size=20000)
    mean, err, var, var_err = block_bootstrap_mean_var(x, seed=3)
    assert abs(mean - 3.0) < 5 * err
    assert 0.6 < err / (2.0 / math.sqrt(len(x))) < 1.4             # textbook sigma / sqrt(n)
    assert abs(var - 4.0) < 5 * var_err


def test_bootstrap_error_grows_with_correlation():
    """Correlated samples carry less information, so the error bar must be wider."""
    x = ar1(0.8, 20000, 4)
    naive = x.std() / math.sqrt(len(x))
    _, err, _, _ = block_bootstrap_mean_var(x, seed=5)
    assert err > 2 * naive                                         # exact ratio is sqrt(2 tau) = 3


def test_new_bootstrap_agrees_with_the_original():
    """At beta = 1 the specific heat is the variance, so the two must match exactly."""
    x = ar1(0.5, 2000, 6)
    assert block_bootstrap(x, 1.0, seed=7) == block_bootstrap_mean_var(x, seed=7)
