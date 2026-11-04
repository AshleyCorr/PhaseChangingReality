"""The phi analysis, checked against joint (S, X) histograms whose answer is known.

The point of amendment 1 is that phi has no comb in it where the energy does. That is an
arithmetic claim, so it can be tested rather than asserted: the second test below builds a case
whose energy spectrum is provably combed and whose phi distribution is a single clean hump, and
requires the phi analysis to report no barrier.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("analyse_t6_phi", SCRIPTS / "analyse_t6_phi.py")
phi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phi)


def joint(n, lam, g, s_centres, weights, sigma, x_of_s=lambda s: 0, total=4e6, lo=None, hi=None):
    """A joint (S, X) histogram with a prescribed distribution over S."""
    lo = 2 if lo is None else lo
    hi = n - 2 if hi is None else hi
    s = np.arange(lo, hi + 1)
    p = np.zeros_like(s, dtype=float)
    for c, w in zip(s_centres, weights):
        p += w * np.exp(-0.5 * ((s - c) / sigma) ** 2)
    p /= p.sum()
    x = np.array([x_of_s(v) for v in s], dtype=int)
    return dict(N=np.int64(n), lam=np.float64(lam), g=np.float64(g),
                s_bin=s.astype(np.int32), x_bin=x.astype(np.int32),
                sx_counts=np.maximum(np.round(p * total), 0).astype(np.int64))


def test_two_humps_in_phi_are_found_with_the_right_gap_and_energy():
    n, lam, g = 100, 0.0, 6.0
    d = joint(n, lam, g, [30, 70], [0.5, 0.5], 6.0, total=8e6)
    got = phi.observables(d, 1)
    assert got is not None, "a clearly bimodal phi distribution was reported as one hump"
    assert got["d_phi"] == pytest.approx(0.40, abs=0.03)
    # lambda = 0 so H = 16(N - S); the two phases sit 40 squares apart, hence 16*40/N per point
    assert got["latent"] == pytest.approx(16 * 40 / n, rel=0.10), got["latent"]
    expected = 0.5 * (40.0 / (2 * 6.0)) ** 2 - np.log(2.0)
    assert got["barrier"] == pytest.approx(expected, rel=0.10), (got["barrier"], expected)


def joint2d(n, lam, g, s_mid, s_sig, x_mid, x_sig, total=2e6):
    """S and X each spread over a range, as in a real run. H then lands on the lattice generated
    by 16 and 4*lambda with a multiplicity that varies from one energy to the next."""
    s = np.arange(max(2, int(s_mid - 4 * s_sig)), int(s_mid + 4 * s_sig) + 1)
    x = np.arange(max(0, int(x_mid - 4 * x_sig)), int(x_mid + 4 * x_sig) + 1)
    ps = np.exp(-0.5 * ((s - s_mid) / s_sig) ** 2)
    px = np.exp(-0.5 * ((x - x_mid) / x_sig) ** 2)
    ss, xx = np.meshgrid(s, x, indexing="ij")
    w = np.outer(ps, px)
    w = w / w.sum() * total
    return dict(N=np.int64(n), lam=np.float64(lam), g=np.float64(g),
                s_bin=ss.ravel().astype(np.int32), x_bin=xx.ravel().astype(np.int32),
                sx_counts=np.round(w.ravel()).astype(np.int64))


def test_a_combed_energy_spectrum_does_not_fool_phi():
    """The case amendment 1 exists for: teeth in H, one clean hump in phi."""
    n, lam, g = 36, 1.25, 3.0
    d = joint2d(n, lam, g, s_mid=20, s_sig=3.0, x_mid=7, x_sig=2.5)

    # First establish that this really is a combed energy spectrum, so the test has teeth.
    h = phi.energy(d["s_bin"].astype(int), d["x_bin"].astype(int), n, lam)
    lev, idx = np.unique(h, return_inverse=True)
    c = np.bincount(idx, weights=d["sx_counts"].astype(float))
    c = c[c > 0]
    ratios = np.maximum(c[1:], 1) / np.maximum(c[:-1], 1)
    assert np.sum((ratios > 10) | (ratios < 0.1)) >= 3, \
        "the fixture is not actually combed; worst ratio %.1f" % max(ratios.max(), 1 / ratios.min())

    # ...and that phi, over the same sweeps, is one clean hump.
    assert phi.observables(d, 1) is None, "phi inherited the comb it is supposed to be free of"


def test_one_hump_in_phi_is_one_hump():
    d = joint(100, 0.0, 6.0, [50], [1.0], 8.0)
    assert phi.observables(d, 1) is None


def test_counting_noise_in_phi_is_not_a_barrier():
    n, rng = 100, np.random.default_rng(20260921)
    found = 0
    for _ in range(20):
        d = joint(n, 0.0, 6.0, [50], [1.0], 8.0, total=2e5)
        d["sx_counts"] = rng.poisson(d["sx_counts"]).astype(np.int64)
        if phi.observables(d, 1) is not None:
            found += 1
    assert found == 0, "%d of 20 noisy single humps were reported as two" % found


def test_an_unvisited_phi_is_not_a_deep_phi():
    """The same confusion as in the energy analysis, which the shared guard must still catch."""
    n = 100
    d = joint(n, 0.0, 6.0, [30, 70], [1.0, 1e-12], 5.0, total=2e5)
    got = phi.observables(d, 1)
    if got is not None:
        assert np.isfinite(got["barrier"]) and got["barrier"] < 50, got["barrier"]
