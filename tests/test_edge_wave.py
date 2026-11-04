"""Form B of the waves brief (docs/design/waves_brief.md), built 1 November 2026: dynamical edge waves with one
inertia pinned by the owner's one c. Rule 6: the stiffness against an independently built dense matrix, the plane-wave
eigenvalue exactly, energy conserved, the speed at long wavelength, families independent, same-seed reproducibility."""
import numpy as np

from graphity.edge_wave import (DT, M_PINNED, dense_stiffness, energy, long_wave_speed, omega, plane_wave,
                                random_state, run, stiffness_apply, wavevector)


def test_stiffness_matches_an_independently_built_dense_matrix():
    dims = (4, 3, 5)
    rng = np.random.default_rng(1)
    delta = rng.standard_normal((3,) + dims)
    K = dense_stiffness(dims)
    for fam in range(3):
        want = K @ delta[fam].ravel()
        assert np.allclose(stiffness_apply(delta)[fam].ravel(), want, atol=1e-12)
    # K is symmetric, 4 times the lattice Laplacian: row sums zero (uniform scaling is free)
    assert np.allclose(K, K.T) and np.allclose(K.sum(axis=1), 0.0)


def test_a_plane_wave_is_an_eigenvector_with_the_exact_frequency():
    dims = (8, 8, 8)
    for j in ((1, 0, 0), (2, 1, 0), (3, 3, 1), (4, 4, 4)):
        delta, _ = plane_wave(dims, 0, j)
        k = wavevector(dims, j)
        want = M_PINNED * omega(k) ** 2 * delta
        assert np.allclose(stiffness_apply(delta), want, atol=1e-12)
    # the band top: omega^2 = 48 / m = 12 at m = 4
    assert np.isclose(omega(wavevector(dims, (4, 4, 4))) ** 2, 48.0 / M_PINNED)


def test_one_c_pins_the_inertia_at_four_and_the_speed_at_one_link_per_sweep():
    assert long_wave_speed(M_PINNED) == 1.0
    assert long_wave_speed(1.0) == 2.0
    k = wavevector((64, 64, 64), (1, 0, 0))
    assert abs(omega(k) / np.linalg.norm(k) - 1.0) < 2e-3   # lattice dispersion at wavelength 64 is below 0.2 %


def test_energy_is_conserved_by_the_leapfrog():
    dims = (8, 8, 8)
    delta, vel = random_state(dims, seed=3)
    _, _, record = run(delta, vel, sweeps=20.0, dt=DT, record_every=20)
    e = np.array([r[1] for r in record])
    assert e[0] > 0
    assert np.max(np.abs(e - e[0])) / e[0] < 2e-3      # (omega dt)^2 / 8 at the band top is 0.0038; a random state sits below it
    # a long standing wave: far tighter
    delta, vel = plane_wave(dims, 1, (1, 0, 0))
    _, _, record = run(delta, vel, sweeps=20.0, record_every=20)
    e = np.array([r[1] for r in record])
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-3      # (omega dt)^2 / 8 = 1.8e-4 at omega = 0.765; measured 3.7e-4, no drift


def test_a_travelling_wave_moves_at_one_link_per_sweep_at_long_wavelength():
    dims = (64, 4, 4)
    delta, vel = plane_wave(dims, 0, (1, 0, 0), travelling=True)
    T = 16.0
    delta_T, _, _ = run(delta, vel, sweeps=T)
    k = wavevector(dims, (1, 0, 0))[0]
    # delta(t) = A cos(k x - omega t): the crest moves omega t / k links; at m = 4 that is t to within dispersion
    x = np.arange(dims[0])
    want = 1e-3 * np.cos(k * x - omega(wavevector(dims, (1, 0, 0))) * T)
    assert np.allclose(delta_T[0][:, 0, 0], want, atol=2e-6)
    shift = omega(wavevector(dims, (1, 0, 0))) * T / k
    assert abs(shift - T) / T < 2e-3


def test_the_families_never_couple():
    dims = (8, 8, 8)
    delta, vel = plane_wave(dims, 0, (0, 0, 2), travelling=True)
    delta_T, vel_T, _ = run(delta, vel, sweeps=5.0)
    assert np.all(delta_T[1:] == 0.0) and np.all(vel_T[1:] == 0.0)
    # the "+" pattern: x-lengths up where y-lengths are down, both along z, propagate together unchanged in shape
    dx, vx = plane_wave(dims, 0, (0, 0, 1), travelling=True)
    dy, vy = plane_wave(dims, 1, (0, 0, 1), travelling=True)
    d, v = dx - dy, vx - vy
    d_T, _, _ = run(d, v, sweeps=5.0)
    assert np.allclose(d_T[0], -d_T[1], atol=1e-15)


def test_same_seed_reproduces_bit_for_bit():
    dims = (6, 6, 6)
    a = run(*random_state(dims, seed=11), sweeps=3.0)
    b = run(*random_state(dims, seed=11), sweeps=3.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_energy_of_a_pure_scaling_is_zero():
    delta = np.ones((3, 5, 5, 5))
    assert energy(delta, np.zeros_like(delta)) == 0.0
