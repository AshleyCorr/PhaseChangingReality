"""Dynamical edge waves on the flat six-link torus: form B of docs/design/waves_brief.md, built on the owner's decision of
1 November 2026 ("Yes proceed", NOTEBOOK; decision (1) confirmed, decision (2) taken with it).

THE CLOCK (hers, 31 October to 1 November). Time is counted in sweeps and speeds in links per sweep. Light moves one
link per sweep (the photon as built, graphity.twist.Photon.advance), so c = 1 link per sweep. For a wave on a fixed
lattice, where nothing is offered at random, the sweep is the chain's sweep; the fair sweep of O90 applies only where a
wave runs beside the Monte Carlo chain (ours, accepted with her "proceed").

THE EQUATION (exact). Each axis family of link lengths carries a field delta (the change of length from flat) and a
velocity; the restoring force is the curvature's own stiffness at the flat point, K = 4 (6 I - A) per family (the
gravity link brief, step 1; graphity.lengths.STIFFNESS), A the cubic adjacency of link positions. With one inertia m per
unit of length,

    m d2(delta)/dt2 = -K delta,      E = (1/2) m |v|^2 + (1/2) delta^T K delta.

The families never couple (K is block diagonal by family), so a wave in the x-lengths leaves the y- and z-lengths alone.

WHAT ONE c FIXES (exact). A plane wave exp(i k.x) is an eigenvector of K with K delta = 4 sum_i 2 (1 - cos k_i) delta, so

    omega^2 = (4 / m) sum_i 2 (1 - cos k_i),     omega -> (2 / sqrt(m)) |k| at long wavelength.

The long-wave speed is 2 / sqrt(m) links per sweep. Her one c for light and gravity (Update 33, answer (b)) with light at
one link per sweep therefore pins m = 4: no free constant (M_PINNED). The band top is omega^2 = 48 / m = 12, so the
leapfrog step must satisfy dt < 2 / sqrt(12) = 0.577 sweeps; DT = 0.05 is the default.

THE INTEGRATOR. Leapfrog (kick, drift, kick), time-reversible and symplectic; the energy above oscillates about its
start by a relative amount of order (omega dt)^2 / 8 and does not drift. Tests: K against an independently built dense
matrix (rule 6's brute force), the plane-wave eigenvalue exactly, energy conserved, the travelling wave moving at
2 / sqrt(m) = 1 link per sweep at m = 4, families independent, same-seed reproducibility.

WHAT THIS IS NOT. Spin 2 is not here (the lengths carry no turning angle); only the squeeze-stretch "+" pattern of a
gravitational wave (x-lengths up where y-lengths are down, both functions of z) is expressible on the axis edges; the
"x" pattern needs the face diagonals' lengths, a second field decided by her (brief, decision (3)). Nothing here is about
the real universe.
"""
import numpy as np

from graphity.lengths import STIFFNESS

M_PINNED = 4.0     # the inertia her one c fixes: long-wave speed 2 / sqrt(m) = 1 link per sweep
DT = 0.05          # leapfrog step in sweeps; stability needs dt < 2 / sqrt(48 / m) = 0.577 at m = 4


def stiffness_apply(delta, stiffness=STIFFNESS):
    """K delta with K = stiffness (6 I - A) on each family separately. delta has shape (3, L0, L1, L2)."""
    out = 6.0 * delta
    for axis in (1, 2, 3):
        out -= np.roll(delta, 1, axis=axis) + np.roll(delta, -1, axis=axis)
    return stiffness * out


def dense_stiffness(dims, stiffness=STIFFNESS):
    """One family's K as a dense matrix, built point by point from the lattice (the independent check of
    stiffness_apply; small dims only)."""
    dims = tuple(int(x) for x in dims)
    n = int(np.prod(dims))
    K = np.zeros((n, n))
    for i in range(n):
        p = np.array(np.unravel_index(i, dims))
        K[i, i] = 6.0 * stiffness
        for axis in range(3):
            for s in (-1, 1):
                q = p.copy()
                q[axis] = (q[axis] + s) % dims[axis]
                K[i, np.ravel_multi_index(tuple(q), dims)] -= stiffness
    return K


def omega(k, m=M_PINNED, stiffness=STIFFNESS):
    """Exact angular frequency of the plane wave with wavevector k = (k0, k1, k2) in radians per link."""
    k = np.asarray(k, dtype=float)
    return np.sqrt((stiffness / m) * np.sum(2.0 * (1.0 - np.cos(k))))


def long_wave_speed(m=M_PINNED, stiffness=STIFFNESS):
    """omega / |k| as k -> 0, in links per sweep: 2 / sqrt(m) at the brief's stiffness 4."""
    return np.sqrt(stiffness / m)


def wavevector(dims, j):
    """k = 2 pi j / L componentwise for integer mode numbers j = (j0, j1, j2)."""
    return 2.0 * np.pi * np.asarray(j, dtype=float) / np.asarray(dims, dtype=float)


def plane_wave(dims, family, j, amplitude=1e-3, travelling=False, m=M_PINNED, stiffness=STIFFNESS):
    """Initial (delta, vel) for one family: a standing cosine (vel = 0) or a wave travelling in the +k direction,
    delta = A cos(k.x), vel = A omega sin(k.x), so that delta(t) = A cos(k.x - omega t)."""
    dims = tuple(int(x) for x in dims)
    grids = np.meshgrid(*[np.arange(L) for L in dims], indexing="ij")
    k = wavevector(dims, j)
    phase = sum(k[i] * grids[i] for i in range(3))
    delta = np.zeros((3,) + dims)
    vel = np.zeros((3,) + dims)
    delta[family] = amplitude * np.cos(phase)
    if travelling:
        vel[family] = amplitude * omega(k, m, stiffness) * np.sin(phase)
    return delta, vel


def energy(delta, vel, m=M_PINNED, stiffness=STIFFNESS):
    """(1/2) m |v|^2 + (1/2) delta^T K delta, summed over the three families."""
    return 0.5 * m * np.sum(vel * vel) + 0.5 * np.sum(delta * stiffness_apply(delta, stiffness))


def step(delta, vel, dt=DT, m=M_PINNED, stiffness=STIFFNESS):
    """One leapfrog step (kick, drift, kick) of length dt sweeps; returns new (delta, vel), inputs untouched."""
    acc = -stiffness_apply(delta, stiffness) / m
    vel = vel + 0.5 * dt * acc
    delta = delta + dt * vel
    acc = -stiffness_apply(delta, stiffness) / m
    vel = vel + 0.5 * dt * acc
    return delta, vel


def run(delta, vel, sweeps, dt=DT, m=M_PINNED, stiffness=STIFFNESS, record_every=None):
    """Integrate for `sweeps` sweeps. Returns (delta, vel, record), record a list of (t, energy) at every
    record_every-th step (None: start and end only)."""
    n_steps = int(round(sweeps / dt))
    record = [(0.0, energy(delta, vel, m, stiffness))]
    for i in range(1, n_steps + 1):
        delta, vel = step(delta, vel, dt, m, stiffness)
        if record_every and i % record_every == 0:
            record.append((i * dt, energy(delta, vel, m, stiffness)))
    if not record_every:
        record.append((n_steps * dt, energy(delta, vel, m, stiffness)))
    return delta, vel, record


def random_state(dims, seed, amplitude=1e-3):
    """A reproducible random initial state (every family, zero mean per family so the free scaling mode is absent)."""
    rng = np.random.default_rng(seed)
    dims = tuple(int(x) for x in dims)
    delta = amplitude * rng.standard_normal((3,) + dims)
    vel = amplitude * rng.standard_normal((3,) + dims)
    delta -= delta.mean(axis=(1, 2, 3), keepdims=True)
    return delta, vel
