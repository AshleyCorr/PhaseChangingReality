"""T9 gate 1 and the bit-for-bit reduction: a bath of C demons conserves energy exactly, and
with C = 1 it is run_sealed with no leak, move for move."""
import numpy as np

from graphity.cqg import NO_CAP, hamiltonian, torus
from graphity.sealed import run_sealed, run_sealed_bath


def test_one_demon_bath_is_run_sealed_exactly():
    lam = 1.25
    a1, part = torus(16, 4, NO_CAP); a2 = a1.copy()
    side_u = np.flatnonzero(part == 0)
    s1, x1, d1, lost, _ = run_sealed(a1, side_u, 12.0, 300, 77, lam, NO_CAP, 0.0)
    s2, x2, mean2, sum2, _ = run_sealed_bath(a2, side_u, np.array([12.0]), 300, 77, lam, NO_CAP)
    assert (s1 == s2).all() and (x1 == x2).all()
    assert np.allclose(d1, mean2) and np.allclose(d1, sum2)
    assert (a1 == a2).all()


def test_energy_is_conserved_to_the_last_unit_for_any_bath_size():
    lam = 1.25
    for c in (1, 4, 16, 64):
        adj, part = torus(16, 4, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        demons = np.zeros(c); demons[0] = 12.0
        e0 = hamiltonian(adj, lam) + demons.sum()
        s, x, mean, tot, _ = run_sealed_bath(adj, side_u, demons, 400, 5 + c, lam, NO_CAP)
        h = 16.0 * (adj.shape[0] - s) + 4.0 * lam * x
        assert np.allclose(h + tot, e0), (c, np.abs(h + tot - e0).max())
        assert abs(hamiltonian(adj, lam) + demons.sum() - e0) < 1e-9
        assert (demons >= -1e-12).all()


def test_a_bigger_bath_runs_cooler_for_the_same_energy():
    """The point of C: the same total energy shared among more demons is a lower temperature.
    Not by the naive factor -- the graph takes its share of the energy, and with one demon
    holding 40 units it can pay for any uphill move and simply melts the tube (measured: 33 of
    the 40 units went into the graph). Only the direction is asserted."""
    lam = 1.25
    temps = []
    for c in (2, 32):
        adj, part = torus(16, 4, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        demons = np.zeros(c); demons[0] = 40.0
        s, x, mean, tot, _ = run_sealed_bath(adj, side_u, demons, 3000, 11, lam, NO_CAP)
        temps.append(mean[-500:].mean())
    assert temps[1] < temps[0], temps
