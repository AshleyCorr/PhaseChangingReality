"""Sealed and leaky runs: energy must be conserved exactly, and the demon must read the right temperature."""
import numpy as np
import pytest

from graphity.cqg import NO_CAP, hamiltonian, is_valid, run_chain, torus
from graphity.sealed import demon_temperature, run_sealed

STEP = {0.0: 16.0, 0.5: 8.0, 1.0: 4.0, 1.25: 5.0}      # the smallest possible change of H at each lambda


def melted(side=8, seed=3):
    adj, part = torus(side, side, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 0.0, 200, 1, seed, 1.0, NO_CAP, False)
    return adj, side_u


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 1.25])
def test_sealed_conserves_energy_exactly(lam):
    """H + demon never changes, to the last bit, and the demon never goes negative."""
    adj, side_u = melted()
    total = hamiltonian(adj, lam) + 40.0
    s, x, demon, lost, acc = run_sealed(adj, side_u, 40.0, 300, 11, lam, NO_CAP, 0.0)
    assert is_valid(adj, NO_CAP) and acc > 0
    h = 16 * (64 - s) + 4 * lam * x
    assert np.all(h + demon == total)
    assert np.all(demon >= 0) and lost[-1] == 0


def test_leak_takes_energy_out_and_the_books_still_balance():
    adj, side_u = melted()
    total = hamiltonian(adj, 1.0) + 60.0
    s, x, demon, lost, _ = run_sealed(adj, side_u, 60.0, 300, 12, 1.0, NO_CAP, 0.05)
    h = 16 * (64 - s) + 4 * x
    assert np.allclose(h + demon + lost, total)
    assert lost[-1] > 0 and np.all(np.diff(lost) >= 0)            # energy only ever leaves
    assert demon[-1] < demon[:20].max()                            # and the demon is drained by it


def test_the_demon_reads_the_temperature_it_was_prepared_at():
    """Bring a graph to equilibrium at coupling g the ordinary way, then seal it with the demon holding what it
    would hold at that temperature. The demon's own energy must then read g back. (A wider scan, g = 5 to 20 at
    N = 144, is in ASSUMPTIONS section D; this is the cheap version.)"""
    g, step = 5.0, STEP[1.0]
    start = round(step / (np.exp(step / g) - 1) / step) * step
    reads = []
    for seed in range(5):
        adj, part = torus(8, 8, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        run_chain(adj, side_u, 1.0 / g, 1500, 1, 40 + seed, 1.0, NO_CAP, False)
        _, _, demon, _, _ = run_sealed(adj, side_u, start, 4000, 60 + seed, 1.0, NO_CAP, 0.0)
        reads.append(demon_temperature(demon[1000:], step))
    mean, err = np.mean(reads), np.std(reads, ddof=1) / np.sqrt(len(reads))
    assert abs(mean - g) < 4 * err, (mean, err, g)


def test_a_sealed_tube_with_nothing_to_spare_cannot_convert():
    """lambda = 1.25: the tube lies 1 per vertex above the flat sheet and has a wall round it. Sealed with an empty
    demon there is nothing to pay the wall with, so it stays a tube however long it runs, even though the state it
    would convert to is lower. The moves that do happen leave S and X alone."""
    adj, part = torus(16, 4, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    s, x, demon, _, acc = run_sealed(adj, side_u, 0.0, 4000, 501, 1.25, NO_CAP, 0.0)
    assert acc > 0                                                 # it is moving, not stuck
    assert np.all(s == 80) and np.all(x == 64) and np.all(demon == 0)
    assert hamiltonian(adj, 1.25) == 64                            # 1 per vertex, still the tube


def test_same_seed_same_run():
    runs = []
    for seed in (7, 7, 8):
        adj, side_u = melted()
        out = run_sealed(adj, side_u, 20.0, 200, seed, 1.0, NO_CAP, 0.02)
        runs.append(([int(v) for v in out[0]], [float(v) for v in out[2]], out[4], adj.copy()))
    assert runs[0][:3] == runs[1][:3] and (runs[0][3] == runs[1][3]).all()
    assert runs[0][:3] != runs[2][:3]
