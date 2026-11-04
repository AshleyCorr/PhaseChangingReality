"""scripts/read_t9_squares.py: a sealed tube cannot melt (ASSUMPTIONS O123). The energy's floor on the squares per
vertex, checked against a chain run for the purpose and against T9's saved final states."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import read_t9_squares as r                                       # noqa: E402
from graphity import sealed                                       # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus               # noqa: E402


def test_the_floor_is_the_energy_spent_on_missing_squares():
    assert r.floor(64) == pytest.approx(1 - (64 + 12) / (16 * 64))           # one unit per vertex and the push
    assert r.floor(192) == pytest.approx(0.93359375)
    assert r.floor(10 ** 9) == pytest.approx(1 - 1 / 16, abs=1e-6)            # about fifteen squares in sixteen
    assert r.floor(96, lam=1.40, push=8.0) == pytest.approx(1 - (0.4 * 4 * 96 + 8) / (16 * 96))
    adj, _part = torus(16, 4, NO_CAP)                                        # the tube holds exactly that energy
    assert hamiltonian(adj, 1.25) == pytest.approx(4 * 0.25 * 64)


def test_a_sealed_tube_never_goes_below_the_floor():
    """One store, the hottest box T9 ran: the squares per vertex stay above the floor at every sweep."""
    adj, part = torus(16, 4, NO_CAP)
    side = np.flatnonzero(part == 0).astype(np.int64)
    demons = np.zeros(1, dtype=np.float64)
    demons[0] = 12.0
    s, _x, _mean, _sum, _acc = sealed.run_sealed_bath(adj, side, demons, 4000, 20261012, 1.25, NO_CAP, None, False)
    assert s.min() / 64 >= r.floor(64) - 1e-12
    assert hamiltonian(adj, 1.25) + demons.sum() == pytest.approx(64 + 12)   # the books balance


def test_every_run_t9_classed_melted_kept_its_squares():
    groups = r.by_class(str(ROOT / "results"))
    melted = [p for (n, kind), phi in groups.items() if kind == "melted" for p in phi]
    assert len(melted) == 154                                                 # as paper 1 reports
    assert min(melted) >= 0.93 and max(melted) <= 1.0
    for (n, _kind), phi in groups.items():
        assert min(phi) >= r.floor(n) - 1e-9                                  # no run is below the energy's floor
    assert min(melted) > 7 * r.RANDOM_PHASE                                   # nowhere near the random phase
