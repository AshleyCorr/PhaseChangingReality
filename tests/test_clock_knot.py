"""The planted knot of scripts/run_clock_knot.py (T101 Amendment 1): valid, one switch from flat, near the origin."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus
from graphity.dimension import local_dimension
from run_clock_knot import distance_to, plant_knot


def test_planted_knot_is_valid_and_local():
    adj, part = torus(16, 16, NO_CAP)
    quad = plant_knot(adj, part, np.random.default_rng(3))
    assert is_valid(adj, NO_CAP)
    assert hamiltonian(adj, 1.0) > 0                       # not flat any more
    knot = np.flatnonzero(local_dimension(adj) != 2)
    assert 0 < len(knot) <= 16
    assert set(int(x) for x in quad) <= set(range(256))
    dist = distance_to(adj, knot)
    assert dist.min() == 0 and dist.max() >= 4             # the FAR band exists on a 16 x 16 torus
    assert distance_to(adj, [0]).min() == 0
