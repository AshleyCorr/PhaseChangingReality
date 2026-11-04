"""T17's seed planting: each seed is move A, at the column asked for, costing exactly 32 - 16 lambda."""
import importlib.util
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus

SPEC = importlib.util.spec_from_file_location(
    "run_seeded_tube", Path(__file__).resolve().parents[1] / "scripts" / "run_seeded_tube.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_k_seeds_cost_k_times_the_cheapest_way_out_and_sit_where_asked():
    lam = 1.25
    for k in (1, 2, 4):
        adj, part = torus(16, 4, NO_CAP)
        h0 = hamiltonian(adj, lam)
        before = adj.copy()
        cols = m.plant_seeds(adj, part, 16, 4, k)
        assert cols == [j * 16 // k for j in range(k)]
        assert is_valid(adj, NO_CAP)
        assert abs(hamiltonian(adj, lam) - h0 - k * (32 - 16 * lam)) < 1e-9
        changed = {v for v in range(len(adj)) if set(adj[v]) != set(before[v])}
        for v in changed:                                   # every touched vertex is next to a seed column
            assert any(m.near(c, m.column(v, 4), 16) for c in cols)


def test_planting_is_deterministic():
    a, pa = torus(24, 4, NO_CAP)
    b, pb = torus(24, 4, NO_CAP)
    m.plant_seeds(a, pa, 24, 4, 3)
    m.plant_seeds(b, pb, 24, 4, 3)
    assert np.array_equal(a, b)
