"""T92's integration and set-up (scripts/curled_x_free_energy.py), on cases with known answers."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import curled_x_free_energy as t  # noqa: E402


def test_equal_energies_leave_only_the_degeneracy_term():
    gs = [0.5, 1.0, 2.0]
    out = t.delta_f(gs, [0.0, 3.0, 9.0], [0.0, 3.0, 9.0], math.log(8))
    assert np.allclose(out, [-g * math.log(8) for g in gs])


def test_integration_matches_a_two_level_system_exactly_in_the_limit():
    # one basin with a single excited level at energy e and m copies; the other with none
    e, m = 2.0, 5
    gs = list(np.linspace(0.05, 3.0, 4000))
    h1 = [m * e * math.exp(-e / g) / (1 + m * math.exp(-e / g)) for g in gs]
    out = t.delta_f(gs, h1, [0.0] * len(gs), 0.0)
    exact = [-g * math.log(1 + m * math.exp(-e / g)) for g in gs]
    assert abs(out[-1] - exact[-1]) < 1e-3


def test_built_shapes_have_the_registered_energies_and_the_basin_flags():
    from graphity.cqg import NO_CAP, surplus, torus, total_squares
    for (lx, ly), sx in (((4, 36), (1.25, 1.0)), ((12, 12), (1.0, 0.0)), ((4, 48), (1.25, 1.0)), ((12, 16), (1.0, 0.0))):
        adj, _ = torus(lx, ly, NO_CAP)
        n = lx * ly
        s, x = total_squares(adj) / n, surplus(adj) / n
        assert (s, x) == sx
        assert t.energy(n, total_squares(adj), surplus(adj)) == 16 * (n - total_squares(adj)) + 4 * surplus(adj)


def test_curled_and_flat_tie_at_lambda_one():
    from graphity.cqg import NO_CAP, surplus, torus, total_squares
    for lx, ly in ((4, 36), (12, 12)):
        adj, _ = torus(lx, ly, NO_CAP)
        n = lx * ly
        assert t.energy(n, total_squares(adj), surplus(adj)) == 0.0
