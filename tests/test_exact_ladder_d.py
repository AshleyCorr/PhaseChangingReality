"""The ladder behind the curling-ladder chart (27 Sep): walls and energies on small rungs match the recorded exact values."""
import importlib.util
from pathlib import Path

from graphity.cqg_d import torus
from graphity.sealed_d import energy_d

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exact_walls_d", ROOT / "scripts" / "exact_walls_d.py")
ew = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ew)


def cheapest(sides, lam):
    adj, part = torus(sides)
    kinds = [k for k in ew.walls(adj, part, 1.0, top=10 ** 6, near=4) if (k[1], k[2]) != (0, 0)]
    return min(-16 * ds + 4 * lam * dx for _, ds, dx, _ in kinds)


def test_walls_match_the_record():
    assert cheapest([4, 48], 1.25) == 12.0                      # paper 1's tube
    assert cheapest([4, 48], 1.40) == 8.0                       # another kind of move is cheapest at 1.40
    assert abs(cheapest([4, 4, 4], 1.10) - 8.0) < 1e-9          # the 6-cube, 96 - 80 lambda (O41)
    assert abs(cheapest([4, 4, 18], 1.40) - 4.8) < 1e-9         # 128 - 88 lambda
    assert cheapest([6, 6, 6], 1.25) == 64.0                    # flat six-link space


def test_each_curled_direction_costs_four_lambda_minus_one_per_point():
    for sides in ([8, 8], [4, 48], [4, 4], [6, 6, 6], [4, 8, 12], [4, 4, 18], [4, 4, 4]):
        adj, _ = torus(sides)
        c, n = sum(1 for s in sides if s == 4), adj.shape[0]
        for lam in (1.0, 1.25, 1.7):
            assert abs(energy_d(adj, lam) / n - 4 * c * (lam - 1)) < 1e-9
