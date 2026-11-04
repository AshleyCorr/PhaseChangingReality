"""The move census must count exactly the moves the chain can make (ASSUMPTIONS O20, correction)."""
import importlib.util
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, torus

SPEC = importlib.util.spec_from_file_location(
    "run_move_census", Path(__file__).resolve().parents[1] / "scripts" / "run_move_census.py")
census_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census_script)


def two_sided_with(adj, side_u):
    """True when every edge joins side 0 to side 1, with side 0 unchanged."""
    on_u = np.zeros(adj.shape[0], dtype=bool)
    on_u[side_u] = True
    return all(on_u[u] != on_u[v] for u in range(adj.shape[0]) for v in adj[u])


def test_every_counted_move_keeps_the_sides():
    for lx, ly in ((8, 8), (16, 4)):
        adj, part = torus(lx, ly, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        moves = list(census_script.chain_switches(adj, side_u))
        assert moves and all(two_sided_with(m, side_u) for m in moves)


def test_no_single_move_adds_a_square_to_perfect_order():
    for lx, ly in ((8, 8), (16, 4)):
        adj, part = torus(lx, ly, NO_CAP)
        row = census_script.census(adj, np.flatnonzero(part == 0), 1.25)
        assert row["adds"] == 0 and row["loses"] > 0


def test_the_cheapest_way_out_of_the_tube_costs_12_at_lambda_125():
    adj, part = torus(16, 4, NO_CAP)
    row = census_script.census(adj, np.flatnonzero(part == 0), 1.25)
    assert row["cheapest_lose"] == 12.0
