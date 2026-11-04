"""A second curl above lambda = 1 (ASSUMPTIONS O110): it costs 64 (lambda - 1) to pinch a knot off a tube, the
knot's own switches all go downhill, and a lone knot runs down to sixteen tube-like points."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exact_knot_descent as knot  # noqa: E402


@pytest.fixture(scope="module")
def beside():
    return knot.beside_a_tube()


def test_a_knot_pinched_off_a_tube_costs_64_lambda_minus_one(beside):
    _kinds, (s0, x0), (sw, xw) = beside
    assert (sw, xw) == (80, 64) and (s0, x0) == (84, 80)
    for lam in (1.25, 1.30):
        assert knot.cost(s0 - sw, x0 - xw, lam) == pytest.approx(64 * (lam - 1))


def test_inside_the_knot_every_switch_is_downhill_above_one(beside):
    kinds, _, _ = beside
    inside = {(ds, dx) for (where, ds, dx, _p) in kinds if where == "knot"}
    assert inside == {(-2, -8)}                                      # cost 32 - 32 lambda
    assert knot.cost(-2, -8, 1.30) == pytest.approx(-9.6) and knot.cost(-2, -8, 0.9) > 0
    # the tube part keeps paper 1's two exits, at their prices
    tube = {(ds, dx) for (where, ds, dx, _p) in kinds if where == "tube"}
    assert (-2, -4) in tube and (-4, -10) in tube
    # joining the two pieces again costs energy at 1.30: 80 - 56 lambda at the least
    joins = [knot.cost(ds, dx, 1.30) for (where, ds, dx, pieces) in kinds if where == "join" and pieces == 1]
    assert min(joins) == pytest.approx(80 - 56 * 1.30)


def test_a_lone_knot_runs_down_to_tube_like_points():
    levels, resting = knot.downhill_from_a_lone_knot(1.30)
    assert sorted(levels, reverse=True) == [(24, 32, (16, 0, 0, 0)), (22, 24, (8, 8, 0, 0)),
                                            (21, 20, (4, 12, 0, 0)), (20, 16, (0, 16, 0, 0))]
    bottom = (20, 16, (0, 16, 0, 0))
    assert resting[bottom] == levels[bottom] and sum(resting.values()) == levels[bottom]
    energy = [16 * (16 - s) + 4 * 1.30 * x for (s, x, _h) in sorted(levels, reverse=True)]
    assert energy == pytest.approx([38.4, 28.8, 24.0, 19.2])        # 19.2 is sixteen points of plain tube
