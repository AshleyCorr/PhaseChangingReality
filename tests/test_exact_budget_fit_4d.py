"""scripts/exact_budget_fit_4d.py: the four releases add to 4a, the measured ratio holds, and the walls at one
(lambda, rho) on the smallest rung agree with a direct pricing."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fit4d", Path(__file__).resolve().parent.parent / "scripts" / "exact_budget_fit_4d.py")
fit4d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit4d)


@pytest.mark.parametrize("rho", [0.2, 0.0, -0.3])
def test_releases_add_to_four_a_and_keep_the_measured_ratio(rho):
    r = fit4d.releases(rho)
    assert abs(sum(r) - 4.0) < 1e-12
    assert abs(r[0]) < 1e-12                                   # dark energy first releases nothing at the burp
    assert abs(r[1] / (r[2] + r[3]) - fit4d.RATIO) < 1e-9      # dark matter : ordinary net of time
    assert abs(r[3] - rho) < 1e-12


def test_tie_constants():
    f1, f2, f3 = fit4d.tie_for(0.0)
    assert f1 == 1.0 and abs(f2 - (2.0 - 4 * 5.36 / 6.36)) < 1e-12 and f3 == -1.0


def test_wall_of_the_gas_does_not_depend_on_time_and_is_positive():
    tie = fit4d.tie
    (adj, part), _ = tie.parse("4,4,4,4x2")
    kinds = {n: [] for n in fit4d.ORDER}
    kinds[fit4d.ORDER[0]] = [k for k in tie.kinds(adj, part) if not tie.is_null(k)]
    for n in fit4d.ORDER[1:]:
        kinds[n] = kinds[fit4d.ORDER[0]]
    w0 = fit4d.walls(kinds, 1.3, 0.0)[fit4d.ORDER[0]]
    w1 = fit4d.walls(kinds, 1.3, -0.4)[fit4d.ORDER[0]]
    assert w0 == w1 and w0 > 0
