"""scripts/exact_relic_d.py: the relic switch is a valid bipartite switch, its energy is what O56 records, and two far
relics are additive in energy."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("exact_relic_d", SCRIPTS / "exact_relic_d.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

from graphity.cqg_d import is_valid, torus     # noqa: E402
from graphity.sealed_d import energy_d         # noqa: E402


def test_one_relic_is_valid_and_costs_what_o56_records():
    dims = [8, 8, 8]
    coords, look = m.index(dims)
    adj, part = torus(dims)
    base = energy_d(adj, 1.02)
    col = m.make_relic(adj, dims, look, (0, 0, 0))
    assert is_valid(adj)
    assert len(col) == 4 and all(part[col[0]] == part[c] or part[col[0]] != part[c] for c in col)
    assert abs(energy_d(adj, 1.02) - base - 120.48) < 1e-6


def test_two_far_relics_are_additive_and_touching_ones_are_not():
    dims = [8, 8, 8]
    coords, look = m.index(dims)
    lam = 1.02
    adj, part = torus(dims)
    base = energy_d(adj, lam)
    one = adj.copy(); m.make_relic(one, dims, look, (0, 0, 0))
    e1 = energy_d(one, lam) - base
    far = adj.copy(); m.make_relic(far, dims, look, (0, 0, 0)); m.make_relic(far, dims, look, (0, 4, 0))
    assert abs(energy_d(far, lam) - base - 2 * e1) < 1e-6
    near = adj.copy(); m.make_relic(near, dims, look, (0, 0, 0)); m.make_relic(near, dims, look, (0, 1, 0))
    assert energy_d(near, lam) - base < 2 * e1 - 1.0        # an attraction at contact
