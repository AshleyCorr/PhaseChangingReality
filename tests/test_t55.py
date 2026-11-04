"""T55's reading (scripts/analyse_t55.py) on hand-built graphs: a scar made by one switch is ONE MOVE FROM FLAT and
the per-point energy sums to H; a flat sheet has no pieces; the classes and the majority rule."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, torus
from graphity.cqg_d import _switch, hamiltonian, is_valid
from graphity.dimension import local_dimension  # noqa: F401 (used below)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t55 = _load("analyse_t55")
LAM = 1.25


def _scarred_sheet():
    """A 12 x 8 flat sheet with one valid switch applied: the cheapest move out of flat space (two nearby links)."""
    adj, part = torus(12, 8, NO_CAP)
    base = hamiltonian(adj, LAM)
    for u1 in np.flatnonzero(part == 0)[:1]:
        u1 = int(u1)
        for v1 in adj[u1]:
            for u2 in np.flatnonzero(part == 0):
                u2 = int(u2)
                if u2 == u1:
                    continue
                for v2 in adj[u2]:
                    v2 = int(v2)
                    if v2 == v1 or v2 in adj[u1] or v1 in adj[u2]:
                        continue
                    t = adj.copy()
                    _switch(t, u1, int(v1), u2, v2)
                    if is_valid(t) and hamiltonian(t, LAM) - base <= 32.0 + 1e-9:
                        return t, part, hamiltonian(t, LAM)
    raise AssertionError("no cheapest switch found")


def test_a_flat_sheet_has_no_pieces():
    adj, part = torus(12, 8, NO_CAP)
    assert t55.read_graph(adj, part, LAM) == []


def test_a_scar_is_one_move_from_flat_and_its_energy_sums_to_h():
    adj, part, h = _scarred_sheet()
    rows = t55.read_graph(adj, part, LAM)
    assert rows, "the switch left no point off d = 2"
    assert all(r["move"] == "ONE MOVE FROM FLAT" for r in rows)
    assert all(t55.classify(r) == "SCAR" for r in rows)
    assert abs(sum(t55.point_energy(adj, v, LAM) for v in range(adj.shape[0])) - h) < 1e-9
    assert rows[0]["cheapest"] < 0 and rows[0]["far"] == 0 and rows[0]["way_round_max"] >= 3


def test_point_energy_is_zero_on_flat_points_and_sums_to_h_on_a_tube():
    tube, _ = torus(16, 4, NO_CAP)
    assert abs(sum(t55.point_energy(tube, v, LAM) for v in range(64)) - hamiltonian(tube, LAM)) < 1e-9
    flat, _ = torus(8, 8, NO_CAP)
    assert all(t55.point_energy(flat, v, LAM) == 0.0 for v in range(64))


def test_distance_to_counts_steps_and_minus_one_when_nothing_is_there():
    adj, _ = torus(8, 8, NO_CAP)
    a = int(adj[0, 0])
    two_away = next(int(w) for w in adj[a] if int(w) != 0 and int(w) not in adj[0])
    assert t55.distance_to(adj, (0,), {a}) == 1
    assert t55.distance_to(adj, (0,), {two_away}) == 2
    assert t55.distance_to(adj, (0,), {0}) == -1


def test_classes_and_the_majority_rule():
    assert t55.classify(dict(move="ONE MOVE FROM FLAT", far=3)) == "SCAR"
    assert t55.classify(dict(move="A DIP", far=1)) == "SEAM"
    assert t55.classify(dict(move="LOWERABLE", far=1)) == "SEAM"
    assert t55.classify(dict(move="A DIP", far=0)) == "KNOT"
    assert t55.classify(dict(move="LOWERABLE", far=0)) == "LOWERABLE"
    assert t55.verdict(["SEAM", "SEAM", "KNOT"]) == "SEAM"
    assert t55.verdict(["SEAM", "KNOT"]) == "MIXED"
    assert t55.verdict([]) == "NONE"


def test_the_config_is_the_preregistered_one():
    import json
    cfg = json.loads((ROOT / "configs" / "t55_pieces.json").read_text())
    assert cfg["name"] == "t55_pieces" and cfg["near"] == t55.NEAR and cfg["far"] == t55.FAR
    assert cfg["verdict_length"] == t55.VERDICT_L


def test_local_dimension_marks_the_scar():
    adj, _, _ = _scarred_sheet()
    assert int((local_dimension(adj) != 2).sum()) > 0
