"""T7 amendment 4 (a): reading a resting state from its wiring, checked on states whose answer is known.

The flat sheet holds no energy and has nothing off the sheet; the perfect tube holds exactly one
unit per point at lambda = 1.25 and is one piece at d = 1 throughout. Naming and the gate's
tolerance are checked directly.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphity.cqg import NO_CAP, torus  # noqa: E402

spec = importlib.util.spec_from_file_location("analyse_t7_states", ROOT / "scripts" / "analyse_t7_states.py")
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)


def test_flat_sheet_holds_nothing():
    adj, _ = torus(16, 16, NO_CAP)
    st = ana.describe(adj, 1.25)
    assert st["h"] == 0
    assert st["pieces"] == []
    assert st["hist"][2] == 256
    assert st["extra_squares"] == 0 and st["surplus"] == 0


def test_tube_holds_one_unit_per_point_and_is_one_piece():
    adj, _ = torus(16, 4, NO_CAP)
    st = ana.describe(adj, 1.25)
    assert abs(st["h"] - 64 * 4 * (1.25 - 1)) < 1e-9
    assert st["hist"][1] == 64
    assert len(st["pieces"]) == 1 and sum(st["pieces"][0]) == 64


def test_names_come_from_composition():
    assert ana.name_piece((0, 4, 0, 0, 0, 0, 0)) == "four-point remnant"
    assert ana.name_piece((0, 2, 0, 2, 0, 0, 0)) == "twist"
    assert ana.name_piece((0, 6, 0, 2, 0, 0, 0)).startswith("unnamed: 6 at d=1, 2 at d=3")


def test_gate_tolerance_is_one_percent_of_the_lump():
    n, lam = 100, 1.25
    h0 = n * 4 * (lam - 1)
    implied, ok = ana.gate(0.86, h0, h0 - 0.86 * n, n, lam)
    assert abs(implied - 0.86) < 1e-12 and ok
    _, ok = ana.gate(0.86, h0, h0 - 0.88 * n, n, lam)
    assert not ok


def test_pieces_are_found_with_their_members():
    adj, _ = torus(8, 8, NO_CAP)
    mask = np.zeros(64, dtype=bool)
    mask[[0, 1, 2]] = True          # vertices 0, 1, 2 of an 8 x 8 torus: 0-1-2 along a row
    pieces = ana.pieces_with_members(adj, mask)
    assert sum(len(p) for p in pieces) == 3
