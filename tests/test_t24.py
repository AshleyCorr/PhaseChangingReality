"""T24's reading rules (scripts/analyse_t24.py): gate 3' from the saved wiring, the exact classification,
and that the sharpness reading is T23's untouched."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t24", SCRIPTS / "analyse_t24.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)

from graphity.cqg import NO_CAP, torus                      # noqa: E402
from graphity.dimension import local_dimension              # noqa: E402


def row(rep, n=64, lam=1.25, waiting=800, released=None):
    d1 = int(round(0.99 * n / 2)); d2 = int(round(0.99 * n)) - d1
    return dict(N=str(n), replica=str(rep), lam=str(lam), f_200="0.0", waiting=str(waiting), reached="0.98",
                released=str(4 * (lam - 1) if released is None else released), d1_50=str(d1), d2_50=str(d2),
                largest_50="0.95", pieces_25="1")


def test_exact_classification_of_flat_ledge_and_other():
    lam = 1.25
    flat, part = torus(16, 4, NO_CAP)
    flat, _ = torus(16, 16, NO_CAP)                  # a flat torus: h = 0
    assert a.classify_exact(a.exact_energy(flat, lam), flat.shape[0], lam) == "FLAT"
    assert a.classify_exact(24 * lam - 16, 64, lam) == "LEDGE"
    assert a.classify_exact(24 * lam - 16 + 8, 64, lam) == "OTHER"
    tube, _ = torus(16, 4, NO_CAP)                   # the curled torus itself: 4(lam - 1) per vertex above flat
    assert abs(a.exact_energy(tube, lam) - 4 * (lam - 1) * 64) < 1e-9


def test_gate3_prime_needs_a_valid_saved_graph_for_every_decay_that_reached(tmp_path):
    lam = 1.25
    rows = [row(i) for i in range(3)]
    for i in range(2):
        adj, part = torus(16, 16, NO_CAP)
        np.savez(tmp_path / ("N64_rep%d.npz" % i), adj=adj, part=part, lam=lam, g=1.5, h0=64.0)
    states = a.read_final_states(rows, tmp_path, lam)
    assert set(states) == {"0", "1"} and all(s["kind"] == "FLAT" for s in states.values())
    assert not a.gate3_prime(rows, states)          # replica 2 has no saved graph
    adj, part = torus(16, 16, NO_CAP)
    np.savez(tmp_path / "N64_rep2.npz", adj=adj, part=part, lam=lam, g=1.5, h0=64.0)
    states = a.read_final_states(rows, tmp_path, lam)
    assert a.gate3_prime(rows, states)
    assert all(abs(s["release_exact"] - 0.25) < 1e-9 for s in states.values())   # (64 - 0) / 256 per vertex


def test_cell_reports_the_exact_census_and_keeps_t23s_sharpness(tmp_path):
    lam = 1.25
    import math
    q = [(i + 0.5) / 120 for i in range(120)]
    waits = [200 - 845 * math.log(1 - p) for p in q]
    rows = [row(i, waiting=w) for i, w in enumerate(waits)]
    states = {}
    for i in range(120):
        kind = "FLAT" if i % 3 else "OTHER"
        states[str(i)] = dict(valid=True, h=0.0 if kind == "FLAT" else 8.0, kind=kind,
                              release_exact=1.0 if kind == "FLAT" else 1.0 - 8.0 / 64, census=(0, 4, 60, 0, 0, 0, 0))
    c = a.cell(rows, 64, lam, states)
    assert c["gate3"] and c["a"] and c["sharp"]
    assert c["flat"] == 80 and c["other"] == 40 and c["ledge"] == 0
    assert c["other_census"] == [(0, 4, 60, 0, 0, 0, 0)]
    assert abs(c["release_exact"] - (80 * 1.0 + 40 * (1.0 - 0.125)) / 120) < 1e-9
    assert abs(c["thermal_excess"] - (c["release_exact"] - 1.0)) < 1e-9


def test_an_unrecognised_resting_state_no_longer_fails_the_cell():
    """The whole point of gate 3': OTHER states are reported, not gated."""
    import math
    lam = 1.30
    q = [(i + 0.5) / 120 for i in range(120)]
    waits = [200 - 419 * math.log(1 - p) for p in q]
    rows = [row(i, lam=lam, waiting=w, released=4 * 0.3) for i, w in enumerate(waits)]
    states = {str(i): dict(valid=True, h=(37.0 if i < 5 else 0.0), kind=("OTHER" if i < 5 else "FLAT"),
                           release_exact=1.2 - (37.0 / 64 if i < 5 else 0.0), census=(0, 3, 58, 3, 0, 0, 0))
              for i in range(120)}
    c = a.cell(rows, 64, lam, states)
    assert c["gate3"] and c["other"] == 5 and c["sharp"]
