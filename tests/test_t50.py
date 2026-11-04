"""T50's runner helpers and analyzer, written before any run (PREREGISTRATION T50)."""
import importlib.util
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, torus

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_band_and_column_reading():
    run = _load("run_front_matter")
    right, mirror = run.band_columns(192, 48, 16)
    assert right == list(range(40, 56)) and mirror == [192 - c for c in range(40, 56)]
    adj, _ = torus(64, 4, NO_CAP)                       # the tube: d = 1 everywhere, nothing open
    assert run.sides(run.opened_columns(adj, 64), 64) == (0, 0)
    opened = np.zeros(64, bool)
    opened[1:6] = True                                  # five columns open on the right
    opened[60:64] = True                                # four on the left
    opened[32] = True                                   # the column opposite the seed is on neither side
    assert run.sides(opened, 64) == (5, 4)


def _blocks(e, rep, right_times, left_times, centre=48, width=16):
    """Readings every 10 sweeps; the right front passes k_in and k_out at the given times, the left likewise."""
    rows = []
    for s in range(0, 2001, 10):
        def count(t_in, t_out):
            return 0 if s < t_in else (40 if s < t_out else 56)
        rows.append(dict(energy_per_point=e, replica=rep, sweep=s, band_centre=centre, band_width=width,
                         right=count(*right_times), left=count(*left_times), band_energy=64 * e))
    return rows


def test_verdicts():
    a = _load("analyse_t50")
    rows = []
    for rep in range(4):
        rows += _blocks(0.0, rep, (100, 300), (100, 300))           # control: R = 1
        rows += _blocks(3.0, rep, (100, 400), (100, 300))           # the band takes 300 against 200: R = 1.5
        rows += _blocks(6.0, rep, (100, 250), (100, 300))           # 150 against 200: R = 0.75
        rows += _blocks(1.0, rep, (100, 310), (100, 300))           # 210 against 200: no effect
    s = a.summarize(rows)
    assert s["table"][0.0]["median_R"] == 1.0
    assert s["verdict"] == {1.0: "NO EFFECT", 3.0: "SLOWS", 6.0: "SPEEDS"}
    rows = []
    for rep in range(4):
        rows += _blocks(0.0, rep, (100, 300), (100, 300))
        rows += _blocks(3.0, rep, (100, 5000), (100, 300))          # the right front never leaves the band
    assert a.summarize(rows)["verdict"] == {3.0: "NO FRONT"}
