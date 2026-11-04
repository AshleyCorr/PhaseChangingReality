"""T37's analysis, tested before any run (PREREGISTRATION T37)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyse_t37 as a  # noqa: E402
from graphity.cqg import NO_CAP, torus  # noqa: E402


def test_avrami_recovers_the_exponent_of_a_synthetic_curve():
    every, k, n = 100, 3e-9, 2.0
    t = every * np.arange(1, 400)
    x = 1 - np.exp(-k * t ** n)
    series = np.round(1000 * x / x[-1]).astype(int)
    fn, fk = a.avrami(series, every)
    assert abs(fn - 2.0) < 0.05 and abs(math.log(fk / k)) < 0.2
    assert abs(math.log(a.avrami_k_at_two(series, every) / k)) < 0.1


def test_kjma_prediction_is_the_integral_of_nucleation():
    """k = I L * integral of exp(-K t^2) dt; check against a direct sum."""
    length, t1, kk = 1000, 50.0, 1e-7
    i = 1.0 / (length * t1)
    t = np.arange(0, 20000, 0.5)
    direct = i * length * np.exp(-kk * t ** 2).sum() * 0.5
    assert abs(a.kjma_prediction(t1, length, kk) - direct) / direct < 1e-3


def test_leftovers_count_a_column_and_nothing_on_a_flat_torus():
    adj, _ = torus(12, 12, NO_CAP)
    assert a.leftovers(adj) == (0, 0)
    tube, _ = torus(4, 12, NO_CAP)                   # all points at d = 1: one piece, not a column
    assert a.leftovers(tube) == (0, 1)


def test_end_state_reads_energy_and_flatness():
    adj, _ = torus(12, 12, NO_CAP)
    assert a.end_state(adj, 1.25) == (0.0, 1.0, "CLEAN")
    tube, _ = torus(4, 12, NO_CAP)                   # one curled direction: 4(lambda - 1) per point above flat
    h, share, state = a.end_state(tube, 1.25)
    assert abs(h - 48.0) < 1e-9 and share == 0.0 and state == "DEFECTED"


def test_verdict_rules():
    assert a.verdict_scaling({256: 2.0, 512: 3.8, 1024: 7.5})[1] is True
    assert a.verdict_scaling({256: 2.0, 512: 2.1, 1024: 2.2})[1] is False
    assert a.verdict_scraps({64: 1.0, 1024: 3.5}) == "MANY SEEDS, MANY SCRAPS"
    assert a.verdict_scraps({64: 1.0, 1024: 1.2}) == "ONE SCRAP HOWEVER LARGE"
    assert a.verdict_scraps({64: 1.0, 1024: 2.0}) == "BETWEEN"
