"""The retrospective reading of the decay rows (ASSUMPTIONS O109): its classification and its counts."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import read_wait_detector as d  # noqa: E402


def row(waiting, f_200=0.0, sweep_25=""):
    return dict(waiting=("" if waiting is None else str(waiting)), f_200=str(f_200), sweep_25=str(sweep_25),
                sweep_50="", sweep_75="", replica="0")


def test_the_counted_wait_is_paper_ones():
    assert abs(d.tau_count(1.25) - 845.1) < 0.1          # paper 1, Eq. (2), at g = 1.5
    assert abs(d.tau_count(1.30) - 419.0) < 0.5


def test_classify():
    assert d.classify(row(None, 0.94, 205)) == "never"
    assert d.classify(row(15365, 0.75, 205)) == "begun"
    assert d.classify(row(900, 0.0, 950)) == "perfect"


def test_poisson_tail():
    assert d.poisson_tail(0, 3.0) == 1.0
    assert abs(d.poisson_tail(1, 0.5) - (1 - math.exp(-0.5))) < 1e-12


def test_a_fast_share_drags_the_preregistered_yardstick_down_and_makes_a_tail_of_one_population():
    """One memoryless population of mean 419, read through a 200-sweep watch: tubes whose first exit comes inside
    the watch are detected at once. The pre-registered tau_hat then falls well below 419, and waits beyond
    10 tau_hat appear in a single population. Measured against the perfect tubes' own scale they do not."""
    rng = np.random.default_rng(7)
    first_exit = rng.exponential(419.0, 4000)
    rows = []
    for t in first_exit:
        if t < d.REST:                                   # left the tube inside the watch: detected at the first check
            rows.append(row(d.REST + 5, 0.5, d.REST + 5))
        else:
            rows.append(row(int(t) + 1, 0.0, int(t) + 40))
    c = d.read_cell(rows, 1.30)
    assert c["never"] == 0 and c["begun"] == c["fast_begun"]
    assert abs(c["begun"] / c["n"] - c["begun_expected"]) < 0.03
    assert c["tau_hat"] < 0.75 * 419.0                   # the yardstick is biased short
    assert abs(c["tau_perfect"] / 419.0 - 1) < 0.08      # the perfect tubes keep the true scale
    assert c["long"] >= 3                                # "TAIL" by the pre-registered count, from one population
    assert c["long_begun"] == 0 and abs(c["long_perfect"] - c["expected_long_perfect"]) < 4 * math.sqrt(c["expected_long_perfect"] + 1)
    assert c["long_own"] <= 1


def test_tubes_that_changed_inside_the_watch_are_told_apart_from_tubes_that_waited():
    rows = [row(300 + 2 * i, 0.0, 340 + 2 * i) for i in range(200)]     # a bulk with tau_hat near 430
    rows += [row(None, 0.94, 205)] * 3                   # converted inside the watch, never detected
    rows += [row(20270, 0.75, 205)]                      # converted inside the watch, detected 20,000 sweeps later
    c = d.read_cell(rows, 1.25)
    assert c["never"] == 3 and c["never_min_f200"] == 0.94 and c["never_quarter_at_first_check"] == 3
    assert c["long"] == 1 and c["long_begun"] == 1 and c["long_perfect"] == 0
