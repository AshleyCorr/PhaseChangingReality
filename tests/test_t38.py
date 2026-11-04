"""T38's analysis, tested before any run (PREREGISTRATION T38)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t38 as a  # noqa: E402


def test_one_exponential_has_no_tail_and_no_mixture():
    rng = np.random.default_rng(1)
    w = rng.exponential(700.0, 4000)
    tau, k10 = a.tail_count(w)
    assert abs(tau / 700 - 1) < 0.1 and k10 <= 1
    assert a.fit_mixture(w)[3] < 12


def test_a_hidden_second_population_is_found():
    rng = np.random.default_rng(2)
    w = np.concatenate([rng.exponential(700.0, 3960), rng.exponential(30000.0, 40)])
    tau, k10 = a.tail_count(w)
    assert k10 >= 3
    p, t1, t2, lr = a.fit_mixture(w)
    assert 0.98 < p < 0.999 and 15000 < t2 < 60000 and lr > 30


def test_verdict_rules():
    assert a.cell_reading(3) == "TAIL" and a.cell_reading(1) == "NO TAIL" and a.cell_reading(2) == "UNCLEAR"
    assert a.verdict(["TAIL", "TAIL", "NO TAIL", "NO TAIL"]) == "TWO POPULATIONS"
    assert a.verdict(["NO TAIL"] * 4) == "ONE POPULATION"
    assert a.verdict(["TAIL", "NO TAIL", "UNCLEAR", "NO TAIL"]) == "UNCLEAR"
