"""Gate C' reading rules (scripts/analyse_gatec2.py) on curves whose answers are known."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_gatec2", SCRIPTS / "analyse_gatec2.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def curve(x9, x6, x2, plateau=11.5, floor=1.1, n=90):
    """Squares per vertex against x (the published axis), piecewise linear through the three crossings."""
    x = np.linspace(-1.0, 2.2, n)
    knots = [(-1.0, plateau), (x9 - 0.3, plateau), (x9, 9.0), (x6, 6.0), (x2, 2.0), (2.2, floor)]
    kx, ky = zip(*knots)
    return x, np.interp(x, kx, ky)


def test_crossings_are_found_by_interpolation():
    x, y = curve(0.666, 0.996, 1.311)
    assert abs(a.crossing(x, y, 9.0) - 0.666) < 0.02
    assert abs(a.crossing(x, y, 6.0) - 0.996) < 0.02
    assert math.isnan(a.crossing(x, y, 20.0))


def test_reading_iv_passes_a_curve_that_matches_under_it_and_fails_under_i():
    x, y = curve(0.666, 0.996, 1.311)
    g = np.exp(2 * x)                                  # under (iv), x = (1/2) ln g
    s = a.score_a(g, y, "(iv)")
    assert s["passes"] and s["a1"] and s["a2"] and s["a3"]
    t = a.score_a(g, y, "(i)")                         # under (i) the crossings sit at 2x: far off
    assert not t["passes"]
    assert a.verdict_a([s]) == "READING (iv) HOLDS"
    assert a.verdict_a([t]) == "FAILS"
    assert a.verdict_a([]) == "NOT READ"


def test_width_tolerance_is_thirty_percent_of_the_published_width():
    x, y = curve(0.666, 0.996 + 0.09, 1.311)           # width 0.42 against 0.33: 27 % over, allowed; crossing 6 is 0.09 off
    s = a.score_a(np.exp(2 * x), y, "(iv)")
    assert s["a3"] and s["a2"] and s["passes"]
    x, y = curve(0.666, 0.996 + 0.12, 1.311)           # crossing of 6 is 0.12 off: A2 fails
    s = a.score_a(np.exp(2 * x), y, "(iv)")
    assert not s["a2"] and not s["passes"]


def test_run_b_width_and_verdicts():
    lng = np.linspace(-3, 4, 200)
    narrow = np.interp(lng, [-3, -1, 1, 4], [3.8, 3.1, 0.17, 0.05])     # width 2 in ln g
    wide = np.interp(lng, [-3, -1, 3, 4], [3.8, 3.1, 0.17, 0.05])       # width 4
    g = np.exp(lng)
    assert abs(a.width_b(g, narrow) - 2.0) < 0.05 and abs(a.width_b(g, wide) - 4.0) < 0.05
    assert a.verdict_b([2.0, 2.4]) == "ONE POWER"
    assert a.verdict_b([3.6, 4.1]) == "TWO POWERS"
    assert a.verdict_b([2.0, 4.1]) == "MIXED"
    assert a.verdict_b([math.nan]) == "NOT READ"


def test_legs_average_replicate_rows_per_coupling():
    rows = [dict(leg="cool", g="2.0", squares_per_vertex="8"), dict(leg="cool", g="2.0", squares_per_vertex="10"),
            dict(leg="cool", g="1.0", squares_per_vertex="11"), dict(leg="heat", g="1.0", squares_per_vertex="11.5")]
    out = a.legs(rows)
    assert list(out["cool"][0]) == [1.0, 2.0] and list(out["cool"][1]) == [11.0, 9.0]
    assert list(out["heat"][1]) == [11.5]
