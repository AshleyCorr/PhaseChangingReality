"""T23's reading rules on rows whose answers are known (scripts/analyse_t23.py)."""
import importlib.util
import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t23", SCRIPTS / "analyse_t23.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def row(rep, n=64, lam=1.25, waiting=800, f200=0.0, released=None, pieces25=1):
    d1 = int(round(0.99 * n / 2)); d2 = int(round(0.99 * n)) - d1
    return dict(N=str(n), replica=str(rep), lam=str(lam), f_200=str(f200), waiting=str(waiting), reached="0.98",
                released=str(4 * (lam - 1) if released is None else released), d1_50=str(d1), d2_50=str(d2),
                largest_50="0.95", pieces_25=str(pieces25))


def test_the_band_is_the_preregistered_one():
    lo, hi = a.cv_band(120)
    assert abs(lo - 0.756) < 0.01 and abs(hi - 1.379) < 0.01
    lo30, hi30 = a.cv_band(30)
    assert lo30 < lo < 1.0 < hi < hi30


def test_memoryless_waits_after_the_rest_pass_and_regular_ones_fail():
    q = [(i + 0.5) / 120 for i in range(120)]
    exp_waits = [200 - 845 * math.log(1 - p) for p in q]           # exponential after the rest: CV near 1
    c = a.cell([row(i, waiting=w) for i, w in enumerate(exp_waits)], 64, 1.25)
    assert c["n_read"] == 120 and c["a"] and c["sharp"]
    even = [200 + 400 + 10 * (i % 5) for i in range(120)]           # nearly constant: CV far below the band
    c = a.cell([row(i, waiting=w) for i, w in enumerate(even)], 64, 1.25)
    assert not c["a"] and not c["sharp"]


def test_the_rest_is_subtracted_before_the_cv():
    # Short exponential waits with the 200-sweep rest left in would have CV well below 1 (T8's problem).
    q = [(i + 0.5) / 120 for i in range(120)]
    waits = [200 - 150 * math.log(1 - p) for p in q]
    c = a.cell([row(i, lam=1.35, waiting=w, released=4 * 0.35) for i, w in enumerate(waits)], 64, 1.35)
    assert c["a"]


def test_window_verdicts():
    all_sharp = {l: "sharp" for l in a.WINDOW}
    assert a.window_verdict(all_sharp) == "SHARP ACROSS THE WINDOW"
    one = {**all_sharp, 1.10: "not sharp"}
    assert a.window_verdict(one) == "NOT SHARP AT 1.10"
    unread = {**all_sharp, 1.05: "unread"}
    assert a.window_verdict(unread).startswith("INCONCLUSIVE")
    assert a.window_verdict({1.05: "sharp"}).startswith("INCONCLUSIVE (no data")


def test_edge_report():
    below = [(row(i, pieces25=1), i < 95) for i in range(120)]      # 79 % flat, one piece
    edge = [(row(i, pieces25=2 if i < 40 else 1), i < 55) for i in range(120)]   # 46 % flat, a third in pieces
    e = a.edge_report(below, edge)
    assert e["e1"] and e["e2"] and e["verdict"] == "BREAK-UP BEGINS AT THE EDGE"
    same = a.edge_report(below, below)
    assert not same["e1"] and not same["e2"] and same["verdict"] == "NO BREAK-UP"
