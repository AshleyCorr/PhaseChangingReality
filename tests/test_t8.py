"""T8's reading rules on rows whose answers are known (scripts/analyse_t8.py)."""
import importlib.util
import math
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyse_t8", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t8.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def row(rep, n=64, lam=1.25, released=None, f200=0.0, waiting=800, reached=0.75, two=0.99, largest=0.95):
    d1 = int(round(two * n / 2)); d2 = int(round(two * n)) - d1
    return dict(N=str(n), replica=str(rep), lam=str(lam), f_200=str(f200), waiting=str(waiting),
                reached=str(reached), released=str(4 * (lam - 1) if released is None else released),
                d1_50=str(d1), d2_50=str(d2), largest_50=str(largest))


def test_tau_matches_the_preregistered_table():
    assert round(a.tau(1.05)) == 8330 and round(a.tau(1.25)) == 845 and round(a.tau(1.45)) == 22


def test_metastable_needs_most_decays_still_a_tube_at_sweep_200():
    assert a.metastable([row(i) for i in range(30)])
    assert not a.metastable([row(i, f200=0.9) for i in range(16)] + [row(i, f200=0.0) for i in range(14)])


def test_release_is_classified_at_the_sheet_the_ledge_or_neither():
    n, lam = 64, 1.25
    assert a.classify_release(row(0), n, lam) == "sheet"
    assert a.classify_release(row(0, released=1.0 - 14.0 / 64), n, lam) == "ledge"
    assert a.classify_release(row(0, released=0.5), n, lam) == "neither"
    assert a.classify_release(row(0, released=0.5), n, lam, read=lambda r: True) == "read"


def test_a_clean_cell_is_sharp_and_a_regular_one_is_not():
    waits = [200 + 1600 * (i + 0.5) / 30 for i in range(30)]       # spread wide: CV near 0.55
    rows = [row(i, waiting=w) for i, w in enumerate(waits)]
    c = a.cell(rows, 64, 1.25)
    assert c["gate2"] and c["gate3"] and c["b"] and c["c"]
    assert not c["a"]                                              # CV below 0.7 fails (a)
    exp_waits = [-845 * math.log(1 - (i + 0.5) / 30) for i in range(30)]   # exponential quantiles: CV near 1
    c = a.cell([row(i, waiting=w) for i, w in enumerate(exp_waits)], 64, 1.25)
    assert c["sharp"] and c["law"]


def test_verdicts():
    assert a.verdict({1.05: "sharp", 1.10: "sharp", 1.30: "sharp", 1.40: "not metastable"}) == "SHARP DOWN TO 1"
    assert a.verdict({1.05: "not sharp", 1.10: "sharp", 1.30: "sharp", 1.40: "not metastable"}).startswith("A FLOOR (lambda* = 1.10")
    assert a.verdict({1.05: "unread", 1.10: "sharp", 1.40: "not metastable"}) == "INCONCLUSIVE"
    assert a.verdict({1.05: "sharp", 1.10: "not metastable", 1.30: "sharp"}).startswith("INCONCLUSIVE")
