"""T22's reading rules on known inputs (scripts/analyse_t22.py)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyse_t22", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t22.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def test_tau_matches_the_paper():
    assert round(a.tau(1.05)) == 8330 and round(a.tau(1.25)) == 845


def test_flags():
    f, late, *_ = a.flags([1] * 20 + [2] * 20, [100] * 20 + [120] * 20, 110)
    assert f and not late
    f, late, *_ = a.flags([1] * 40, [300, 310] * 20, 100)
    assert not f and late


def test_verdict():
    assert a.verdict([(True, False), (True, False)]) == "FALL-BACKS"
    assert a.verdict([(True, False), (False, False)]) == "MIXED"
    assert a.verdict([(False, False), (False, False)]) == "NEITHER"


def test_first_exit_includes_replica_that_did_not_convert():
    rows = [dict(first_exit_sweeps="10", went_through="True"),
            dict(first_exit_sweeps="90", went_through="False"),
            dict(first_exit_sweeps="-1", went_through="False"),
            dict(first_exit_sweeps="", went_through="False")]
    assert a.observed_first_exits(rows).tolist() == [10.0, 90.0]
    assert a.observed_first_exits(rows).mean() == 50.0
