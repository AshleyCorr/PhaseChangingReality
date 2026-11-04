"""T17's reading rules on known inputs (scripts/analyse_t17.py)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyse_t17", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t17.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def test_slope_recovers_known_lines():
    ks = [1, 1, 2, 2, 4, 4, 8, 8]
    b, _ = a.slope(ks, ks)
    assert abs(b - 1.0) < 1e-12
    b, _ = a.slope(ks, [1] * 8)
    assert abs(b) < 1e-12


def test_verdict_bands():
    assert a.verdict(1.0, True) == "ONE PER SEED"
    assert a.verdict(0.1, True) == "ONE PER TUBE"
    assert a.verdict(-0.2, True) == "ONE PER TUBE"
    assert a.verdict(0.5, True) == "BETWEEN"
    assert a.verdict(1.5, True).startswith("INCONCLUSIVE")
    assert a.verdict(1.0, False).startswith("INCONCLUSIVE")
