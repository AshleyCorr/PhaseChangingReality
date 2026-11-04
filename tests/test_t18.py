"""T18's reading rules on known inputs (scripts/analyse_t18.py)."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "analyse_t18", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t18.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def test_c_star_is_the_smallest_majority_sheet():
    by_c = {6: ["melted"] * 12 + ["sheet"] * 8, 12: ["sheet"] * 11 + ["other"] * 9, 24: ["sheet"] * 20}
    assert a.c_star(by_c) == 12
    assert a.c_star({6: ["melted"] * 20}) is None


def test_reading_bands():
    assert a.reading(4.0) == "PROPORTIONAL"
    assert a.reading(2.0) == "WEAKER THAN PROPORTIONAL"
    assert a.reading(1.0) == "FLAT"
    assert a.reading(None).startswith("NOT READ")
