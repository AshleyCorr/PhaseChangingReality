"""T21's reading rules on known inputs (scripts/analyse_t21.py)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyse_t21", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t21.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def test_folded_share_ignores_undamaged_replicas():
    rows = [dict(folded="2", melted="6"), dict(folded="0", melted="0"), dict(folded="4", melted="4")]
    share, n = a.folded_share(rows)
    assert n == 2 and abs(share - (0.25 + 0.5) / 2) < 1e-12


def test_verdicts():
    assert a.verdict({1: (0.1, 0.2), 2: (0.2, 0.6)}) == "FOLDS WITH INTERCHANGEABLE POINTS"
    assert a.verdict({1: (0.1, 0.2), 2: (0.3, 0.4)}) == "MELTS EITHER WAY"
    assert a.verdict({1: (0.6, 0.7)}) == "FOLDS EITHER WAY"
    assert a.verdict({1: (None, 0.2)}).startswith("INCONCLUSIVE")
