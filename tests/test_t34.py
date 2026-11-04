"""T34 reading rules (scripts/analyse_t34.py) on rows whose answers are known."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t34", SCRIPTS / "analyse_t34.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)

N = 216


def block(sweep, folded, melted, rep=0, e=260.0, local="False", n=N):
    row = dict(N=str(n), spark=str(e), replica=str(rep), sweep=str(sweep), local_heat=local, melted=str(melted))
    for k in range(8):
        row["d%d" % k] = "0"
    row["d2"] = str(folded)                 # all folded points at d = 2 for the test's purposes
    row["d3"] = str(n - folded - melted)
    row["d4"] = str(melted)
    return row


def path(seq, **kw):
    return [block(500 * i, f, m, **kw) for i, (f, m) in enumerate(seq)]


def test_outcomes_and_reach():
    assert a.outcome(path([(0, 0), (8, 2)])) == "FOLDED"
    assert a.outcome(path([(0, 0), (2, 8)])) == "MELTED"
    assert a.outcome(path([(0, 0), (1, 2)])) == "HEALED"
    assert a.reach(path([(0, 0), (80, 0)]), N) == (True, False)
    assert a.reach(path([(0, 0), (150, 0)]), N) == (True, True)
    assert a.reach(path([(0, 0), (20, 0)]), N) == (False, False)
    assert a.melt_then_fold(path([(0, 0), (0, 10), (12, 2)]))
    assert not a.melt_then_fold(path([(0, 0), (12, 2), (0, 10)]))


def test_cells_split_packed_from_spread_and_verdicts():
    rows = []
    for rep in range(8):
        rows += path([(0, 0), (2, 20)], rep=rep)                                   # spread: melts
        rows += path([(0, 0), (60, 4), (150, 4)], rep=rep, local="True")           # packed: folds and cascades
    cm = a.cells(rows)
    assert cm[("spread", N, 260.0)]["majority"] == "MELTED"
    assert cm[("packed", N, 260.0)]["majority"] == "FOLDED"
    assert cm[("packed", N, 260.0)]["one_direction"] == 8 and cm[("packed", N, 260.0)]["cascade"] == 8
    assert a.verdict(cm) == "RE-CURLS"
    melts = a.cells([b for rep in range(8) for b in path([(0, 0), (2, 20)], rep=rep)])
    assert a.verdict(melts) == "MELTS"
    heals = a.cells([b for rep in range(8) for b in path([(0, 0), (0, 2)], rep=rep)])
    assert a.verdict(heals) == "HEALS"
    few = a.cells([b for rep in range(3) for b in path([(0, 0), (8, 2)], rep=rep)])
    assert a.verdict(few) == "NOT READ"
