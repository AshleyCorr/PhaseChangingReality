"""T33 pattern rules (scripts/analyse_t33.py) on rung sequences whose answers are known."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t33", SCRIPTS / "analyse_t33.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)

N = 2304


def block(sweep, rung, dims="4x(4 4 4 4)", melted=0, c=4608, rep=0):
    row = dict(dims=dims, lam="1.25", N=str(N), C=str(c), spark="20.0", replica=str(rep), sweep=str(sweep),
               melted=str(melted), left="True", final="False")
    for d in range(8):
        row["d%d" % d] = "0"
    if rung is None:
        row["d1"], row["d2"] = str(N // 2), str(N // 2)        # no majority
    else:
        row["d%d" % rung] = str(N)
    if melted:
        row["d5"] = str(melted)
    return row


def path(rungs_by_time, **kw):
    """rungs_by_time: [(sweep_from, rung)], held until the next entry, sampled every 500 sweeps to 60,000."""
    out = []
    for sw in range(0, 60001, 500):
        rung = None
        for start, r in rungs_by_time:
            if sw >= start:
                rung = r
        out.append(block(sw, rung, **kw))
    return out


def test_patterns_from_the_gas():
    assert a.pattern(path([(0, 0), (2000, 4)]), N) == "FOUR TOGETHER"
    assert a.pattern(path([(0, 0), (2000, 1), (12000, 4)]), N) == "SINGLETON PLUS THREE"
    assert a.pattern(path([(0, 0), (2000, 1), (12000, 2), (22000, 3), (32000, 4)]), N) == "ONE AT A TIME"
    assert a.pattern(path([(0, 0), (2000, 1), (12000, 2)]), N) == "ONE AT A TIME"       # as far as it got, rung by rung
    assert a.pattern(path([(0, 0)]), N) == "STALLS"
    assert a.pattern(path([(0, 0), (2000, 1), (3000, 2), (4000, 3)]), N) == "STALLS"   # brief visits, no rests, no flat
    assert a.pattern(path([(0, 0)], melted=N // 2), N) == "MELTED"
    assert a.pattern(path([(0, 0), (2000, 2), (12000, 4)]), N) == "OTHER"             # rests on 2 only


def test_patterns_from_three_curled():
    d = "4 4 4 36"
    assert a.pattern(path([(0, 1), (2000, 4)], dims=d), N) == "THREE TOGETHER"
    assert a.pattern(path([(0, 1), (2000, 2), (12000, 3), (22000, 4)], dims=d), N) == "ONE AT A TIME"
    assert a.pattern(path([(0, 1), (2000, 2), (3000, 4)], dims=d), N) == "THREE TOGETHER"   # a brief visit is not a rest


def test_cells_and_verdicts():
    rows = []
    for rep in range(4):
        rows += path([(0, 0), (2000, 1), (12000, 4)], rep=rep)
    for rep in range(4, 6):
        rows += path([(0, 0)], rep=rep)
    rows3 = []
    for rep in range(6):
        rows3 += path([(0, 1), (2000, 2), (12000, 3), (22000, 4)], dims="4 4 4 36", rep=rep)
    cm = a.cells(rows + rows3)
    assert cm[("4x(4 4 4 4)", 1.25, N, 4608)]["majority"] == "SINGLETON PLUS THREE"
    assert cm[("4 4 4 36", 1.25, N, 4608)]["majority"] == "ONE AT A TIME"
    assert a.verdict(cm, True) == "SINGLETON PLUS THREE"
    assert a.verdict(cm, False) == "ONE AT A TIME"
    stalled = a.cells([b for rep in range(6) for b in path([(0, 0)], rep=rep)])
    assert a.verdict(stalled, True) == "NO CASCADE"
    assert a.verdict(stalled, False) == "NOT READ"
