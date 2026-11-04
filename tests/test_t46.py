"""T46's analyzer, before any run (PREREGISTRATION T46): four directions, FLAT at d = 4."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(spark, rep, final_d, h):
    base = dict(tie_label="C4", lam=1.3, N=100, C=200, spark=spark, replica=rep, total=spark)
    r0 = dict(base, sweep=0, h_per_vertex=4.8, **{"d%d" % k: (100 if k == 0 else 0) for k in range(8)})
    r1 = dict(base, sweep=250, h_per_vertex=h, **{"d%d" % k: (final_d[k] if k < len(final_d) else 0) for k in range(8)})
    return [r0, r1]


def test_verdicts_and_the_resting_rung():
    a = _load("analyse_t46")
    flat, two_open, three_open = [0, 0, 0, 5, 95], [0, 5, 95], [0, 0, 5, 95]
    rows = []
    for rep in range(3):
        rows += _rows(17.61, rep, two_open, 2.4) + _rows(173.91, rep, flat, 0.05)
    s = a.summarize(rows)
    assert s["verdict"][("C4", 1.3)] == "PUSHED THROUGH"
    assert [r for _, r, _ in s["ledgers"][("C4", 1.3, 200, 17.61)]] == [2, 2, 2]
    rows = []
    for rep in range(3):
        rows += _rows(17.61, rep, three_open, 1.2)                # three open is not flat in four directions
    s = a.summarize(rows)
    assert s["verdict"][("C4", 1.3)] == "NOT ALL"
    assert s["majority"][("C4", 1.3, 200, 17.61)] == "PARTLY OPEN"
    rows = []
    for rep in range(3):
        rows += _rows(17.61, rep, flat, 0.05)
    assert a.summarize(rows)["verdict"][("C4", 1.3)] == "ONE PUSH OPENS ALL"
