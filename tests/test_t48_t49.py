"""T48's and T49's analyzers, written before any run (PREREGISTRATION T48, T49)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(dims, n, lam, spark, rep, sweep, d, largest, **extra):
    r = dict(dims=dims, N=n, lam=lam, spark=spark, replica=rep, sweep=sweep, largest_open_d=largest, **extra)
    r.update({"d%d" % k: (d[k] if k < len(d) else 0) for k in range(8)})
    return r


def test_t48_one_space_bigger_push_advances_and_damage():
    a = _load("analyse_t48")
    start = [0, 288, 0, 0]                                              # 4 x 4 x 18: every point at d = 1
    rows = []
    for rep in range(3):
        rows += [_row("4 4 18", 288, 1.25, 16.01, rep, 0, start, 0),
                 _row("4 4 18", 288, 1.25, 16.01, rep, 500, [0, 20, 200, 60], 40),
                 _row("4 4 18", 288, 1.25, 64.04, rep, 0, start, 0),
                 _row("4 4 18", 288, 1.25, 64.04, rep, 500, [0, 0, 20, 250, 18], 240)]
    s = a.summarize(rows)
    assert s["majority"][("4 4 18", 1.25, 16.01)] == "ADVANCES"
    assert s["majority"][("4 4 18", 1.25, 64.04)] == "OPENS"
    assert s["verdict"][("4 4 18", 1.25)] == "ONE SPACE WITH A BIGGER PUSH"
    # open everywhere but in pieces: not one space
    rows = [_row("4 4 18", 288, 1.40, 6.41, r, 500, [0, 0, 0, 288], 100) for r in range(3)]
    s = a.summarize(rows)
    assert s["majority"][("4 4 18", 1.40, 6.41)] == "ADVANCES" and s["verdict"][("4 4 18", 1.40)] == "ADVANCES ONLY"
    # four directions: damage is d >= 5; d = 4 is open
    rows = [_row("4 4 4 12", 768, 1.25, 20.01, r, 250, [0, 0, 0, 100, 468, 200], 460) for r in range(3)]
    s = a.summarize(rows)
    assert s["majority"][("4 4 4 12", 1.25, 20.01)] == "DAMAGED" and s["verdict"][("4 4 4 12", 1.25)] == "DAMAGED"
    rows = [_row("4 4 4 12", 768, 1.25, 20.01, r, 250, [0, 768], 0) for r in range(3)]      # one side open: d = 1
    assert a.summarize(rows)["verdict"][("4 4 4 12", 1.25)] == "STAYS"
    # the front's arrival is read from the largest open piece
    blocks = [_row("4 48", 192, 1.25, 12.01, 0, s_, [0, 192 - k, k], k) for s_, k in ((0, 0), (200, 30), (400, 120))]
    assert a.read_replica(blocks) == ("OPENS", 2, 200, 400)


def test_t49_ends_and_paths():
    a = _load("analyse_t49")

    def run(rep, seq, how="named", lam=1.1):
        extra = dict(interchangeable=True, weighted=True) if how == "interchangeable" else {}
        return [_row("1x(4 4 4)", 64, lam, 8.01, rep, 5 * i, d, 0, **extra) for i, d in enumerate(seq)]
    in_order = [[64], [20, 40, 4], [0, 10, 44, 10], [0, 0, 6, 58]]
    together = [[64], [40, 10, 4, 10], [0, 4, 6, 54]]
    stuck = [[64], [60, 4]]
    rows = run(0, in_order) + run(1, in_order) + run(2, together)
    s = a.summarize(rows)[(64, 1.1, "named", 8.01)]
    assert s["end"] == "ALL OPEN" and s["path"] == "IN ORDER" and s["paths"] == {"IN ORDER": 2, "TOGETHER": 1}
    rows = run(0, together, "interchangeable") + run(1, stuck, "interchangeable") + run(2, stuck, "interchangeable")
    s = a.summarize(rows)[(64, 1.1, "interchangeable", 8.01)]
    assert s["end"] == "STUCK" and s["path"] == "TOGETHER"
    # the second rung before the first is not in order; damage wins over everything
    backwards = [[64], [10, 4, 40, 10], [0, 40, 4, 20], [0, 0, 0, 64]]
    assert a.path_of(run(0, backwards), 3) == "PARTLY IN ORDER"
    melt = [[64], [0, 0, 0, 30, 34]]
    assert a.end_of(run(0, melt)[-1], 3) == "DAMAGED"
    assert a.dim_of({"dims": "1x(4 4 4 4)"}) == 4
