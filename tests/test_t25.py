"""T25's reading rules (scripts/analyse_t25.py) and the cooling schedule (scripts/run_scrap_race.py)."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a = load("analyse_t25")
race = load("run_scrap_race")


def blocks(t_cool, rep, gone_at=None, nb=10):
    """Rows of one followed replica; the leftover vanishes at block `gone_at` (None: survives)."""
    out = []
    for b in range(nb + 1):
        present = gone_at is None or b < gone_at
        out.append(dict(t_cool=str(t_cool), replica=str(rep), followed="True", block=str(b), sweep=str(500 * b),
                        g=str(1.25 if b == 0 else 0.25), n_d1=str(4 if present else 0), pieces_d1=str(1 if present else 0)))
    return out


def test_survival_and_last_seen():
    assert a.survived(blocks(300, 0))
    assert not a.survived(blocks(300, 0, gone_at=4))
    assert a.last_seen(blocks(300, 0)) is None
    assert a.last_seen(blocks(300, 0, gone_at=4)) == (1500, 0.25)


def test_verdict_freezes_in_when_fast_cooling_keeps_a_majority():
    rows = []
    for rep in range(10):
        rows += blocks(300, rep)                                 # all survive
        rows += blocks(3000, rep, gone_at=(3 if rep < 4 else None))   # 6 of 10 survive
        rows += blocks(100000, rep, gone_at=2)                   # none survive
    v, t_star = a.verdict(a.shares(rows))
    assert v == "FREEZES IN" and t_star == 3000


def test_verdict_always_heals_when_no_schedule_keeps_a_majority():
    rows = []
    for rep in range(10):
        for t in (300, 3000, 100000):
            rows += blocks(t, rep, gone_at=(None if rep < 3 else 2))
    v, t_star = a.verdict(a.shares(rows))
    assert v == "ALWAYS HEALS" and t_star is None


def test_too_few_followed_is_not_read_and_unfollowed_rows_are_ignored():
    rows = []
    for rep in range(5):
        rows += blocks(300, rep)
    rows.append(dict(t_cool="300", replica="9", followed="False", block="0", sweep="0", g="1.25", n_d1="8", pieces_d1="2"))
    assert a.shares(rows) == {300: (5, 5)}
    assert a.verdict(a.shares(rows)) == ("NOT READ", None)


def test_schedule_falls_by_the_same_factor_and_then_holds():
    gs = race.schedule(1.25, 0.25, 3000, 1000, 500)
    assert len(gs) == 6 + 2
    assert abs(gs[5] - 0.25) < 1e-12 and gs[6] == 0.25 and gs[7] == 0.25
    ratios = [gs[i + 1] / gs[i] for i in range(5)]
    assert max(ratios) - min(ratios) < 1e-9
    assert abs(gs[0] - 1.25 * (0.2) ** (1 / 6)) < 1e-12
