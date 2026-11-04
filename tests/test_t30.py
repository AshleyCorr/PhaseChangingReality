"""T30 and T32 reading rules (scripts/analyse_t30.py) on rows whose answers are known."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t30", SCRIPTS / "analyse_t30.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)

LAM, N = 1.10, 288
ONE = 4 * (LAM - 1)


def block(sweep, h, d1=0, d2=0, d3=0, melted=0, left=True, c=576, spark=8.0, rep=0, n=N):
    row = dict(lam=str(LAM), N=str(n), C=str(c), spark=str(spark), replica=str(rep), sweep=str(sweep),
               h_per_vertex=str(h), melted=str(melted), left=str(left), final="False")
    for k in range(8):
        row["d%d" % k] = "0"
    row["d1"], row["d2"], row["d3"] = str(d1), str(d2), str(d3)
    row["d4"] = str(melted)
    return row


def path(kind, rep=0, c=576):
    """Block sequences per replica, every 500 sweeps to 20,000."""
    out = []
    for sw in range(0, 20001, 500):
        if kind == "cascade":            # straight through: middle rung passed inside one block, then flat
            h = 2 * ONE if sw < 1000 else (ONE if sw == 1000 else 0.0)
            d1, d2, d3 = (N, 0, 0) if sw < 1000 else ((0, N, 0) if sw == 1000 else (0, 0, N))
        elif kind == "stepwise":         # rests on the middle rung for 8,000 sweeps, then flat
            h = 2 * ONE if sw < 1000 else (ONE if sw <= 9000 else 0.0)
            d1, d2, d3 = (N, 0, 0) if sw < 1000 else ((0, N, 0) if sw <= 9000 else (0, 0, N))
        elif kind == "middle":           # reaches the middle rung and stays
            h = 2 * ONE if sw < 1000 else ONE
            d1, d2, d3 = (N, 0, 0) if sw < 1000 else (0, N, 0)
        elif kind == "stuck":
            h, d1, d2, d3 = 2 * ONE, N, 0, 0
        elif kind == "melted":
            h, d1, d2, d3 = 2 * ONE + 1.0, 0, 0, 0
        out.append(block(sw, h, d1, d2, d3, melted=(N if kind == "melted" else 0), rep=rep, c=c))
    return out


def test_outcomes_and_resting():
    for kind, want, rested in (("cascade", "FLAT", False), ("stepwise", "FLAT", True), ("middle", "MIDDLE", True),
                               ("stuck", "STUCK", False), ("melted", "MELTED", False)):
        o, r = a.outcome(path(kind), LAM, N)
        assert (o, r) == (want, rested), (kind, o, r)


def test_verdicts_all_at_once_and_one_at_a_time_and_first_only():
    rows = []
    for rep in range(6):
        rows += path("cascade", rep)
    s = a.read_t30(rows)
    assert s["majority"][(LAM, N, 576)] == "FLAT" and s["c_star"][(LAM, N)] == 576
    assert a.verdict_t30(s) == "ALL AT ONCE"
    rows = []
    for rep in range(6):
        rows += path("stepwise", rep)
    assert a.verdict_t30(a.read_t30(rows)) == "ONE AT A TIME"
    rows = []
    for rep in range(6):
        rows += path("middle", rep, c=576) + path("melted", rep, c=36)
    s = a.read_t30(rows)
    assert s["majority"][(LAM, N, 36)] == "MELTED" and a.verdict_t30(s) == "FIRST ONLY"
    rows = []
    for rep in range(6):
        rows += path("stuck", rep)
    assert a.verdict_t30(a.read_t30(rows)) == "NEVER OPENS"


def test_c_star_is_the_smallest_flat_majority():
    rows = []
    for rep in range(6):
        rows += path("cascade", rep, c=576) + path("cascade", rep, c=288) + path("melted", rep, c=72)
    s = a.read_t30(rows)
    assert s["c_star"][(LAM, N)] == 288


def test_t32_threshold_fixed_grows_and_sharp():
    def rows_for(n, e_star, leak_below=False):
        out = []
        for e in (4.0, 6.0, 8.0, 10.0, 12.0):
            for rep in range(5):
                left = e >= e_star or (leak_below and e == e_star - 2 and rep == 0)
                out.append(block(20000, 0.0, left=left, c=1, spark=e, rep=rep, n=n))
        return out
    rows = rows_for(192, 8.0) + rows_for(288, 8.0) + rows_for(512, 8.0)
    v, e_star, sharp = a.verdict_t32(a.read_t32(rows))
    assert v == "FIXED WALL" and set(e_star.values()) == {8.0} and sharp
    rows = rows_for(192, 6.0) + rows_for(288, 8.0) + rows_for(512, 10.0, leak_below=True)
    v, e_star, sharp = a.verdict_t32(a.read_t32(rows))
    assert v == "GROWS" and not sharp
    rows = rows_for(192, 8.0) + [block(20000, 0.0, left=False, c=1, spark=e, rep=r, n=288) for e in (4.0, 8.0) for r in range(3)]
    assert a.verdict_t32(a.read_t32(rows))[0] == "NOT READ"


def test_the_gas_is_told_apart_from_the_tori():
    """The pre-registration reports the gas separately (T30-gas); the reader tells it by its dims label."""
    assert a.is_gas({"dims": "8x(4 4 4)"}) and not a.is_gas({"dims": "4 4 18"})
