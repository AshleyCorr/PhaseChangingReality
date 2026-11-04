"""T39's analysis, tested before any run (PREREGISTRATION T39): synthetic block records for D = 3 and D = 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t39 as a  # noqa: E402


def block(dims, lam, n, c, rep, sweep, h, d):
    r = dict(dims=" ".join(map(str, dims)), lam=str(lam), N=str(n), C=str(c), replica=str(rep), sweep=str(sweep),
             h_per_vertex=str(h))
    for k in range(8):
        r["d%d" % k] = str(d[k] if k < len(d) else 0)
    return r


def test_flat_cascade_in_four_directions():
    n, lam = 1000, 1.5
    rows = [block([4, 4, 8, 8], lam, n, 250, 0, 0, 4.0, [0, 0, n]),
            block([4, 4, 8, 8], lam, n, 250, 0, 250, 2.0, [0, 0, 400, 600]),
            block([4, 4, 8, 8], lam, n, 250, 0, 500, 0.05, [0, 0, 0, 20, 980])]
    s = a.summarize(rows)
    assert s["majority"][(4, lam, n, 250)] == "FLAT"
    assert s["rows"][(4, lam)] == "WINDOW"
    assert s["mechanism"][(4, lam)] == {"CASCADE": 1}


def test_rest_on_the_middle_rung_is_stepwise_and_a_stall_is_stuck():
    n, lam = 288, 1.25
    rows = [block([4, 4, 18], lam, n, 72, 0, t, 1.0, [0, 0, n]) for t in range(0, 6001, 200)]
    rows.append(block([4, 4, 18], lam, n, 72, 0, 6200, 0.02, [0, 0, 5, n - 5]))
    rows += [block([4, 4, 18], lam, n, 72, 1, 0, 2.0, [0, n]), block([4, 4, 18], lam, n, 72, 1, 200, 1.95, [0, 200, 88])]
    s = a.summarize(rows)
    assert s["mechanism"][(3, lam)] == {"STEPWISE": 1}
    assert s["cells"][(3, lam, n, 72)] == {"FLAT": 1, "STUCK": 1}
    assert s["census"][(3, lam, n, 72)] == {"PAST": 1, "STALLED": 1}


def test_melted_reads_above_d():
    n, lam = 100, 1.4
    rows = [block([4, 4, 18], lam, n, 6, 0, 0, 3.0, [0, 10, 10, 50, 30])]
    assert a.summarize(rows)["majority"][(3, lam, n, 6)] == "MELTED"
