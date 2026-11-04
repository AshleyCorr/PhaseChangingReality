"""T41's analysis, tested before any run (PREREGISTRATION T41)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t41 as a  # noqa: E402


def blk(lam, n, c, rep, sweep, h, d):
    r = dict(dims="4 8 12", lam=str(lam), N=str(n), C=str(c), replica=str(rep), sweep=str(sweep), h_per_vertex=str(h))
    for k in range(8):
        r["d%d" % k] = str(d[k] if k < len(d) else 0)
    return r


def test_room_needed():
    n, lam = 100, 1.25
    rows = []
    for rep in range(3):
        rows += [blk(lam, n, 200, rep, 0, 1.0, [0, 0, n]), blk(lam, n, 200, rep, 500, 0.02, [0, 0, 2, 98])]
        rows += [blk(lam, n, 25, rep, 0, 1.0, [0, 0, n]), blk(lam, n, 25, rep, 500, 3.0, [0, 0, 10, 50, 40])]
    cells, majority = a.summarize(rows)
    assert majority[(lam, n, 200)] == "FLAT" and majority[(lam, n, 25)] == "MELTED"
    assert a.verdicts(majority)[lam] == (200, "ROOM NEEDED")


def test_stays_and_never_opens():
    n, lam = 100, 1.40
    rows = [blk(lam, n, 50, rep, t, 1.6, [0, 0, n]) for rep in range(3) for t in (0, 500)]
    cells, majority = a.summarize(rows)
    assert majority[(lam, n, 50)] == "STAYS"
    assert a.verdicts(majority)[lam] == (None, "NEVER OPENS")
