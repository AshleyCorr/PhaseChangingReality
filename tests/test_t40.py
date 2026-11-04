"""T40's analysis, tested before any run (PREREGISTRATION T40)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t40 as a  # noqa: E402


def blk(dims, lam, n, spark, rep, sweep, d):
    r = dict(dims=dims, lam=str(lam), N=str(n), spark=str(spark), replica=str(rep), sweep=str(sweep))
    for k in range(8):
        r["d%d" % k] = str(d[k] if k < len(d) else 0)
    return r


def test_flat_points_are_not_melted_at_eight_links():
    n = 100
    rows = [blk("4 4 4 12", 1.25, n, 80, 0, 0, [0, n]), blk("4 4 4 12", 1.25, n, 80, 0, 250, [0, 0, 0, 0, n])]
    s = a.summarize(rows)
    assert s[("4 4 4 12", 1.25, n, 80.0)]["majority"] == "THREE TOGETHER"


def test_threshold_and_pattern():
    n, rows = 100, []
    for rep in range(3):
        rows += [blk("4 4 4 12", 1.25, n, 20, rep, 0, [0, n]), blk("4 4 4 12", 1.25, n, 20, rep, 50000, [0, n])]
        rows += [blk("4 4 4 12", 1.25, n, 60, rep, t, [0, 0, n]) for t in range(0, 6001, 250)]
        rows[-25] = blk("4 4 4 12", 1.25, n, 60, rep, 0, [0, n])
        rows += [blk("4 4 4 12", 1.25, n, 60, rep, t, [0, 0, 0, n]) for t in range(6250, 12001, 250)]  # rests on rung 3 too
    s = a.summarize(rows)
    assert s[("4 4 4 12", 1.25, n, 20.0)]["majority"] == "STALLS" and s[("4 4 4 12", 1.25, n, 20.0)]["leave"] == 0
    assert s[("4 4 4 12", 1.25, n, 60.0)]["leave"] == 3
    v = a.verdicts(s)[("4 4 4 12", 1.25, n)]
    assert v[0] == 60.0 and v[1] == "ONE AT A TIME"


def test_melted_counts_only_above_four():
    n = 100
    rows = [blk("4 4 4 12", 1.25, n, 160, 0, 0, [0, n]), blk("4 4 4 12", 1.25, n, 160, 0, 250, [0, 10, 10, 10, 40, 30])]
    assert a.summarize(rows)[("4 4 4 12", 1.25, n, 160.0)]["majority"] == "MELTED"
