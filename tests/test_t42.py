"""T42's analysis, tested before any run: T34's rules applied separately to the weighted runs and the named control."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t42 as a  # noqa: E402
from analyse_t34 import cells, verdict  # noqa: E402


def row(weighted, rep, sweep, d, melted):
    r = dict(N="512", spark="256.0", replica=str(rep), sweep=str(sweep), local_heat="False", weighted=str(weighted),
             melted=str(melted))
    for k in range(8):
        r["d%d" % k] = str(d[k] if k < len(d) else 0)
    return r


def test_split_and_read():
    rows = []
    for rep in range(6):
        rows += [row(True, rep, 0, [0, 0, 0, 512], 0), row(True, rep, 250, [0, 300, 0, 212], 0)]
        rows += [row(False, rep, 0, [0, 0, 0, 472], 40), row(False, rep, 250, [0, 0, 0, 472], 40)]
    w, n = a.split(rows)
    assert verdict(cells(w)) == "RE-CURLS" and verdict(cells(n)) == "MELTS"
