"""scripts/exact_walls_d.py against walls already on the record: paper 1's moves A and B out of the 16 x 4 torus,
O41's move out of the 4 x 4 x 6 torus, and O49's out of a long two-curled torus."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("exact_walls_d", SCRIPTS / "exact_walls_d.py")
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)

from graphity.cqg_d import torus     # noqa: E402


def kinds(out):
    return {(ds, dx): (round(c, 6), ways) for c, ds, dx, ways in out}


def test_two_dimensional_tube_gives_paper_ones_moves():
    adj, part = torus([16, 4])
    k = kinds(w.walls(adj, part, 1.25, top=10))
    # move A: (-2, -4), 3N ways in all; each switch is seen from both of its side-0 vertices, so from one
    # vertex there are 3N * 2 / (N/2) = 12; cost 32 - 16 lam = 12
    assert k[(-2, -4)] == (12.0, 12)
    # move B: (-4, -10), 2N ways, 8 from one vertex; cost 64 - 40 lam = 14
    assert k[(-4, -10)] == (14.0, 8)


def test_short_and_long_two_curled_tori_differ_as_o41_and_o49_say():
    lam = 1.25
    short = kinds(w.walls(*torus([4, 4, 6]), lam, top=10))
    assert (-6, -20) in short and short[(-6, -20)][0] == -4.0          # 96 - 80 lam: downhill at 1.25
    long = kinds(w.walls(*torus([4, 4, 18]), lam, top=10))
    assert (-6, -20) not in long
    assert long[(-6, -16)][0] == 16.0                                 # 96 - 64 lam


def test_gas_spec_parses_and_is_separate_tori():
    (adj, part), label = w.parse("4,4,4x2")
    assert adj.shape == (128, 6) and label == "4,4,4x2"
    assert set(int(v) for v in adj[:64].ravel()) <= set(range(64))
