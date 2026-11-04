"""T67 (PREREGISTRATION T67): the feed runner, its analyser and its configs."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t67 as an                                   # noqa: E402
import make_t67_configs as mk                              # noqa: E402
from run_slab_feed import run_cell                         # noqa: E402


def test_run_cell_conserves_energy_reproduces_and_stops():
    a = list(run_cell([4, 6, 6], 1.25, 4.0, "shared", 144, 400, 50, 11))
    b = list(run_cell([4, 6, 6], 1.25, 4.0, "shared", 144, 400, 50, 11))
    assert a[0][1] == dict(open=0, curled=144, damaged=0, regions=0, largest=0)   # the slab, every point curled
    assert all(r[4] < 1e-6 for r in a)                                            # drift
    assert [r[1] for r in a] == [r[1] for r in b]                                 # same seed, same census
    assert all(r[5] is None for r in a[:-1]) and a[-1][5] in ("OPEN", "DAMAGED", "TIME")
    assert abs(a[-1][3] - 4.0 * a[-1][0]) < 1e-9                                  # fed = rate x sweeps
    c = list(run_cell([4, 6, 6], 1.25, 4.0, "shared", 144, 400, 50, 12))
    assert [r[1] for r in c] != [r[1] for r in a]


def test_spot_mode_runs_with_per_vertex_stores():
    a = list(run_cell([4, 6, 6], 1.25, 1.0, "spot", 144, 200, 50, 5))
    assert all(r[4] < 1e-6 for r in a) and a[-1][5] in ("OPEN", "DAMAGED", "TIME")


def _rows(n, open_, largest, damaged, curled=None, every=50):
    k = len(open_)
    return pd.DataFrame(dict(N=n, mode="shared", C=n, rate=1.0, replica=0, sweep=[i * every for i in range(k)],
                             open=open_, curled=curled if curled is not None else [n - o for o in open_],
                             damaged=damaged, regions=[1 if o else 0 for o in open_], largest=largest,
                             store_mean=0.2, fed=[i * 50.0 for i in range(k)], drift=0.0, stop=""))


def test_replica_words():
    n = 256
    opens = an.replica_outcome(_rows(n, [0, 40, 150, 240], [0, 40, 150, 240], [0, 2, 4, 6]))
    assert (opens["word"], opens["first_escape"], opens["front"], opens["completion"]) == ("OPENS", 50, 100, 150)
    front = an.replica_outcome(_rows(n, [0, 40, 150, 160], [0, 40, 150, 160], [0, 2, 4, 30]))
    assert front["word"] == "FRONT"
    breaks = an.replica_outcome(_rows(n, [0, 40, 60, 70], [0, 10, 12, 12], [0, 4, 20, 30]))
    assert breaks["word"] == "BREAKS" and breaks["stop_sweep"] == 150
    sits = an.replica_outcome(_rows(n, [0, 0, 0], [0, 0, 0], [0, 0, 0]))
    assert sits["word"] == "SITS" and sits["first_escape"] is None
    late = an.replica_outcome(_rows(n, [0, 40, 60, 200], [0, 10, 12, 200], [0, 4, 30, 2]))
    assert late["word"] == "BREAKS"                        # a front after the break does not count


def test_cell_and_size_words():
    assert an.cell_word(["BREAKS"] * 5 + ["OPENS"] * 3) == "BREAKS"
    assert an.cell_word(["BREAKS"] * 4 + ["OPENS"] * 4) == "MIXED"
    assert an.cell_word(["OPENS"] * 3) == "MIXED"          # too few replicas
    assert an.size_verdict(["BREAKS", "FRONT", "BREAKS"]) == "OPENS"
    assert an.size_verdict(["BREAKS", "BREAKS"]) == "BREAKS"
    assert an.size_verdict(["BREAKS", "SITS"]) == "MIXED"


def test_config_maker_cells():
    names = [c[0] for c in mk.cells()]
    assert names == ["t67_shared_cn_l8", "t67_shared_c4n_l8", "t67_spot_l8", "t67_shared_cn_l12", "t67_shared_c4n_l12", "t67_spot_l12"]
    assert [c[0] for c in mk.cells(held_out=True)] == ["t67_shared_cn_l16", "t67_shared_c4n_l16", "t67_spot_l16"]
    cfg = mk.config("t67_spot_l8", 8, "spot", "N", "x", 1)
    assert cfg["dims"] == [4, 8, 8] and cfg["rates"] == [0.0625, 0.25, 1.0, 4.0] and cfg["mode"] == "spot"
