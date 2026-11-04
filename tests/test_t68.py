"""T68 (PREREGISTRATION T68): the feed schedules and the many-point places added to the T67 runner, its configs and
its analyser; and that T67's cells are unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t68 as an                                                                 # noqa: E402
import make_t68_configs as mk                                                            # noqa: E402
from run_slab_feed import MODES, SCHEDULES, PULSE, cumulative_feed, run_cell, seed_for    # noqa: E402


def test_cumulative_feed_shapes():
    assert cumulative_feed("fixed", 0.25, 400, 2000) == 100.0
    assert abs(cumulative_feed("ramp", 0.25, 2000, 2000) - 500.0) < 1e-12          # the mean over the run is the rate
    assert cumulative_feed("ramp", 0.25, 1000, 2000) == 125.0                      # a quarter of the energy by half time
    assert cumulative_feed("pulse", 0.25, 500, 2000) == 0.0 and cumulative_feed("pulse", 0.25, 512, 2000) == PULSE
    assert cumulative_feed("pulse", 1.0, 2000, 2000) == 15 * PULSE


def test_t67_seeds_unchanged_and_t68_seeds_differ():
    old = int(np.random.SeedSequence([20266701, 256, 0, 250, 3]).generate_state(1)[0])
    assert seed_for(20266701, 256, "shared", 0.25, 3) == old
    assert seed_for(20266701, 256, "shared", 0.25, 3, "ramp", 1) != old
    assert seed_for(20266701, 256, "spots", 0.25, 3, "fixed", 16) != seed_for(20266701, 256, "all", 0.25, 3, "fixed", 256)
    assert MODES[:2] == ("shared", "spot") and SCHEDULES[0] == "fixed"


def test_run_cell_schedules_and_places_conserve_energy_and_feed_as_written():
    for sched, mode, k in [("ramp", "shared", 1), ("pulse", "spot", 1), ("fixed", "spots", 8), ("ramp", "all", 144)]:
        rows = list(run_cell([4, 6, 6], 1.25, 1.0, mode, 144, 400, 50, 7, sched, k))
        assert all(r[4] < 1e-6 for r in rows), (sched, mode)                       # drift
        assert rows[0][1]["curled"] == 144
        last = rows[-1]
        assert abs(last[3] - cumulative_feed(sched, 1.0, last[0], 400)) < 1e-9       # fed follows F(t)
        assert last[5] in ("OPEN", "DAMAGED", "TIME")
    a = list(run_cell([4, 6, 6], 1.25, 1.0, "spots", 144, 200, 50, 7, "fixed", 8))
    b = list(run_cell([4, 6, 6], 1.25, 1.0, "spots", 144, 200, 50, 7, "fixed", 8))
    assert [r[1] for r in a] == [r[1] for r in b]                                    # same seed, same spots, same census


def test_config_maker_and_analyser_grouping():
    cs = mk.cells()
    assert len(cs) == 36 and cs[0][0] == "t68_fixed_shared_lam125" and cs[-1][0] == "t68_pulse_all_lam160"
    assert all(c[1] == [8, 12] for c in cs)
    assert [c[0] for c in mk.cells(held_out=True)][:2] == ["t68_fixed_shared_lam125_l16", "t68_fixed_spot_lam125_l16"]
    cfg = mk.config(*cs[5], 1)
    assert cfg["schedule"] == "ramp" and cfg["mode"] == "spot" and cfg["rates"] == [0.0625, 0.25, 1.0] and cfg["lambda"] == 1.25
    n = 256
    rows = []
    for rep in range(4):
        for i, (o, d) in enumerate([(0, 0), (40, 2), (60, 20), (70, 30)]):
            rows.append(dict(N=n, lam=1.4, schedule="ramp", mode="spots", k=16, C=n, rate=0.25, replica=rep, sweep=50 * i,
                             open=o, curled=n - o, damaged=d, regions=1, largest=10, store_mean=0.1, fed=10.0 * i, drift=0.0, stop=""))
    table = an.cells(pd.DataFrame(rows))
    assert list(table) == [(1.4, "ramp", "spots", 16, n, 0.25, n)]
    assert table[(1.4, "ramp", "spots", 16, n, 0.25, n)]["word"] == "BREAKS"
