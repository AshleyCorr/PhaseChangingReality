"""T88, the pole test (PREREGISTRATION.md section T88): the runner's stores and reading, conservation, and the
verdict words on hand-built rows."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_t88 import cell_word, replica_word                              # noqa: E402
from graphity.cqg_d import hamiltonian, torus                                # noqa: E402
from graphity.sealed_tie_d import run_sealed_bath_table_d, table_total       # noqa: E402
from run_curled_bath_tie_d import table_of                                   # noqa: E402
from run_rod_pole import fed_stores, layers_of, reading                      # noqa: E402

DIMS = [4, 4, 8]


def test_the_energy_sits_in_the_side_0_stores_of_the_fed_layers_only():
    adj, part = torus(DIMS)
    stores = fed_stores(DIMS, part, 120.0, 1)
    z = layers_of(DIMS)
    assert np.isclose(stores.sum(), 120.0)
    assert np.all(stores[z >= 1] == 0) and np.all(stores[part == 1] == 0)
    fed = (z < 1) & (part == 0)
    assert fed.sum() == 8 and np.allclose(stores[fed], 15.0)


def test_the_exact_rod_reads_closed_everywhere():
    adj, part = torus(DIMS)
    ftab = table_of("all_at_the_last", 1.25, 3)
    r = reading(adj, DIMS, 1.25, ftab, np.zeros(adj.shape[0]))
    assert r["d1"] == adj.shape[0] and r["n_open3"] == 0 and r["open_regions3"] == 0 and r["damaged"] == 0
    assert r["profile"] == " ".join(["0"] * 8) and r["fed_open3"] == 0 and r["far_open3"] == 0
    assert np.isclose(r["t_tie"], adj.shape[0] * 4 * 0.25)      # every point d = 1 costs a = 4 (lambda - 1)


def test_with_nothing_in_the_stores_nothing_moves_and_with_energy_the_total_is_conserved():
    adj, part = torus(DIMS)
    side_u = np.flatnonzero(part == 0)
    ftab = table_of("all_at_the_last", 1.25, 3)
    empty = np.zeros(adj.shape[0])
    before = adj.copy()
    run_sealed_bath_table_d(adj, side_u, empty, 20, 5, 1.25, ftab, True)
    assert np.array_equal(adj, before) and empty.sum() == 0
    stores = fed_stores(DIMS, part, 240.0, 1)
    e0 = hamiltonian(adj, 1.25) + table_total(adj, ftab) + stores.sum()
    run_sealed_bath_table_d(adj, side_u, stores, 200, 7, 1.25, ftab, True)
    e1 = hamiltonian(adj, 1.25) + table_total(adj, ftab) + stores.sum()
    assert abs(e1 - e0) < 1e-9
    assert not np.array_equal(adj, before)         # 30 a store at one end, above the rod's wall of 24, moves something


def test_same_seed_reproduces():
    a = [torus(DIMS) for _ in range(2)]
    out = []
    for adj, part in a:
        stores = fed_stores(DIMS, part, 120.0, 1)
        run_sealed_bath_table_d(adj, np.flatnonzero(part == 0), stores, 50, 3, 1.25,
                                table_of("all_at_the_last", 1.25, 3), True)
        out.append((adj.copy(), stores.copy()))
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])


def row(sweep, largest, regions, fed, far):
    return dict(sweep=str(sweep), largest_open3=str(largest), open_regions3=str(regions), fed_open3=str(fed),
                far_open3=str(far))


def test_the_verdict_words():
    nothing = [row(s, 4, 0, 0, 0) for s in (0, 100, 200)]
    assert replica_word(nothing) == ("NOTHING OPENS", None, None)
    one = [row(0, 0, 0, 0, 0), row(100, 20, 1, 6, 0), row(200, 60, 1, 16, 2), row(300, 120, 1, 16, 12)]
    assert replica_word(one) == ("ONE FRONT", 100, 300)
    two = [row(0, 0, 0, 0, 0), row(100, 20, 1, 6, 0), row(200, 40, 2, 16, 0), row(300, 120, 1, 16, 12)]
    assert replica_word(two) == ("SEVERAL", 100, 300)
    elsewhere = [row(0, 0, 0, 0, 0), row(100, 20, 1, 0, 0), row(200, 120, 1, 16, 12)]
    assert replica_word(elsewhere) == ("SEVERAL", 100, 200)
    late_split = [row(0, 0, 0, 0, 0), row(100, 20, 1, 6, 0), row(200, 120, 1, 16, 12), row(300, 100, 2, 16, 16)]
    assert replica_word(late_split)[0] == "ONE FRONT"          # a split after the far side opened is not scored
    assert cell_word(["ONE FRONT", "ONE FRONT", "SEVERAL"]) == "ONE FRONT"
    assert cell_word(["ONE FRONT", "SEVERAL"]) == "SEVERAL"
