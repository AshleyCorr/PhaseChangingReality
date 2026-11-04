"""The sealed bath at any dimension (graphity.sealed_d) and the any-degree local dimension (graphity.dimension).

At four links the new kernel must make exactly the chain of sealed.run_sealed_bath without the cap, draw for
draw; at six links H + stores is conserved to the last unit, with a shared bath and with per-vertex stores; and
the local dimension reads the 3D ladder (VISION Update 22; O41) as 3, 2, 1, 0.
"""
import numpy as np

from graphity import cqg_d
from graphity.cqg import NO_CAP, torus
from graphity.dimension import local_dimension, local_dimension_d
from graphity.sealed import run_sealed_bath
from graphity.sealed_d import energy_d, run_sealed_bath_d


def test_four_links_is_run_sealed_bath_draw_for_draw():
    lam = 1.25
    a1, part = torus(16, 4, NO_CAP)
    a2 = a1.copy()
    side_u = np.flatnonzero(part == 0)
    d1 = np.zeros(32); d1[0] = 40.0
    d2 = d1.copy()
    s1, x1, m1, t1, acc1 = run_sealed_bath(a1, side_u, d1, 300, 2024, lam, NO_CAP)
    s2, x2, m2, t2, acc2 = run_sealed_bath_d(a2, side_u, d2, 300, 2024, lam)
    assert (s1 == s2).all() and (x1 == x2).all() and (a1 == a2).all()
    assert np.allclose(t1, t2) and np.allclose(d1, d2) and acc1 == acc2


def test_four_links_by_vertex_matches_too():
    lam = 1.25
    a1, part = torus(12, 12, NO_CAP)
    a2 = a1.copy()
    side_u = np.flatnonzero(part == 0)
    st1 = np.zeros(144); st1[side_u[5]] = 96.0
    st2 = st1.copy()
    s1, x1, _, t1, _ = run_sealed_bath(a1, side_u, st1, 120, 7, lam, NO_CAP, by_vertex=True)
    s2, x2, _, t2, _ = run_sealed_bath_d(a2, side_u, st2, 120, 7, lam, by_vertex=True)
    assert (s1 == s2).all() and (x1 == x2).all() and (a1 == a2).all() and np.allclose(st1, st2)


def test_six_links_conserves_energy_shared_and_per_vertex():
    lam = 1.10
    for by_vertex in (False, True):
        adj, part = cqg_d.torus([4, 4, 6])
        n = adj.shape[0]
        side_u = np.flatnonzero(part == 0)
        stores = np.zeros(n if by_vertex else 2 * n)
        stores[int(side_u[0]) if by_vertex else 0] = 40.0
        e0 = energy_d(adj, lam) + stores.sum()
        s, x, mean, tot, _ = run_sealed_bath_d(adj, side_u, stores, 200, 3, lam, by_vertex=by_vertex)
        h = 16.0 * (3 * n - s) + 4.0 * lam * x
        assert np.allclose(h + tot, e0), by_vertex
        assert abs(energy_d(adj, lam) + stores.sum() - e0) < 1e-9
        assert (stores >= -1e-12).all()
        assert cqg_d.is_valid(adj)


def test_six_links_seed_below_zero_carries_on():
    lam = 1.10
    a1, part = cqg_d.torus([4, 4, 6]); a2 = a1.copy()
    side_u = np.flatnonzero(part == 0)
    d1 = np.zeros(192); d1[0] = 20.0; d2 = d1.copy()
    s_one, x_one, _, t_one, _ = run_sealed_bath_d(a1, side_u, d1, 150, 11, lam)
    s_a, x_a, _, t_a, _ = run_sealed_bath_d(a2, side_u, d2, 90, 11, lam)
    s_b, x_b, _, t_b, _ = run_sealed_bath_d(a2, side_u, d2, 60, -1, lam)
    assert (np.concatenate([s_a, s_b]) == s_one).all() and (np.concatenate([x_a, x_b]) == x_one).all()
    assert (a1 == a2).all()


def test_local_dimension_d_reads_the_ladder_and_agrees_at_four_links():
    for dims, expect in (([6, 6, 6], 3), ([4, 6, 6], 2), ([4, 4, 6], 1), ([4, 4, 4], 0)):
        adj, _ = cqg_d.torus(dims)
        d = local_dimension_d(adj)
        assert (d == expect).all(), (dims, set(d.tolist()))
    adj, _ = torus(16, 4, NO_CAP)
    assert (local_dimension_d(adj) == local_dimension(adj)).all()
    adj, _ = torus(8, 8, NO_CAP)
    assert (local_dimension_d(adj) == 2).all()
    # the ladder's energies: 4 (lam - 1) per vertex per curled direction, additive (O41)
    lam = 1.25
    for dims, rungs in (([6, 6, 6], 0), ([4, 6, 6], 1), ([4, 4, 6], 2), ([4, 4, 4], 3)):
        adj, _ = cqg_d.torus(dims)
        assert abs(energy_d(adj, lam) / adj.shape[0] - 4 * (lam - 1) * rungs) < 1e-9


def test_the_runner_gas_start_is_separate_valid_tori_that_conserve_energy():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_sealed_curled_d", Path(__file__).resolve().parents[1] / "scripts" / "run_sealed_curled_d.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    adj, part = m.gas([4, 4, 4], 3)
    assert adj.shape == (192, 6) and cqg_d.is_valid(adj)
    from graphity.dimension import pieces_of
    assert pieces_of(adj, np.ones(192, dtype=bool)) == [64, 64, 64]      # connectivity.py is four-link only
    assert (local_dimension_d(adj) == 0).all()
    lam = 1.10
    assert abs(energy_d(adj, lam) / 192 - 12 * (lam - 1)) < 1e-9          # three rungs
    side_u = np.flatnonzero(part == 0)
    stores = np.zeros(384); stores[0] = 8.0
    e0 = energy_d(adj, lam) + stores.sum()
    s, x, _, tot, _ = run_sealed_bath_d(adj, side_u, stores, 100, 5, lam)
    assert abs(energy_d(adj, lam) + stores.sum() - e0) < 1e-9


def test_the_runner_local_heat_puts_the_spark_in_one_vertex_store_and_conserves(tmp_path):
    import csv
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_sealed_curled_d", Path(__file__).resolve().parents[1] / "scripts" / "run_sealed_curled_d.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = dict(name="smoke_local", dims=[6, 6, 6], capacities=["2*N"], sparks=[80.0], replicas=1, n_sweeps=40,
               record_every=20, seed=3, local_heat=True)
    cfg["lambda"] = 1.25
    p = tmp_path / "smoke_local.json"
    p.write_text(json.dumps(cfg))
    m.main(str(p), str(tmp_path))
    rows = list(csv.DictReader(open(tmp_path / "smoke_local.csv", newline="")))
    assert rows[0]["C"] == "216" and rows[0]["local_heat"] == "True"
    assert all(float(r["drift"]) < 1e-9 for r in rows)
    assert float(rows[0]["total"]) == 80.0
