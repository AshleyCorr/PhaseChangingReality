"""T54's patch construction and reading (scripts/exact_t54_patch.py), written before any computation on the
pre-registered sizes: the helix switch, the whole-sheet limit, the energy under the tie, the fit and the verdict words."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from graphity.cqg_d import hamiltonian, is_valid, surplus, torus, total_squares
from graphity.dimension import local_dimension_d

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t54 = _load("exact_t54_patch")
L = 6


def test_one_helix_switch_is_valid_and_opens_the_pair_of_rings():
    adj, _ = torus([4, L, L])
    s0, x0 = total_squares(adj), surplus(adj)
    t = adj.copy()
    t54.helix_switch(t, L, 0, 0)
    assert is_valid(t)
    assert total_squares(t) - s0 == -6 and surplus(t) - x0 == -8
    d = local_dimension_d(t)
    assert int((d == 3).sum()) > 0 and int((d == 2).sum()) < adj.shape[0]


def test_the_whole_sheet_patch_is_flat_space_exactly():
    adj, _ = torus([4, L, L])
    t = adj.copy()
    for x in range(L):
        for y in range(L):
            t54.helix_switch(t, L, x, y)
    assert is_valid(t) and hamiltonian(t, 1.25) == 0.0
    assert (local_dimension_d(t) == 3).all()
    assert t54.energy(t, 1.25, "all_at_the_last") == 0.0


def test_patch_sizes_nest_and_the_slab_energies_are_the_ladder():
    adj, part = torus([4, L, L])
    n = adj.shape[0]
    assert t54.energy(adj, 1.25, "none") == n * 1.0
    assert t54.energy(adj, 1.25, "all_at_the_last") == n * 3.0            # 2a per point of tie on a slab at d = 2
    p1, touched1 = t54.patch(adj, L, 1)
    p2, touched2 = t54.patch(adj, L, 2)
    assert is_valid(p1) and is_valid(p2) and set(touched1) < set(touched2)
    assert t54.energy(p1, 1.25, "none") == 56.0 + n * 1.0
    moves = t54.cheapest_move(p1, part, touched1, [1.25], ["none"])
    assert (1.25, "none") in moves and moves[(1.25, "none")][0] < 56.0     # undoing the switch is cheaper than the wall


def test_tie_table_and_fit_and_critical():
    assert np.allclose(t54.tie_table("all_at_the_last", 1.25), [0, 1, 2, 0])
    assert np.allclose(t54.tie_table("none", 1.25), [0, 0, 0, 0])
    ks = [1, 2, 3, 4, 5]
    a, b, c = t54.fit(ks, [60 * k - 4 * k * k + 3 for k in ks])
    assert abs(a - 60) < 1e-9 and abs(b - 4) < 1e-9 and abs(c - 3) < 1e-9
    k_star, dh_star = t54.critical(a, b, c)
    assert abs(k_star - 7.5) < 1e-9 and abs(dh_star - (3 + 225)) < 1e-9
    assert t54.critical(1.0, 0.0, 0.0) == (math.inf, math.inf)
    assert all(math.isnan(v) for v in t54.fit([1, 2], [1.0, 2.0]))


def test_verdict_words():
    assert t54.verdict([56, 40, 30]) == "FIXED WALL"
    assert t54.verdict([56, 104, 144, 176]) == "NO FINITE PATCH"
    assert t54.verdict([56, 104, 100, 80]) == "CRITICAL PATCH"
    assert t54.verdict([56, 104, 100, 120]) == "RISES AGAIN"


def test_the_stage_one_config_is_the_preregistered_one():
    cfg = json.loads((ROOT / "configs" / "t54_patch_stage1.json").read_text())
    assert cfg["lengths"] == [8, 12] and cfg["ties"] == ["none", "all_at_the_last"]
    assert 1.25 in cfg["lambdas"] and 1.10 in cfg["lambdas"] and 1.40 in cfg["lambdas"]


def test_held_out_fit_on_exact_quadratic_data_and_its_ranges():
    rows = []
    for L in (8, 12):
        for k in range(1, L // 2):
            rows.append(dict(L=L, k=k, lam=1.25, tie="none", dH=60 * k - 4 * k * k, in_fit=True))
    f = t54.held_out_fit(rows)[(1.25, "none")]
    assert abs(f["k_star"] - 7.5) < 1e-9 and abs(f["dh_star"] - 225) < 1e-9 and f["verdict"] == "CRITICAL PATCH"
    assert abs(f["k_range"][0] - 6.75) < 1e-9 and abs(f["k_range"][1] - 8.25) < 1e-9      # +-10 % when the sizes agree
    rows = [dict(L=L, k=k, lam=1.1, tie="none", dH=10 * k + 0.5 * k * k, in_fit=True) for L in (8, 12) for k in range(1, L // 2)]
    assert t54.held_out_fit(rows)[(1.1, "none")]["verdict"] == "NO FINITE PATCH"
