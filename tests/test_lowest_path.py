"""T69 (PREREGISTRATION T69): the local energy of a switch against the full energy, the slab's cheapest move against
O41, and the search's bookkeeping on a case small enough to check by hand."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import exact_lowest_path as lp                                   # noqa: E402
from graphity.cqg_d import _switch, hamiltonian, is_valid, torus  # noqa: E402


def test_local_delta_equals_full_recomputation_on_the_slab_and_on_a_disturbed_slab():
    rng = np.random.default_rng(3)
    for lam in (1.25, 1.4):
        adj, part = torus([4, 6, 6])
        side0 = np.flatnonzero(part == 0).astype(np.int64)
        for _round in range(3):
            window = np.sort(rng.choice(side0, size=10, replace=False))
            u1s, v1s, u2s, v2s, dhs = lp.expand(adj, window, lam)
            assert len(dhs) > 0
            h0 = hamiltonian(adj, lam)
            for idx in rng.choice(len(dhs), size=min(40, len(dhs)), replace=False):
                child = adj.copy()
                _switch(child, int(u1s[idx]), int(v1s[idx]), int(u2s[idx]), int(v2s[idx]))
                assert is_valid(child)
                assert abs(hamiltonian(child, lam) - h0 - dhs[idx]) < 1e-9
            # disturb the slab by one cheap valid move and check again from there
            i = int(np.argmin(dhs))
            _switch(adj, int(u1s[i]), int(v1s[i]), int(u2s[i]), int(v2s[i]))
            assert is_valid(adj)


def test_expand_lists_every_valid_switch_once_and_the_slab_wall_is_o41():
    adj, part = torus([4, 6, 6])
    side0 = np.flatnonzero(part == 0).astype(np.int64)
    u1s, v1s, u2s, v2s, dhs = lp.expand(adj, side0[:12], 1.25)
    assert len(set(zip(u1s.tolist(), v1s.tolist(), u2s.tolist(), v2s.tolist()))) == len(dhs)
    assert all(u1 < u2 for u1, u2 in zip(u1s, u2s))
    # O41: the slab's cheapest move costs 96 - 48 lam = 36 at 1.25; from a whole window the minimum is that
    full = lp.expand(adj, side0, 1.25)[4]
    assert abs(full.min() - 36.0) < 1e-9


def test_search_bookkeeping_on_a_tiny_budget():
    table, table_b, exhausted, popped, seen, e0 = lp.search(6, 1.25, 1, 3, [1, 4, 8])
    assert popped == 3 and seen > 3
    assert abs(e0 - 144.0) < 1e-9                      # the 6 x 6 x 4 slab holds 4(lam - 1) N = 144 above flat
    assert table_b[1] is not None and abs(table_b[1][0] - 36.0) < 1e-9   # the first opened points come at the single wall
    assert table_b[1][3] > 0                            # and that cheapest move breaks points (O130's swapped pairs)
    assert table[8] is None                             # three states cannot open eight points cleanly
    assert exhausted <= 36.0 + 1e-9
