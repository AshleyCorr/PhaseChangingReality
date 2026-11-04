"""Known-answer tests for the post-hoc checks of paper 1 (scripts/analyse_paper_stats.py) and the exact
count of the ways out of the curled torus (scripts/exact_torus_level.py)."""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyse_paper_stats as ps       # noqa: E402
import exact_torus_level as lvl        # noqa: E402
from graphity.cqg import NO_CAP, torus  # noqa: E402


def test_tau_is_eq2_of_the_paper():
    assert round(ps.tau_eq2(1.05, 1.5)) == 8330 and round(ps.tau_eq2(1.25, 1.5)) == 845


def test_ks_distance_matches_scipy():
    x = np.random.default_rng(3).exponential(1.0, 57)
    assert abs(ps.ks_to_unit_exponential(x) - stats.kstest(x, "expon").statistic) < 1e-12


def test_recording_rounds_up_to_the_sweep_or_the_block():
    x = np.array([0.2, 4.9, 5.0, 5.1])
    assert list(ps.record(x, "sweep")) == [1, 5, 5, 6]
    assert list(ps.record(x, "block")) == [5, 5, 5, 10]


def test_bootstrap_passes_exponential_times_and_fails_regular_ones():
    rng = np.random.default_rng(0)
    expo = [rng.exponential(m, 30) for m in (100, 1000, 5000)]
    regular = [rng.gamma(6.0, m / 6.0, 30) for m in (100, 1000, 5000)]     # CV 0.41: a schedule, not a clock
    assert ps.bootstrap_p(expo, "exact", n_boot=500)[1] > 0.05
    assert ps.bootstrap_p(regular, "exact", n_boot=500)[1] < 0.01


def _row(start, cols, half=8.0):
    return dict(N="64", rings="1", dist="nan", half_length=str(half), start=str(start),
                ring_cols=" ".join(str(c) for c in cols))


def test_rotation_test_sees_a_leftover_that_sits_at_the_start():
    rows = [_row(s, [s, (s + 1) % 16]) for s in range(0, 16, 2)] * 3     # leftover always half a column away
    obs, null, p, k = ps.rotation_test(rows, n_perm=5000)
    assert k == 24 and abs(obs - 0.5) < 1e-9 and abs(null - 4.0) < 1e-9 and p < 0.001


def test_rotation_test_is_quiet_when_the_leftover_goes_everywhere():
    rows = [_row(0.0, [k, (k + 1) % 16]) for k in range(16)]              # every position once
    obs, null, p, _ = ps.rotation_test(rows, n_perm=5000)
    assert abs(obs - null) < 1e-9 and p > 0.9


def test_the_chain_offers_move_a_three_times_and_move_b_twice_per_sweep():
    adj, part = torus(4, 8, cap=NO_CAP)
    n = len(adj)
    c = lvl.ordered_census(adj, np.flatnonzero(part == 0))
    assert c[lvl.A] == 6 * n and c[lvl.B] == 4 * n and c[lvl.NEUTRAL] == n
    assert 2 * n * c[lvl.A] / (4 * n * n) == 3 and 2 * n * c[lvl.B] / (4 * n * n) == 2


def test_neutral_switches_twist_the_torus_without_changing_its_ways_out():
    adj, part = torus(4, 8, cap=NO_CAP)
    side_u = np.flatnonzero(part == 0)
    base, neutral = lvl.census(adj, side_u)
    assert len(neutral) == len(adj) // 2
    assert all(lvl.is_column_reflection(m, 8) for _, m in neutral)
    g0 = lvl.as_graph(adj)
    twisted, _ = neutral[0]
    assert not nx.is_isomorphic(g0, lvl.as_graph(twisted))
    after, _ = lvl.census(twisted, side_u)
    for key in (lvl.A, lvl.B, lvl.NEUTRAL):
        assert after[key] == base[key]
