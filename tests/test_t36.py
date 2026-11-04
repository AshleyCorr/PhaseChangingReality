"""T36 reading rules (scripts/analyse_t36.py) and the square count of scripts/run_allotrope_search.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a = load("analyse_t36")
run = load("run_allotrope_search")

from graphity.cqg import NO_CAP, torus     # noqa: E402

N = 196


def test_square_counts_on_the_flat_and_curled_tori():
    adj, _ = torus(14, 14, NO_CAP)
    assert (run.square_counts(adj) == 4).all()
    adj, _ = torus(16, 4, NO_CAP)
    assert (run.square_counts(adj) == 5).all()            # a curled column adds one square at every point


def rows_for(sets, g=3.5, rep=0, start="melt"):
    return [dict(N=str(N), g=str(g), start=start, replica=str(rep), snap=str(t), low=" ".join(map(str, sorted(s))))
            for t, s in enumerate(sets)]


def test_excess_is_zero_for_fresh_sets_and_large_for_stuck_ones():
    rng = np.random.default_rng(1)
    fresh = [set(rng.choice(N, 20, replace=False).tolist()) for _ in range(60)]
    e, rho = a.excess(fresh, N, 20)
    assert abs(rho - 20 / N) < 1e-9 and abs(e) < 0.05
    stuck = [set(range(10)) | set(rng.choice(range(10, N), 10, replace=False).tolist()) for _ in range(60)]
    e, _ = a.excess(stuck, N, 20)
    assert e > 0.4
    assert a.persistent_set(stuck, 20) >= set(range(10))


def test_cell_reading_words():
    rng = np.random.default_rng(2)
    adj, _ = torus(14, 14, NO_CAP)                         # points 0..3 are consecutive along a row: a region
    stuck = {}
    fresh = {}
    for rep in range(6):
        st = [set(range(4)) | set(rng.choice(range(4, N), 16, replace=False).tolist()) for _ in range(60)]
        fr = [set(rng.choice(N, 20, replace=False).tolist()) for _ in range(60)]
        stuck[(N, 3.5, "melt", rep)] = rows_for(st, rep=rep)
        fresh[(N, 3.5, "melt", rep)] = rows_for(fr, rep=rep)
    r = a.cell_reading(stuck, 100, adj_for=lambda key, snap: adj)
    assert r["persistent"] and r["with_region"] == 6 and r["reading"] == "ALLOTROPES"
    r2 = a.cell_reading(fresh, 100, adj_for=lambda key, snap: adj)
    assert r2["reading"] == "TRANSIENT"
    assert a.verdict([r, r2]) == "ALLOTROPES" and a.verdict([r2]) == "TRANSIENT" and a.verdict([]) == "NOT READ"
