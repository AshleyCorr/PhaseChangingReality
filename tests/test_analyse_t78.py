"""scripts/analyse_t78.py: the estimators on synthetic rows with known answers (PREREGISTRATION T78).

With H constant, E_X = 0 for both objects, so Delta = S_patch - S_ring = (F_ring - F_patch)/g = ln(<n_patch>/<n_ring>)
exactly. Identical counts give Delta = 0 and NO GAP; a patch count twice the ring count gives Delta = ln 2 at every
coupling, hence CURL PENALISED, FLAT; a count ratio that grows with the deficit gives CURL PENALISED, GROWING; fewer
than 50 sightings gives NOT READ.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t78 as a78  # noqa: E402


def cells_from(ratio_by_g, n_rep=16, n_snap=200, base=0.5, seed=1):
    """Rows (N, H, D, n_ring, n_patch): rings Poisson(base), patches Poisson(base * ratio); H constant; D grows with g."""
    rng = np.random.default_rng(seed)
    cells = {}
    for g, ratio in ratio_by_g.items():
        cells[g] = {}
        for rep in range(n_rep):
            rows = []
            for _ in range(n_snap):
                rows.append((196, 1000.0, int(196 * g), int(rng.poisson(base)), int(rng.poisson(base * ratio))))
            cells[g][rep] = rows
    return cells


def test_identical_counts_give_no_gap():
    cells = cells_from({3.3: 1.0, 3.7: 1.0})
    for g in cells:
        for rep in cells[g]:
            cells[g][rep] = [(n, h, d, r, r) for n, h, d, r, _ in cells[g][rep]]
    table, word = a78.read(cells)
    assert all(t["delta"] == 0.0 for t in table)
    assert word == "NO GAP"


def test_constant_ratio_two_is_ln2_and_flat():
    cells = cells_from({3.3: 2.0, 3.5: 2.0, 3.7: 2.0}, n_snap=400)
    table, word = a78.read(cells)
    for t in table:
        r, p = t["ring"]["mean"], t["patch"]["mean"]
        assert math.isclose(t["delta"], math.log(p / r), rel_tol=1e-12)     # exact given the sample means
        assert abs(t["delta"] - math.log(2)) < 3 * t["sigma"]
        assert t["ring"]["E"] == 0.0 and t["patch"]["E"] == 0.0
    assert word in ("CURL PENALISED, FLAT", "MIXED")


def test_growing_ratio_is_growing():
    cells = cells_from({3.3: 1.5, 3.5: 4.0, 3.7: 12.0}, n_snap=400)
    table, word = a78.read(cells)
    assert word == "CURL PENALISED, GROWING"


def test_too_few_sightings_is_not_read():
    cells = cells_from({3.3: 1.0, 3.7: 1.0}, n_snap=5, base=0.1)
    table, word = a78.read(cells)
    assert all(not t["read"] for t in table)
    assert word == "NOT READ"
