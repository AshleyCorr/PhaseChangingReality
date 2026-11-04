"""scripts/read_column_wiring.py: the wiring of the four-point leftover. A built tube first, then one saved graph of
T51 whose single column was read by hand on 14 October 2026."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import read_column_wiring as r                                     # noqa: E402
from run_sixloop_widen import seam_shifted                         # noqa: E402


def test_clean_tubes_have_no_column():
    assert r.columns(seam_shifted(24, 2)[0]) == []                  # six around, flat everywhere
    assert r.columns(seam_shifted(24, 0)[0]) == []                  # the curled tube is one piece of 96, not four


def test_a_ring_of_the_curled_tube_cuts_it():
    adj = seam_shifted(24, 0)[0]
    ring = [x * 24 for x in range(4)]                               # (x, 0) for x = 0..3: one ring around the tube
    assert r.groups(adj, ring) == 2


def test_the_saved_column():
    adj = np.load(ROOT / "results" / "t51_l256_100k_00_adj" / "N1024_rep0_tc100000.npz")["adj"]
    cols = r.columns(adj)
    assert [sorted(c) for c in cols] == [[256, 257, 258, 259]]
    assert r.pattern(adj, cols[0]) == ((2, 2, 3, 3), (2, 2, 2, 2, 3, 3, 3, 3), True)
    assert r.groups(adj, cols[0]) == 1
