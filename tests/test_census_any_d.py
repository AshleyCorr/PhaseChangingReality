"""The runner's census reads open space and damage at the run's own number of links (T48, 2026-09-27)."""
import importlib.util
from pathlib import Path

import numpy as np

from graphity.cqg_d import torus

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sealed_curled_d", ROOT / "scripts" / "run_sealed_curled_d.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_flat_tori_are_open_everywhere_at_two_three_and_four_directions():
    for dims in ([8, 8], [6, 6, 6], [6, 6, 8, 8]):
        adj, _ = torus(dims)
        c = runner.census(adj, 1.25)
        assert c["damaged_d"] == 0 and c["largest_open_d"] == adj.shape[0]


def test_curled_tori_have_no_open_piece_and_six_links_keep_their_old_columns():
    for dims in ([4, 48], [4, 4, 18], [4, 4, 4, 12]):
        adj, _ = torus(dims)
        c = runner.census(adj, 1.25)
        assert c["damaged_d"] == 0 and c["largest_open_d"] == 0
    adj, _ = torus([6, 6, 6])
    c = runner.census(adj, 1.25)
    assert c["largest_flat"] == c["largest_open_d"] == 216 and c["melted"] == c["damaged_d"] == 0
    adj, _ = torus([6, 6, 8, 8])                     # at eight links d = 4 is flat; the six-link column counts it as melted
    c = runner.census(adj, 1.25)
    assert c["melted"] == adj.shape[0] and c["damaged_d"] == 0


def test_a_switch_in_a_flat_sheet_is_damage_at_four_links():
    adj, _ = torus([8, 8])
    a = adj.copy()
    # swap the partners of two edges far apart (0-1 and 36-37 become 0-37 and 36-1), as a switch does
    u1, v1, u2, v2 = 0, int(adj[0, 0]), 36, int(adj[36, 0])
    for u, old, new in ((u1, v1, v2), (v1, u1, u2), (u2, v2, v1), (v2, u2, u1)):
        a[u][list(a[u]).index(old)] = new
    c = runner.census(a, 1.25)
    assert c["damaged_d"] > 0 and c["largest_open_d"] < 64
    assert int(np.sum([c["d%d" % k] for k in range(8)])) == 64
