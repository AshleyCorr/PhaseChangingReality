"""scripts/read_leftover_width.py: the tube runs' end states that are not exactly flat, read from the points
clear of flaws. Built graphs first, then one saved cell of paper 1's fourth map."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import read_clear_points as c                                     # noqa: E402
import read_leftover_width as w                                   # noqa: E402
from exact_sixloop_walls import switches                          # noqa: E402
from graphity.cqg import NO_CAP, _switch, is_valid                # noqa: E402
from run_sixloop_widen import balls, seam_shifted                 # noqa: E402


def test_the_three_clean_tubes():
    none = (False, False)
    assert w.read(seam_shifted(24, 0)[0]) == (w.STILL, 0.0, 0, 0, 0) + none   # the curled tube: every link flawed
    assert w.read(seam_shifted(24, 2)[0]) == (w.FLAT, 0.0, 96, 0, 96) + none  # six around: every point sees 22
    assert w.read(seam_shifted(24, 4)[0]) == (w.FLAT, 1.0, 96, 96, 0) + none  # eight around: every point sees 25
    assert [w.word(s) for s in (0.0, 0.049, 0.05, 0.499, 0.5)] == ["TUBE", "TUBE", "PATCHWORK", "PATCHWORK", "WIDE"]


def with_one_flaw(shift):
    adj, side = seam_shifted(24, shift)
    move = min(switches(adj, side), key=lambda m: (-16 * m[1] + 5 * m[2], m[0]))   # the cheapest at lambda = 1.25
    _switch(adj, *move[0])
    assert is_valid(adj, NO_CAP)
    return adj


def test_a_flaw_changes_the_counts_near_it_and_the_clear_points_do_not_see_it():
    six, eight = with_one_flaw(2), with_one_flaw(4)
    assert w.read(six)[:5] == (w.LEFTOVER, 0.0, 64, 0, 64)    # one flaw in a tube six around: the clear points see 22
    kind, share, clear, wide, narrow, both, touch = w.read(eight)
    assert (both, touch) == (False, False)
    assert (kind, clear, wide, narrow) == (w.LEFTOVER, 44, 44, 0)             # in a tube eight around: they see 25
    dist = w.flaw_distance(eight)
    assert int(np.sum(dist == 0)) == 8 and int(np.sum(dist <= w.REACH)) == 52
    near = set(balls(eight)[:, 0][dist <= w.REACH].tolist())
    assert near == {25, 26, 28}                               # beside the flaw the count is not a width
    assert abs(share - 80 / 96) < 1e-12                       # which is why sees3 alone is not read as one
    flat = seam_shifted(24, 2)[0]
    assert np.all(w.flaw_distance(flat) == -1) and len(w.clear_counts(flat)) == 96


def test_one_saved_cell_of_the_fourth_map():
    cells = w.table(str(ROOT / "results"), ("t60_lam125_n96_adj",))
    assert list(cells) == [("t60", "125", 96)]
    s = w.summary(cells[("t60", "125", 96)])
    assert (s["states"], s["flat"], s["flat_tube"], s["still"], s["leftover"]) == (120, 102, 102, 0, 18)
    assert dict(s["words"]) == {"WIDE": 13, "PATCHWORK": 4, "TUBE": 1}
    assert (s["clear"], s["clear25"], s["clear22"]) == (550, 441, 108)
    assert (s["some_wide"], s["mostly_wide"], s["all_six"], s["no_clear"]) == (14, 13, 4, 0)
    assert (s["all_wide"], s["both"]) == (12, 1)             # 12 of the 18 have no loop of six away from their flaws
    assert (s["two_widths"], s["touching"]) == (1, 0)
    longer = w.summary(w.table(str(ROOT / "results"), ("t60_lam125_n192_adj",))[("t60", "125", 192)])
    assert (longer["leftover"], longer["all_wide"], longer["all_six"], longer["both"]) == (22, 2, 8, 12)
    assert (longer["two_widths"], longer["touching"]) == (12, 0)   # the two widths never meet except across a flaw


def test_the_report_runs_on_a_folder_with_nothing_in_it(tmp_path, capsys):
    assert w.main(str(tmp_path)) == 1
    assert "no saved end states" in capsys.readouterr().out
    assert c.main(str(tmp_path)) == 1 and c.cell(str(tmp_path), "t64_mu0100_n96") is None


def test_clear_points_of_two_saved_cells_of_t62_and_t64():
    results = str(ROOT / "results")
    control = c.cell(results, "t62_mu000_g150_n96")           # no price: every tube stayed exactly flat and six around
    assert (control["states"], control["points"], control["clear"], control["see22"], control["all22"]) == (
        24, 2304, 2304, 2304, 24)
    wide = c.cell(results, "t64_mu0100_n96")                  # a cell that reads WIDENS: flawed nearly everywhere
    assert (wide["states"], wide["points"], wide["clear"], wide["see25"], wide["see22"]) == (24, 2304, 277, 276, 0)
    assert (wide["with_clear"], wide["all25"], wide["all22"], wide["two_widths"], wide["touching"]) == (22, 21, 0, 0, 0)
    assert len(c.T62) == 30 and len(c.T64) == 15              # a named set of runs, three sizes each
