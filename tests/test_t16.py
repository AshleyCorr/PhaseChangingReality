"""T16's reading rules on rows whose answers are known (scripts/analyse_t16.py)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyse_t16", Path(__file__).resolve().parents[1] / "scripts" / "analyse_t16.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def row(g, replica=0, leg="", phi=0.5, chi=0.1, xi=0.3, err=0.01, trips=9, swap=0.3, snaps=100, skip=0):
    return dict(g=str(g), replica=str(replica), leg=leg, phi=str(phi), chi=str(chi), xi_over_diam=str(xi),
                xi_over_diam_err=str(err), round_trips=str(trips), swap_rate=str(swap), snapshots=str(snaps),
                snap_no_fluct=str(skip), snap_no_xi="0")


def test_e_gate_needs_trips_and_swap_rates_inside_the_range():
    assert a.e_gate([row(5, 0), row(4, 1)])[0]
    assert not a.e_gate([row(5, 0, trips=4), row(4, 1)])[0]
    assert not a.e_gate([row(5, 0, swap=0.7), row(4, 1)])[0]          # above the range fails, as written


def test_p_is_read_only_where_the_legs_agree():
    rows = [row(6, leg="cool", phi=0.40), row(6, leg="heat", phi=0.41),
            row(3, leg="cool", phi=0.70), row(3, leg="heat", phi=0.95)]
    assert a.read_couplings_p(rows) == [6.0]


def test_pooling_averages_rows_and_combines_errors():
    c = a.pooled([row(5, 0, xi=0.2, err=0.03), row(5, 1, xi=0.4, err=0.04)])[5.0]
    assert abs(c["xi"] - 0.3) < 1e-12 and abs(c["xi_err"] - 0.025) < 1e-12


def test_peak_reports_the_edge():
    curve = {g: dict(chi=v, xi=v, xi_err=0.0) for g, v in ((9, 0.1), (6, 0.3), (3, 0.2))}
    assert a.peak(curve, [9, 6, 3], "chi")[:2] == (6, 0.3) and not a.peak(curve, [9, 6, 3], "chi")[3]
    assert a.peak(curve, [9, 6], "chi")[3]                             # largest at the last read coupling


def test_tendency_rules():
    rising = {100: (5, 0.2, 0.01, False, 0.0), 196: (5, 0.3, 0.01, False, 0.0)}
    assert a.tendency(rising) == "DIVERGENT TENDENCY"
    flat = {100: (5, 0.2, 0.05, False, 0.0), 196: (5, 0.21, 0.05, False, 0.0)}
    assert a.tendency(flat).startswith("NONE")
    edge = {100: (5, 0.2, 0.01, True, 0.0), 196: (5, 0.3, 0.01, False, 0.0)}
    assert a.tendency(edge).startswith("NOT READABLE")
    skipped = {100: (5, 0.2, 0.01, False, 0.6), 196: (5, 0.3, 0.01, False, 0.0)}
    assert a.tendency(skipped).startswith("NOT READABLE")
    assert a.tendency({196: (5, 0.3, 0.01, False, 0.0)}).startswith("NOT READABLE")
