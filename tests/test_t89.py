"""T89's reader (scripts/analyse_t89.py): the crossing, the fit and the verdict words on hand-built inputs."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t89 as t89  # noqa: E402


def test_crossing_interpolates_in_ln_g_and_reports_the_bracket():
    curve = [(2.0, 0.8), (4.0, 0.6), (8.0, 0.4)]
    g, bracket = t89.crossing(curve, 0.5)
    assert bracket == (4.0, 8.0)
    assert abs(g - math.exp(0.5 * (math.log(4) + math.log(8)))) < 1e-12   # halfway in ln g
    assert t89.crossing(curve, 0.9) == (None, None)


def test_fit_recovers_an_exact_line():
    a0, b0 = -0.19, 0.07
    pts = [(n, 1.0 / (a0 + b0 * math.log(n))) for n in (196, 484, 676, 1156)]
    a, b, rms = t89.fit(pts)
    assert abs(a - a0) < 1e-9 and abs(b - b0) < 1e-9 and rms < 1e-12


def test_gate_needs_both_legs_close_and_short_tau():
    rows = [dict(g="3.0", leg="cool", phi="0.50", tau_int="10"), dict(g="3.0", leg="heat", phi="0.52", tau_int="10"),
            dict(g="3.5", leg="cool", phi="0.44", tau_int="10"), dict(g="3.5", leg="heat", phi="0.45", tau_int="10")]
    assert t89.gate(rows, (3.0, 3.5)) == (True, "ok")
    rows[1]["phi"] = "0.60"
    assert t89.gate(rows, (3.0, 3.5))[0] is False
    rows[1]["phi"] = "0.52"
    rows[2]["tau_int"] = "5000"
    assert t89.gate(rows, (3.0, 3.5))[0] is False
    assert t89.gate(rows, None) == (False, "not bracketed")


def test_verdict_words():
    assert t89.word(["inside", "inside"]) == "LOGARITHMIC DRIFT"
    assert t89.word(["below", "below"]) == "FASTER"
    assert t89.word(["above", "above"]) == "SLOWER"
    assert t89.word(["inside", "above"]) == "MIXED"
    assert t89.word(["inside", "unscored"]) == "UNSCORED"
