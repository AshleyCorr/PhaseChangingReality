"""The exchange-sign rule (scripts/exact_exchange_sign.py, exact_exchange_averages.py; VISION Update 29; O57)."""
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


sign = load("exact_exchange_sign")
avg = load("exact_exchange_averages")

from graphity.cqg import NO_CAP, torus     # noqa: E402


def test_parity_of_permutations():
    assert sign.parity([0, 1, 2, 3]) == 1
    assert sign.parity([1, 0, 2, 3]) == -1           # one transposition
    assert sign.parity([1, 2, 0, 3]) == 1            # a 3-cycle
    assert sign.parity([1, 0, 3, 2]) == 1            # two transpositions
    assert sign.parity([1, 2, 3, 0]) == -1           # a 4-cycle


def test_basic_arrangements_are_allowed_and_the_sum_is_count_or_zero():
    for dims, a in (((8, 8), 256), ((16, 4), 128), ((4, 4), 192)):
        adj, part = torus(*dims, NO_CAP)
        count, odd, s = sign.signed_sum(adj, part)
        assert count == a and odd == 0 and s == a
    # the signed sum of a group with an odd element is exactly zero: half the elements are odd
    count, odd, s = 16, 8, 0
    assert s == count - 2 * odd


def test_n16_has_five_classes_and_one_is_forbidden():
    cls = avg.classes_of(8)
    assert len(cls) == 5
    forbidden = [c for c in cls if c[3]]
    assert len(forbidden) == 1 and forbidden[0][:3] == (22, 24, 16) and forbidden[0][3] == 8
    # the exchange ensemble is the interchangeable one without that class
    phi_i, _ = avg.averages(cls, 8, 20.0, 1.0, "interchangeable")
    phi_e, _ = avg.averages(cls, 8, 20.0, 1.0, "exchange")
    assert abs(phi_i - 1.3375) < 1e-4 and abs(phi_e - 1.3281) < 1e-4
