"""The twist-energy reader (graphity/twist_energy.py; the brief docs/design/twist_energies_2026-11-04.md). Rule 6 as it
applies to a reader: the exact numbers the brief states; a brute-force check that the parts sum; constraints
(non-negative energies, the electron at its unit); the inverse round-trips; the same census gives the same totals."""
import math

import pytest

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState
from graphity.twist_energy import DARK, ELECTRON, PROTON, STORE, SQ2, SQ6, TwistEnergy


def test_count_rule_alone():
    r = TwistEnergy(form="count")
    assert r.energy(ELECTRON) == pytest.approx(0.511)
    assert r.energy(DARK) == pytest.approx(0.511 / 3)
    assert r.ratio_store_dark() == pytest.approx(1.0)
    assert r.energy(PROTON) == pytest.approx(5508 * 0.511 / 3)


def test_rod_rule_store_is_four_darks():
    for kappa in (0.1, 1.0, 37.0):
        r = TwistEnergy(e_loop=0.0, kappa=kappa, beta=0.0)
        assert r.ratio_store_dark() == pytest.approx(4.0)
        assert r.energy(DARK) == pytest.approx(kappa * 0.25 / SQ2)


def test_rod_rule_dark_is_heaviest_per_length_by_twist():
    r = TwistEnergy(e_loop=0.0, kappa=1.0)
    dark, elec, prot = (r.twist_part(x) / x.length for x in (DARK, ELECTRON, PROTON))
    assert dark / elec == pytest.approx(6.75)
    assert dark > elec > prot


def test_parts_sum_brute_force():
    r = TwistEnergy(e_loop=0.17, kappa=2.5, beta=11.0)
    for loop in (DARK, STORE, ELECTRON, PROTON):
        by_hand = 0.17 * loop.n + 2.5 * loop.twist ** 2 / loop.length + (11.0 if loop.store else 0.0)
        assert r.energy(loop) == pytest.approx(by_hand)
        assert r.energy(loop) >= 0.0


def test_beta_for_ratio_round_trips():
    r = TwistEnergy(kappa=0.3)
    for target in (1.0, 4.0, 26.0, 1e9):
        beta = r.beta_for_ratio(target)
        assert TwistEnergy(kappa=0.3, beta=beta).ratio_store_dark() == pytest.approx(target)
    # her billion at the count rule's dark particle: about 170 TeV
    assert TwistEnergy(form="count").beta_for_ratio(1e9) == pytest.approx(1e9 * 0.511 / 3 - 0.511 / 3)


def test_bad_form_refused():
    with pytest.raises(ValueError):
        TwistEnergy(form="spring")


def _run(seed):
    adj, part = torus(12, 12, NO_CAP)
    return PairMoves(CurlState(adj, part), lam=1.5, barrier=0.0, seed=seed, price_from="gravity",
                     clock="simultaneous").run()


def test_sheet_totals_exact_and_same_seed():
    a, b = _run(7), _run(7)
    r = TwistEnergy(kappa=1.0, beta=3.0)
    sa, sb = r.sheet(a), r.sheet(b)
    assert sa == sb
    assert sa["stores"] == a["stores"] and sa["dark_particles"] == a["frozen_curls"]
    assert sa["dark_total_mev"] == pytest.approx(a["frozen_curls"] * r.energy(DARK))
    assert sa["store_total_mev"] == pytest.approx(a["stores"] * r.energy(STORE))
    assert sa["stores_per_pair"] == pytest.approx(a["stores"] / a["pairs_made"])
    assert math.isfinite(sa["store_to_dark_total"])
