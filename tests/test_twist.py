"""The twisted-diagonal state against the owner's sheet: her four test particles, the checks the form promises, and net
twist under pair creation (docs/design/twisted_diagonals_build_2026-10-30.md, sections 3 and 5)."""
from fractions import Fraction

import pytest

from graphity.twist import BOX_STEPS, FACE_STEPS, HALF, THIRD, Loop, Particle, mirror, net_twist, step_kind, triangle

L = 8


def test_the_step_kinds_are_what_the_sheet_says():
    assert len(BOX_STEPS) == 24 and len(FACE_STEPS) == 12
    assert step_kind((1, 1, 2)) == "box" and step_kind((-2, 1, -1)) == "box"
    assert step_kind((1, -1, 0)) == "face"
    assert step_kind((1, 1, 4)) == "long" and step_kind((4, -1, 1)) == "long"
    assert step_kind((1, 1, 1)) is None and step_kind((1, 1, 3)) is None and step_kind((2, 2, 0)) is None


def test_a_triangle_of_three_box_diagonals_closes_and_is_one_colour():
    e = triangle(L)
    assert e.kind == "box" and len(e.points) == 3
    assert len({sum(p) % 2 for p in e.points}) == 1


def test_her_four_test_particles():
    electron = Particle("electron", [triangle(L)])
    assert (electron.charge, electron.spin, electron.mass_count) == (Fraction(-1), HALF, 3)
    assert electron.check_whole_charge()
    pion = Particle("pion+", [triangle(L, (0, 0, 0), (2 * THIRD, 0, 0), spin=HALF),
                              triangle(L, (2, 2, 0), (THIRD, 0, 0), spin=-HALF)])
    assert (pion.charge, pion.spin) == (Fraction(1), Fraction(0))
    neutron = Particle("neutron", [triangle(L, (0, 0, 0), (-THIRD, 0, 0), spin=HALF),
                                   triangle(L, (2, 2, 0), (-THIRD, 0, 0), spin=-HALF),
                                   triangle(L, (4, 4, 0), (2 * THIRD, 0, 0), spin=HALF)])
    assert (neutron.charge, neutron.spin) == (Fraction(0), HALF)
    neutrino = Particle("neutrino", [triangle(L, (0, 0, 0), (THIRD, -THIRD, 0), spin=HALF)])
    assert (neutrino.charge, neutrino.spin, neutrino.parts["tau_plus"]) == (Fraction(0), HALF, 2 * THIRD)
    assert electron.parts == dict(L=1, n=3, tau_plus=Fraction(1), T=Fraction(-1), s=HALF, C=1, orb=0)


def test_the_checks_the_form_promises():
    with pytest.raises(ValueError):                                   # not a diagonal
        Loop(L, (0, 0, 0), [(1, 1, 1), (-1, -1, -1)], [0, 0])
    with pytest.raises(ValueError):                                   # does not close
        Loop(L, (0, 0, 0), [(1, 1, 2), (1, 1, 2)], [0, 0])
    with pytest.raises(ValueError):                                   # twist not in thirds
        triangle(L, twists=(HALF, 0, 0))
    with pytest.raises(ValueError):                                   # twist on a long diagonal
        Loop(L, (0, 0, 0), [(1, 1, 4), (-1, -1, -4)], [THIRD, 0])
    with pytest.raises(ValueError):                                   # spin without twist present
        triangle(L, twists=(0, 0, 0), spin=HALF)
    with pytest.raises(ValueError):                                   # mixed kinds
        Loop(L, (0, 0, 0), [(1, 1, 2), (1, -1, 0), (-2, 0, -2)], [0, 0, 0])
    with pytest.raises(ValueError):                                   # a third of charge is not a particle on its own
        Particle("quark", [triangle(L, twists=(THIRD, 0, 0), spin=HALF)]).check_whole_charge()


def test_a_long_diagonal_counts_one_of_mass_and_a_loop_closes_on_the_torus():
    gluon = Loop(L, (0, 0, 0), [(1, 1, 4), (-1, -1, 4)], [0, 0])     # closes through the wrap: (0, 0, 8) = 0 mod 8
    assert gluon.kind == "long" and gluon.mass_count == 2 and gluon.spin == 0


def test_pair_creation_conserves_net_twist():
    e = triangle(L)
    assert net_twist([Particle("e-", [e]), Particle("e+", [mirror(e)])]) == 0
    assert net_twist([Particle("e-", [e])]) == -1


def test_writhe_is_zero_for_the_flat_triangle_and_flips_with_the_mirror_image():
    from graphity.twist import twist_energy, writhe
    assert abs(writhe(triangle(L))) < 1e-9
    turn = [(2, 1, 1), (-1, 2, 1), (-2, -1, 1), (1, -2, 1)]
    helix = turn + [(2, 1, 1), (2, 1, -1)] + [tuple(-x for x in s) for s in reversed(turn)] + [(-2, -1, -1), (-2, -1, 1)]
    # sandbox/coil_writhe.py's coil: a turn rising 4, a bridge, the turn traversed backwards (same handedness), the bridge back
    coil = Loop(L, (0, 0, 0), helix, [0] * len(helix))
    mirror_coil = Loop(L, (0, 0, 0), [(-a, b, c) for a, b, c in helix], [0] * len(helix))
    w, wm = writhe(coil), writhe(mirror_coil)
    assert abs(w) > 0.05 and abs(w + wm) < 1e-9
    assert twist_energy(triangle(L), eps=1.0, eps_w=0.0) == pytest.approx(1 / 3)       # three equal thirds: elastic only
    assert twist_energy(triangle(L, twists=(2 * THIRD, THIRD, 0)), eps=1.0, eps_w=0.0) == pytest.approx(5 / 9 + 6 / 9)


def test_a_winding_loop_has_no_writhe():
    from graphity.twist import writhe
    assert writhe(Loop(L, (0, 0, 0), [(1, 1, 4), (-1, -1, 4)], [0, 0])) is None


def test_the_photon_as_she_stated_it_and_her_annihilation_ledger():
    from graphity.twist import Photon, annihilate, ledger
    with pytest.raises(ValueError):                                   # not a body diagonal
        Photon(L, (0, 0, 0), (1, 1, 2))
    with pytest.raises(ValueError):                                   # a whole curl is a loop's, not a photon's
        Photon(L, (0, 0, 0), (1, 1, 1), curl=1)
    with pytest.raises(ValueError):                                   # spin 1 only
        Photon(L, (0, 0, 0), (1, 1, 1), spin=HALF)
    c = Photon(L, (0, 0, 0), (1, 1, 1))
    assert (c.charge, c.net_twist, c.twist_present, c.curl, c.spin, c.mass_count) == (0, 0, 0, HALF, 1, 0)   # massless: its diagonals are not matter's
    assert [c.advance(k).angle_degrees for k in range(6)] == [0, 90, 180, 270, 0, 90]      # repeats every four links
    assert [c.advance(k).colour for k in range(4)] == ["black", "white", "black", "white"]  # the two-colour diagonal
    assert c.advance(3).head == (3, 3, 3)
    e = triangle(L, twists=(THIRD, THIRD, THIRD), spin=HALF)                                  # her ledger's A: +1 twist, +1 loop, 1/2 spin
    electron, positron = Particle("A", [e]), Particle("B", [mirror(e)])
    assert positron.parts["T"] == -1 and positron.parts["C"] == -1                            # B: -1 twist, -1 loop
    before = ledger([electron, positron])
    after = ledger(annihilate(electron, positron))
    assert before["T"] == after["T"] == 0 and before["C"] == after["C"] == 0 and before["s"] == after["s"] == 0
    assert (before["n"], after["n"]) == (6, 0)                                               # the mass count becomes the spin and curl the photons carry


def test_signed_curl_lets_a_particle_net_zero_curl():
    nu = Particle("her neutrino", [triangle(L, twists=(THIRD, -THIRD, 0), spin=HALF, curl=1),
                                    triangle(L, (2, 2, 0), (THIRD, -THIRD, 0), spin=0, curl=-1)])
    assert (nu.charge, nu.spin, nu.parts["C"]) == (0, HALF, 0)
    with pytest.raises(ValueError):
        triangle(L, curl=0)


def test_an_excited_electron_holds_the_photon_and_gives_it_back():
    from graphity.twist import Photon, absorb, emit, ledger
    e = Particle("e-", [triangle(L)])
    photon = Photon(L, (0, 0, 0), (1, 1, 1))
    before = ledger([e, photon])
    ex = Particle("e-*", [absorb(e.loops[0], photon)])
    assert (ex.spin, ex.parts["orb"], ex.parts["C"], ex.charge, ex.mass_count) == (HALF, 1, Fraction(3, 2), -1, 3)
    during = ledger([ex])
    keep = lambda d: {k: d[k] for k in ("T", "C", "J", "n")}                      # the photon's spin became orbital: s and orb trade, J holds
    assert keep(during) == keep(before) and (before["s"], before["orb"], during["s"], during["orb"]) == (Fraction(3, 2), 0, HALF, 1)
    lp, out = emit(ex.loops[0], direction=(-1, -1, 1))
    after = ledger([Particle("e-", [lp]), out])
    assert after == before and lp.held == [] and (out.curl, out.spin) == (HALF, 1)
    with pytest.raises(ValueError):
        emit(lp)
    with pytest.raises(ValueError):                                                # an excitation holds a photon's values only
        triangle(L, held=[(1, 1)]) if False else Loop(L, (0, 0, 0), list(triangle(L).steps), [-THIRD] * 3, spin=HALF, held=[(1, 1)])
    assert mirror(ex.loops[0]).held == [(-HALF, -1)]


def test_the_twist_constant_may_differ_by_kind_and_the_dark_pair_wins_below_four_ninths():
    from fractions import Fraction
    from graphity.twist import loop_kind, particle_energy, twist_energy
    face = Loop(L, (0, 0, 0), [(1, 1, 0), (-1, 0, 1), (0, -1, -1)], [Fraction(1, 2), 0, 0], curl=1, spin=HALF)   # three faces of one cube
    box = triangle(L, twists=(THIRD, 0, 0), spin=HALF)
    assert (loop_kind(face), loop_kind(box)) == ("face", "box")
    assert twist_energy(face, eps=1.0, eps_w=0.0) == 0.75 and twist_energy(box, eps=1.0, eps_w=0.0) == pytest.approx(1 / 3)
    assert twist_energy(face, eps=1.0, eps_w=0.0, eps_face=0.1) == pytest.approx(0.075)
    assert twist_energy(box, eps=1.0, eps_w=0.0, eps_face=0.1) == pytest.approx(1 / 3)            # the box constant is untouched
    dark = Particle("dark pair", [face, mirror(face)])
    light = Particle("box pair", [box, mirror(box)])
    assert (dark.charge, dark.parts["C"], light.charge, light.parts["C"]) == (0, 0, 0, 0)
    for ratio, wins in ((0.4, True), (4 / 9, False), (0.5, False)):
        d = particle_energy(dark, eps=1.0, eps_w=0.0, eps_face=ratio)
        assert (d < particle_energy(light, eps=1.0, eps_w=0.0) - 1e-12) is wins


def test_the_photon_ladder_of_31_october_repeat_n_curl_two_over_n_energy_as_frequency():
    """Her decision of 31 October: the curl angle may turn less than a quarter turn per link, so the repeat is any whole
    n >= 4 and the curl per link is 2/n; the energy is the curl per link (the reading that matches measurement), not curl
    times length; the ledger and the excitation carry the period through."""
    from graphity.twist import Photon, Loop, absorb, emit, annihilate, ledger, photon_curl_allowed, triangle
    L = 6
    four = Photon(L, (0, 0, 0), (1, 1, 1))
    assert four.period == 4 and four.curl == HALF and four.energy == HALF and four.angle_degrees == 0
    assert four.advance(3).angle_degrees == 270 and four.advance(4).angle_degrees == 0
    eight = Photon(L, (0, 0, 0), (1, 1, 1), period=8)
    assert eight.curl == Fraction(1, 4) and eight.energy == Fraction(1, 4)
    assert eight.advance(4).angle_degrees == 180 and eight.advance(8).angle_degrees == 0
    five = Photon(L, (0, 0, 0), (1, 1, 1), curl=Fraction(-2, 5))
    assert five.period == 5 and five.energy == Fraction(2, 5)
    # a flatter photon of twice the length carries less energy, not the same: energy is per link
    assert Photon(L, (0, 0, 0), (1, 1, 1), length=8, period=8).energy < four.energy
    for bad in (Fraction(3, 4), Fraction(1), Fraction(2, 3), Fraction(3, 7)):   # 2/n for no whole n >= 4
        assert not photon_curl_allowed(bad)
    assert photon_curl_allowed(Fraction(1, 3)) and photon_curl_allowed(Fraction(1, 5)) and photon_curl_allowed(Fraction(-2, 7))
    with pytest.raises(ValueError):
        Photon(L, (0, 0, 0), (1, 1, 1), period=3)
    with pytest.raises(ValueError):
        Photon(L, (0, 0, 0), (1, 1, 1), curl=Fraction(3, 4))
    e = triangle(L)
    held = absorb(e, eight)
    assert held.held == [(Fraction(1, 4), Fraction(1))]
    back, out = emit(held)
    assert out.period == 8 and out.energy == Fraction(1, 4) and back.held == []
    with pytest.raises(ValueError):
        Loop(L, (0, 0, 0), list(e.steps), list(e.twists), held=[(Fraction(3, 4), Fraction(1))])
    from graphity.twist import Particle, mirror
    el = Particle("electron", [e]); po = Particle("positron", [mirror(e)])
    c, d = annihilate(el, po, period=10)
    assert c.period == d.period == 10 and c.curl == -d.curl == Fraction(1, 5)
    before, after = ledger([el, po]), ledger([c, d])
    assert before["T"] == after["T"] == 0 and before["C"] == after["C"] == 0 and before["J"] == after["J"]


def test_charge_is_carried_only_on_thirds():
    """Her rule of 31 October: a charge needs two sides and can only be applied on thirds. A lone one-sided face loop
    (one half twist) has charge 0 and is a particle on its own; the same lone third on a box loop is still refused."""
    from fractions import Fraction
    one_sided = Loop(L, (0, 0, 0), [(1, 1, 0), (-1, 0, 1), (0, -1, -1)], [Fraction(1, 2), 0, 0], curl=1, spin=HALF)
    dark = Particle("one-sided dark loop", [one_sided])
    assert one_sided.net_twist == Fraction(1, 2) and dark.charge == 0 and dark.parts["T"] == 0
    assert dark.check_whole_charge()
    with pytest.raises(ValueError):
        Particle("quark", [triangle(L, twists=(THIRD, 0, 0), spin=HALF)]).check_whole_charge()
