"""Two defects in a sheet are exactly additive unless they touch (ASSUMPTIONS O22).

Why this is a theorem and not only a measurement: H = 4 sum over edges of [(2 - S_e) + lam (S_e - 2)+],
a sum of per-edge terms each depending only on the squares through that one edge. Two defects that
break or make no square in common therefore contribute independently, and their energies add exactly.
No long-range force between leftovers is possible at fixed wiring, whatever the defects are made of.
`scripts/run_defect_interaction.py` measures it at every separation on a 10 x 10 torus and finds
exactly that: -16 where they touch, and 0.0 at every greater separation, at lambda = 1 and 1.25.

The quadruples below are taken from that run: the cheapest switch out of the flat sheet costs 32,
and the pair is the same switch translated around the torus.
"""
from graphity.cqg import NO_CAP, _switch, hamiltonian, is_valid, torus

LX = LY = 10
LAM = 1.25
DEFECT = (0, 10, 1, 11)              # the cheapest valid switch out of the flat 10 x 10 sheet
FAR = (55, 65, 56, 66)               # the same switch translated by (5, 5): no square in common
TOUCHING = (2, 12, 3, 13)            # translated by (0, 2): the damage overlaps


def apply(adj, quad):
    before = hamiltonian(adj, LAM)
    _switch(adj, *quad)
    assert is_valid(adj, NO_CAP)
    return hamiltonian(adj, LAM) - before


def test_one_defect_costs_32():
    flat, _ = torus(LX, LY, NO_CAP)
    assert apply(flat.copy(), DEFECT) == 32.0


def test_two_defects_far_apart_cost_exactly_twice_one():
    """The additivity the whole reading rests on, and it is exact rather than approximate."""
    flat, _ = torus(LX, LY, NO_CAP)
    pair = flat.copy()
    first = apply(pair, DEFECT)
    second = apply(pair, FAR)
    assert first == second == 32.0
    assert hamiltonian(pair, LAM) - hamiltonian(flat, LAM) == 64.0


def test_defects_that_touch_cost_less_together():
    """Where the damage overlaps they share the broken squares and pay once: an attraction of 16,
    which is half of one defect, and the only interaction there is."""
    flat, _ = torus(LX, LY, NO_CAP)
    pair = flat.copy()
    apply(pair, DEFECT)
    second = apply(pair, TOUCHING)
    assert second == 16.0                                  # cheaper beside the first than alone
    assert hamiltonian(pair, LAM) - hamiltonian(flat, LAM) == 48.0 == 2 * 32 - 16


def test_a_far_defect_changes_nothing_about_the_near_one():
    """The locality that makes the additivity a theorem: the second defect leaves the first's
    wiring and its share of the energy untouched."""
    flat, _ = torus(LX, LY, NO_CAP)
    a = flat.copy()
    apply(a, DEFECT)
    b = a.copy()
    apply(b, FAR)
    assert (b[:2] == a[:2]).all() and (b[10:12] == a[10:12]).all()
