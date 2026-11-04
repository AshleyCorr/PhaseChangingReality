"""Checks on the energy function. Run with: pytest"""
import math

import networkx as nx
import numpy as np

from graphity.energy import ORIENTATIONS, cycle_census, cycle_weights, total_energy
from graphity.graphs import konopka_ground_state, random_regular_connected, to_adj


def brute_force_census(g, l_max):
    """Independent cycle count using networkx (slow, for testing only)."""
    counts = np.zeros(l_max + 1, dtype=np.int64)
    for cyc in nx.simple_cycles(g, length_bound=l_max):
        counts[len(cyc)] += 1
    return counts


def test_census_matches_networkx():
    for seed in range(3):
        g = random_regular_connected(30, 3, seed)
        assert np.array_equal(cycle_census(to_adj(g, 3), 9), brute_force_census(g, 9))


def test_ground_state_energy_matches_konopka():
    """[K08, Sec. IV A 2]: lowest energy per node at r = -2.5 is -12.2."""
    g = konopka_ground_state(6)                     # N = 36, as in [K08] Fig. 2(a)
    w = cycle_weights(r=-2.5, l_max=9)
    eps0 = total_energy(to_adj(g, 3), w) / 36
    assert ORIENTATIONS == 2
    assert abs(eps0 - (-12.2)) < 0.05, eps0


def test_weights_formula():
    w = cycle_weights(r=-2.5, l_max=9)
    assert math.isclose(w[4], -2 * 4 * (2.5**4) / 24)   # even cycles lower the energy
    assert w[5] > 0                                      # odd cycles raise it [K08, Sec. II A]
