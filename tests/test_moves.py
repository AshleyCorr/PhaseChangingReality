"""Checks on the Monte Carlo move and the menu generators."""
import networkx as nx
import numpy as np

from graphity.energy import cycle_weights, total_energy
from graphity.graphs import (menu_to_array, random_menu_with_planted_cubic,
                             random_regular_connected, to_adj, to_networkx,
                             torus_menu_with_cubic_start)
from graphity.mc import dummy_menu, run_chain

W = cycle_weights(r=-2.5, l_max=9)


def check_state(adj, energies):
    g = to_networkx(adj)
    assert all(d == 3 for _, d in g.degree())           # degree preserved
    assert nx.is_connected(g)                           # connectedness preserved
    assert (adj >= 0).all()
    assert abs(total_energy(adj, W) - energies[-1]) < 1e-6   # incremental dE is exact


def test_complete_menu_chain_is_consistent():
    adj = to_adj(random_regular_connected(40, 3, 1), 3)
    energies, acc = run_chain(adj, dummy_menu(), False, W, 0.15, 5, 20, 7)
    assert 0 < acc <= 1
    check_state(adj, energies)


def test_random_menu():
    menu, start = random_menu_with_planted_cubic(60, 8, seed=3)
    assert all(d == 8 for _, d in menu.degree())
    assert all(menu.has_edge(*e) for e in start.edges())
    adj = to_adj(start, 3)
    energies, _ = run_chain(adj, menu_to_array(menu), True, W, 0.0, 5, 20, 11)
    check_state(adj, energies)
    assert all(menu.has_edge(*e) for e in to_networkx(adj).edges())  # never leaves the menu


def test_torus_menus():
    for sides in [(6, 6), (8, 4, 4), (6, 4, 4, 4)]:
        menu, start = torus_menu_with_cubic_start(sides)
        assert all(d == 2 * len(sides) for _, d in menu.degree())
        assert all(d == 3 for _, d in start.degree())
        assert nx.is_connected(start)
        adj = to_adj(start, 3)
        energies, _ = run_chain(adj, menu_to_array(menu), True, W, 0.0, 2, 5, 5)
        check_state(adj, energies)
        assert all(menu.has_edge(*e) for e in to_networkx(adj).edges())


def test_same_seed_same_result():
    runs = []
    for _ in range(2):
        adj = to_adj(random_regular_connected(30, 3, 2), 3)
        runs.append(run_chain(adj, dummy_menu(), False, W, 0.1, 2, 10, 99)[0])
    assert np.array_equal(runs[0], runs[1])
