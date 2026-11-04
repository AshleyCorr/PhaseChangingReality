"""T86 (which object carries a wave) and T87 (does a loop's curl have a sign), checked on graphs whose answers are known
by hand."""
import networkx as nx
import numpy as np
import pytest

from scripts.t86_wave_carrier import best_mode, dense, eigenspaces, laplacian, stay, word as word86
from scripts.t87_curl_sign import cycle_order, has_mirror, word as word87


def cycle_adj(n):
    adj = -np.ones((n, 4), dtype=np.int64)
    for v in range(n):
        adj[v, 0], adj[v, 1] = (v - 1) % n, (v + 1) % n
    return adj


def test_the_isolated_loop_of_four_keeps_all_its_probability_and_lives_on_itself():
    a = dense(cycle_adj(4))
    w, v = np.linalg.eigh(laplacian(a))
    spaces = eigenspaces(w, v)
    assert best_mode(spaces, [0, 1, 2, 3]) == pytest.approx(1.0)
    for t in (0.3, np.pi / 2, np.pi):
        assert stay(w, v, [0], [0, 1, 2, 3], t) == pytest.approx(1.0)
    # the brief's exact answer: the opposite point carries sin^4 t
    phase = np.exp(-1j * w * np.pi / 2)
    psi = v @ (phase * v[0, :])
    assert abs(psi[2]) ** 2 == pytest.approx(1.0)


def test_a_loop_joined_to_a_long_cycle_leaks_its_probability():
    # a 4-cycle (0..3) with every point also linked into a 40-cycle: the whole thing is 4-regular-ish and open
    n = 44
    adj = -np.ones((n, 4), dtype=np.int64)
    g = nx.cycle_graph(40)
    g = nx.relabel_nodes(g, {i: i + 4 for i in range(40)})
    for i in range(4):
        g.add_edge(i, (i + 1) % 4)
        g.add_edge(i, 4 + 10 * i)
    for v in g.nodes:
        for k, w_ in enumerate(g.neighbors(v)):
            adj[v, k] = w_
    a = dense(adj)
    w, v = np.linalg.eigh(laplacian(a))
    assert stay(w, v, [0, 1, 2, 3], [0, 1, 2, 3], np.pi) < 0.7


def test_words_for_t86():
    rows = [dict(label="object", kind="column", mode=0.9, stay_half=0.8)] * 3 + [dict(label="object", kind="column", mode=0.1, stay_half=0.1)]
    assert word86(rows, "column") == ("CARRIES", 4)
    rows = [dict(label="object", kind="column", mode=0.1, stay_half=0.2)] * 3
    assert word86(rows, "column") == ("LEAKS", 3)
    assert word86([], "column") == ("NONE", 0)


def mirror_setup(g):
    side = {v: v % 2 for v in g.nodes}
    return side


def test_a_four_cycle_in_a_cube_has_a_mirror():
    g = nx.hypercube_graph(3)
    g = nx.convert_node_labels_to_integers(g)
    # find a 4-cycle: nodes of the cube with first coordinate 0 are a face
    face = [v for v in g.nodes][:0]
    cycles = [c for c in nx.cycle_basis(g) if len(c) == 4]
    loop = cycle_order(g, cycles[0])
    side = {v: nx.bipartite.color(g)[v] for v in g.nodes}
    assert loop is not None
    assert has_mirror(g, side, loop, 3)


def test_a_four_cycle_with_one_tail_is_signed():
    g = nx.cycle_graph(4)
    # a tail of length 2 at point 0 and a tail of length 3 at point 1: no reflection of the loop survives
    nx.add_path(g, [0, 10, 11])
    nx.add_path(g, [1, 20, 21, 22])
    side = {v: nx.bipartite.color(g)[v] for v in g.nodes}
    loop = cycle_order(g, [0, 1, 2, 3])
    assert not has_mirror(g, side, loop, 3)
    # the same loop with equal tails at 0 and 2 has the reflection fixing 0 and 2
    h = nx.cycle_graph(4)
    nx.add_path(h, [0, 10, 11])
    nx.add_path(h, [2, 20, 21])
    side = {v: nx.bipartite.color(h)[v] for v in h.nodes}
    assert has_mirror(h, side, cycle_order(h, [0, 1, 2, 3]), 3)


def test_words_for_t87():
    rows = [dict(r3="SIGNED")] * 3 + [dict(r3="MIRROR")]
    assert word87(rows) == ("SIGNED", 4)
    assert word87([dict(r3="MIRROR")] * 2 + [dict(r3="SIGNED")]) == ("MIRROR", 3)
    assert word87([]) == ("NONE", 0)
