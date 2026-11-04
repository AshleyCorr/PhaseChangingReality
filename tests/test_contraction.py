"""The contraction model, layer form (graphity.contraction): rule-6 checks.

Brute force against networkx (simple, 6-regular, bipartite, valid) after every move; exactness of the layer moves
against the built tori; the moves as each other's inverse; the ladder's energies; refusals where the wiring is not
the lattice's; conservation of energy in the driven protocol to 1e-9; same-seed reproducibility; the gate; and the
thermal chain over widths against the exact Boltzmann distribution.
"""
import networkx as nx
import numpy as np
import pytest

from graphity import contraction as C
from graphity.cqg_d import _switch, is_valid, torus
from graphity.dimension import local_dimension_d


def same_graph(a, b):
    return np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))


def brute_ok(state):
    g = nx.Graph()
    for u in range(state.n):
        for v in state.adj[u]:
            assert u != v
            g.add_edge(u, int(v))
    assert g.number_of_nodes() == state.n
    assert all(d == 6 for _, d in g.degree())
    assert nx.is_bipartite(g)
    colour = nx.bipartite.color(g)
    assert all(colour[u] == state.part[u] for u in range(state.n)) or all(colour[u] != state.part[u] for u in range(state.n))
    assert is_valid(state.adj)
    # canonical: index = ravel of coords
    assert np.array_equal(np.ravel_multi_index(tuple(state.coords.T), state.dims), np.arange(state.n))


@pytest.mark.parametrize("dims,axis", [((6, 6, 6), 0), ((6, 6, 8), 0), ((8, 6, 6), 0), ((6, 8, 6), 1), ((6, 6, 8), 2)])
def test_contracting_a_layer_pair_gives_the_built_torus(dims, axis):
    st = C.build(dims)
    for x0 in range(dims[axis]):
        out = C.contract_layers(st, axis, x0)
        assert out is not None
        target = tuple(d - 2 if a == axis else d for a, d in enumerate(dims))
        assert out.dims == target
        ref, _ = torus(target)
        assert same_graph(out.adj, ref)
        brute_ok(out)


@pytest.mark.parametrize("dims,axis", [((4, 6, 6), 0), ((6, 6, 6), 0), ((6, 4, 6), 1)])
def test_splitting_gives_the_built_torus_and_inverts_contraction(dims, axis):
    st = C.build(dims)
    for x0 in range(dims[axis]):
        out = C.split_layers(st, axis, x0)
        assert out is not None
        target = tuple(d + 2 if a == axis else d for a, d in enumerate(dims))
        assert out.dims == target
        ref, _ = torus(target)
        assert same_graph(out.adj, ref)
        brute_ok(out)
        back = C.contract_layers(out, axis, x0 + 1)
        assert back is not None and back.dims == dims and same_graph(back.adj, st.adj)


@pytest.mark.parametrize("lam", [1.10, 1.25, 1.40])
@pytest.mark.parametrize("L", [6, 8])
def test_the_ladder_is_free_above_six_and_costs_the_curling_cost_at_four(lam, L):
    st = C.build((10, L, L))
    assert C.energy(st, lam) == 0.0
    for width in (8, 6, 4):
        st = C.contract_layers(st, 0, 0)
        assert st is not None and st.dims == (width, L, L)
        expected = 4.0 * (lam - 1.0) * st.n if width == 4 else 0.0
        assert C.energy(st, lam) == pytest.approx(expected, abs=1e-9)
        d = local_dimension_d(st.adj)
        assert set(d.tolist()) == ({2} if width == 4 else {3})
    assert C.contract_layers(st, 0, 0) is None          # width 4 cannot contract to 2
    with_mu = C.energy(st, lam, mu_n=0.5)
    assert with_mu == pytest.approx(4.0 * (lam - 1.0) * st.n + 0.5 * st.n)


def test_stores_are_carried_and_pooled_through_a_contraction():
    st = C.build((6, 6, 6), stores=np.arange(216, dtype=float))
    out = C.contract_layers(st, 0, 2)
    assert out is not None
    assert out.stores.sum() == pytest.approx(st.stores.sum())
    # the removed layers' stores land on the spliced pairs, half each
    removed = np.isin(st.coords[:, 0], [2, 3])
    rec = C.receiving_mask(out, 0, 2, True)
    assert out.stores[rec].sum() == pytest.approx(st.stores[removed].sum() + st.stores[np.isin(st.coords[:, 0], [1, 4])].sum())
    back = C.split_layers(out, 0, 1)
    assert back.stores.sum() == pytest.approx(st.stores.sum())


def test_a_switch_across_the_cut_is_refused_and_elsewhere_is_not():
    st = C.build((8, 6, 6))
    # a switch that moves a link between layer 2 and layer 3 (the pair to be removed) and layer 1: crosses the cut
    u1 = st.index((2, 0, 0)); v1 = st.index((3, 0, 0))
    u2 = st.index((2, 2, 2)); v2 = st.index((1, 2, 2))
    adj = st.adj.copy()
    _switch(adj, u1, v1, u2, v2)
    broken = C.LayerState(adj, st.part, st.coords, st.dims, st.stores)
    assert C.contract_layers(broken, 0, 2) is None
    # the same graph contracted far from the switch (layers 5, 6) is allowed
    assert C.contract_layers(broken, 0, 5) is not None
    # a split needs the (x0, p)-(x0+1, p) links: refused where the switch removed one
    assert C.split_layers(broken, 0, 2) is None
    assert C.split_layers(broken, 0, 5) is not None


def test_the_driven_protocol_conserves_energy_and_repeats_with_the_seed():
    st = C.build((4, 6, 6))
    lam = 1.25
    f0 = C.energy(st, lam)
    a, rows_a = C.run_feed(st, lam, rate=0.5, n_sweeps=60, seed=7, gate=0.0, leak=0.01, proposals_per_sweep=2,
                           max_width=10, log_every=1)
    b, rows_b = C.run_feed(st, lam, rate=0.5, n_sweeps=60, seed=7, gate=0.0, leak=0.01, proposals_per_sweep=2,
                           max_width=10, log_every=1)
    assert rows_a == rows_b and same_graph(a.adj, b.adj) and np.array_equal(a.stores, b.stores)
    for row in rows_a:
        _, n, dims, f, s, fed, leaked, *_ = row
        assert abs(f + s - fed + leaked - f0) < 1e-9 * max(1.0, fed)
    assert rows_a[-1][7] >= 1                                 # the slab split at least once
    brute_ok(a)


def test_the_gate_holds_the_slab_until_the_stores_hold_it():
    st = C.build((4, 6, 6))
    lam = 1.25
    gate = 4.0 * st.n                                          # the switch path's climb, 4 per slab point
    rate = 1.0
    final, rows = C.run_feed(st, lam, rate=rate, n_sweeps=400, seed=3, gate=gate, proposals_per_sweep=1,
                             max_width=6)
    first = next(r for r in rows if r[7] >= 1)
    # the pool is the two layers beside the cut (72 points), filling at `rate` each a sweep
    assert first[0] >= gate / (72 * rate)
    # at the first split nothing had contracted yet, and the release was credited to the stores exactly
    assert first[8] == 0 and first[2] == (6, 6, 6)
    assert first[4] == pytest.approx(first[5] - first[6] + 4.0 * (lam - 1.0) * 144, abs=1e-9)
    # the stores then hold the lump, which can pay the contraction back: the product is not held open by anything
    # in this protocol (a fact for the brief, not a fault of the kernel), so the final width is whichever the last
    # paid move left; conservation holds throughout
    f0 = C.energy(st, lam)
    for row in rows:
        assert abs(row[3] + row[4] - row[5] + row[6] - f0) < 1e-9 * max(1.0, row[5])
    _, rows0 = C.run_feed(st, lam, rate=rate, n_sweeps=5, seed=3, gate=0.0, proposals_per_sweep=4, max_width=6)
    assert rows0[-1][7] >= 1                                   # with no gate it splits at once


def test_the_thermal_chain_over_widths_matches_the_exact_distribution():
    lam, g, mu = 1.25, 2.0, 0.05
    st = C.build((6, 6, 6))                                    # transverse widths >= 6: flat at every width along axis 0
    _, widths = C.run_thermal_widths(st, lam, g, n_sweeps=6000, seed=11, mu_n=mu, max_width=10, axis=0)
    ws, p = C.exact_width_distribution(lam, g, mu, (6, 6), max_width=10)
    counts = np.array([(widths[1000:] == w).sum() for w in ws]) / len(widths[1000:])
    assert np.all(np.abs(counts - p) < 0.06), (counts, p)
