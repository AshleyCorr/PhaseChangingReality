"""T6: the density of states by Wang-Landau, checked against answers that are known exactly.

The point of this file is that nothing here is checked against another simulation. Every
number the sampler produces is compared with a count done by brute force: all 65 536 states
of the 4x4 Ising model listed one by one, and every state of the graph model at N = 16 listed
by the exhaustive enumeration of T4. A sampler that agrees with both is right for reasons that
have nothing to do with our own reasoning about it.
"""
import csv
from collections import defaultdict
from math import exp, log
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg import NO_CAP, is_valid, run_chain, surplus, total_squares
from graphity.small_graphs import count_labelled_states
from graphity.wang_landau import (_wl_sweeps, canonical, energy_histogram, dos_graph, graph_sweeper,
                                  normalise_to, spread)
from graphity.wl_ising import _ising_sweeps, dos_ising, exact_dos

ROOT = Path(__file__).resolve().parents[1]


def _ising_pieces(side):
    n = side * side
    idx = np.arange(n)
    return (n,
            (idx // side) * side + (idx % side + 1) % side,
            ((idx // side + 1) % side) * side + idx % side,
            (idx // side) * side + (idx % side - 1) % side,
            ((idx // side - 1) % side) * side + idx % side)


def test_brute_force_ising_counts_are_sane():
    """The reference itself, checked three ways before anything is compared against it."""
    energies, counts = exact_dos(4)
    assert counts.sum() == 2 ** 16                       # every configuration counted once
    assert counts[0] == 2 and counts[-1] == 2            # all up / all down, and the two checkerboards
    assert (counts == counts[::-1]).all()                # the model is symmetric under flipping every spin
    assert counts[1] == 0 and counts[-2] == 0            # E = -2n + 4 cannot be made
    assert int((energies * counts).sum()) == 0           # mean energy at infinite temperature


def test_ising_density_of_states_matches_the_brute_force_count():
    """The schedule, end to end, against a system whose answer is countable by hand."""
    energies, counts = exact_dos(4)
    good = counts > 0
    out = dos_ising(4, seed=11, ln_f_final=1e-5, sweeps_per_check=2000,
                    max_moves=60_000_000, refine_sweeps=400_000)
    assert (out["seen"][:, 0] == good).all(), "found a state that cannot exist, or missed one"
    lng = normalise_to(out["lng"], out["seen"], (0, 0), 2.0)[good, 0]
    err = np.abs(lng - np.log(counts[good]))
    assert err.max() < 0.12, (err, "worst error in ln g")


def test_graph_density_of_states_matches_the_exhaustive_enumeration():
    """N = 16, against every state of the model listed by T4. Bins are (squares, surplus)."""
    exact = defaultdict(int)
    with (ROOT / "results" / "ergodicity_small.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["N"] == "16" and r["cap"] == "none" and r["class_id"] != "-1":
                exact[(int(r["squares"]), int(r["surplus"]))] += int(r["labelled_states"])
    assert len(exact) == 4, "the fixture changed; this test is pinned to the N = 16 enumeration"

    _, adj = count_labelled_states(8, NO_CAP)
    s_lo, s_hi = min(k[0] for k in exact), max(k[0] for k in exact)
    x_lo, x_hi = min(k[1] for k in exact), max(k[1] for k in exact)
    out = dos_graph(adj.copy(), np.arange(8), (s_lo, s_hi), (x_lo, x_hi), cap=NO_CAP, seed=5,
                    ln_f_final=1e-3, sweeps_per_check=2000, max_moves=20_000_000,
                    refine_sweeps=200_000)

    found = {(i + out["s_min"], j + out["x_min"]) for i, j in np.argwhere(out["seen"])}
    assert found == set(exact), (sorted(found), sorted(exact))

    ref = sorted(exact)[0]
    at = lambda k: out["lng"][k[0] - out["s_min"], k[1] - out["x_min"]]
    for k in sorted(exact):
        got = at(k) - at(ref)
        want = log(exact[k] / exact[ref])
        assert abs(got - want) < 0.12, (k, got, want)


def test_one_dimensional_walk_in_s_matches_the_exhaustive_count_summed_over_x():
    """The lambda = 0 instrument: bins are S alone, and the count in each is the sum over X."""
    exact = defaultdict(int)
    with (ROOT / "results" / "ergodicity_small.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["N"] == "16" and r["cap"] == "none" and r["class_id"] != "-1":
                exact[int(r["squares"])] += int(r["labelled_states"])
    _, adj = count_labelled_states(8, NO_CAP)
    s_lo, s_hi = min(exact), max(exact)
    out = dos_graph(adj.copy(), np.arange(8), (s_lo, s_hi), None, cap=NO_CAP, seed=5,
                    ln_f_final=1e-3, sweeps_per_check=2000, max_moves=20_000_000,
                    refine_sweeps=200_000)
    assert out["lng"].shape == (s_hi - s_lo + 1, 1)
    found = {i + out["s_min"] for i, _ in np.argwhere(out["seen"])}
    assert found == set(exact), (sorted(found), sorted(exact))
    ref = min(exact)
    for s in sorted(exact):
        got = out["lng"][s - s_lo, 0] - out["lng"][ref - s_lo, 0]
        want = log(exact[s] / exact[ref])
        assert abs(got - want) < 0.12, (s, got, want)


def test_averages_rebuilt_from_an_exact_density_of_states_are_exact():
    """Pure arithmetic, no sampling: canonical() must reproduce the enumeration's own averages."""
    exact = defaultdict(int)
    with (ROOT / "results" / "ergodicity_small.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["N"] == "18" and r["cap"] == "none" and r["class_id"] != "-1":
                exact[(int(r["squares"]), int(r["surplus"]))] += int(r["labelled_states"])
    s_lo, x_lo = min(k[0] for k in exact), min(k[1] for k in exact)
    shape = (max(k[0] for k in exact) - s_lo + 1, max(k[1] for k in exact) - x_lo + 1)
    lng = np.full(shape, np.nan)
    seen = np.zeros(shape, dtype=bool)
    for (s, x), count in exact.items():
        lng[s - s_lo, x - x_lo] = log(count)
        seen[s - s_lo, x - x_lo] = True

    for lam in (0.0, 1.0):
        for g in (20.0, 6.0, 3.0):
            phi, sur, _, _ = canonical(lng, seen, s_lo, x_lo, 18, g, lam)
            w = {k: c * exp(-(16 * (18 - k[0]) + 4 * lam * k[1]) / g) for k, c in exact.items()}
            z = sum(w.values())
            assert phi == pytest.approx(sum(v * k[0] for k, v in w.items()) / z / 18, abs=1e-12)
            assert sur == pytest.approx(sum(v * k[1] for k, v in w.items()) / z / 18, abs=1e-12)

            e, p = energy_histogram(lng, seen, s_lo, x_lo, 18, g, lam)
            assert p.sum() == pytest.approx(1.0)
            assert (np.diff(e) > 0).all()                       # one entry per distinct energy, in order


def test_a_negative_seed_continues_the_stream_rather_than_starting_one():
    """Two blocks with the stream carried on must equal one block of twice the length (Q14)."""
    n = 16
    pieces = _ising_pieces(4)
    results = []
    for split in (False, True):
        lng = np.zeros((n + 1, 1))
        hist = np.zeros((n + 1, 1), dtype=np.int64)
        seen = np.zeros((n + 1, 1), dtype=bool)
        spins = np.ones(n, dtype=np.int64)
        state = np.array([0], dtype=np.int64)
        args = (spins,) + pieces[1:] + (lng, hist, seen)
        if split:
            _ising_sweeps(*args, 0.5, 300, 4242, state, 0.0)
            _ising_sweeps(*args, 0.5, 300, -1, state, 0.0)
        else:
            _ising_sweeps(*args, 0.5, 600, 4242, state, 0.0)
        results.append((lng.copy(), hist.copy(), state.copy(), spins.copy()))
    for a, b in zip(*results):
        assert (a == b).all()


def test_run_chain_also_continues_on_a_negative_seed():
    """The same property on the production kernel; seed >= 0 keeps its old meaning (Q14)."""
    from graphity.cqg import torus
    adj_one, part = torus(8, 8, cap=NO_CAP)
    side_u = np.flatnonzero(part == 0)
    adj_two = adj_one.copy()

    s_one, x_one, _ = run_chain(adj_one, side_u, 1.0 / 5.0, 0, 40, 909, 1.0, NO_CAP)
    a = run_chain(adj_two, side_u, 1.0 / 5.0, 0, 20, 909, 1.0, NO_CAP)
    b = run_chain(adj_two, side_u, 1.0 / 5.0, 0, 20, -1, 1.0, NO_CAP)
    assert (np.concatenate([a[0], b[0]]) == s_one).all()
    assert (np.concatenate([a[1], b[1]]) == x_one).all()
    assert (adj_one == adj_two).all()


def test_the_walk_only_ever_visits_valid_graphs_and_tracks_them_correctly():
    """Every constraint the kernel enforces, plus the running (S, X) against a full recount."""
    _, adj = count_labelled_states(9, NO_CAP)
    side_u = np.arange(9)
    advance = graph_sweeper(adj, side_u, 0, 0, NO_CAP)
    lng = np.zeros((40, 40))
    hist = np.zeros((40, 40), dtype=np.int64)
    seen = np.zeros((40, 40), dtype=bool)
    for block in range(6):
        advance(lng, hist, seen, 0.5, 40, 31337 if block == 0 else -1)
        assert is_valid(adj, NO_CAP)
        assert advance.state[0] == total_squares(adj)
        assert advance.state[1] == surplus(adj)
    assert hist.sum() == 6 * 40 * 2 * 18          # one update per attempted move
    assert seen.sum() > 1, "the walk never left its starting bin"


def test_the_same_seed_gives_the_same_answer_and_a_different_one_does_not():
    runs = [dos_ising(4, seed=s, ln_f_final=1e-3, sweeps_per_check=500,
                      max_moves=2_000_000, refine_sweeps=20_000) for s in (3, 3, 4)]
    assert (runs[0]["lng"] == runs[1]["lng"]).all()
    assert not (runs[0]["lng"] == runs[2]["lng"]).all()
    worst, per_bin = spread([r["lng"] for r in runs], runs[0]["seen"])
    assert worst > 0 and np.isfinite(per_bin[runs[0]["seen"]]).all()
