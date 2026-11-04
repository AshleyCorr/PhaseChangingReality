"""Bianconi's discrete entropy action on this project's wirings (ASSUMPTIONS O111): the cells are the model's own,
the operators obey the identities her paper states, and the numbers recorded are the numbers computed."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exact_entropy_action as ea  # noqa: E402
from graphity.cqg import total_squares  # noqa: E402

ARRANGED = dict(ea.arrangements(64))


@pytest.mark.parametrize("name", list(ARRANGED))
def test_cells_and_operators(name):
    adj = ARRANGED[name]
    n = adj.shape[0]
    links, squares = ea.cells(adj)
    b1, b2 = ea.boundaries(adj)
    # the cells are the model's: 2N links, and as many squares as the kernel counts and as a trace formula counts
    a = np.zeros((n, n))
    for u in range(n):
        for v in adj[u]:
            a[u, int(v)] = 1.0
    assert len(links) == 2 * n
    assert len(squares) == int(total_squares(adj)) == round((np.trace(np.linalg.matrix_power(a, 4)) - 28 * n) / 8)
    assert all(len(set(cycle)) == 4 and all(a[cycle[i], cycle[(i + 1) % 4]] for i in range(4)) for cycle in squares)
    # the identities of [Bia24]: a boundary has no boundary; D^2 is the Gauss-Bonnet Laplacian block by block
    # (Eqs. 35 to 40); D anticommutes with gamma0 (Eq. 34)
    assert np.allclose(b1 @ b2, 0.0)
    d, gamma0 = ea.dirac(b1, b2)
    e = len(links)
    square, laps = d @ d, ea.hodge(b1, b2)
    assert np.allclose(square[:n, :n], laps[0]) and np.allclose(square[n:n + e, n:n + e], laps[1])
    assert np.allclose(square[n + e:, n + e:], laps[2])
    assert np.allclose(square[:n, n:], 0.0) and np.allclose(square[n:n + e, n + e:], 0.0)
    assert np.allclose(d @ gamma0 + gamma0 @ d, 0.0)


def test_holes():
    spec = {name: ea.spectra(adj) for name, adj in ARRANGED.items()}
    assert ea.betti(spec["flat torus"]) == (1, 2, 1)                    # a torus: one piece, two loops, one cavity
    assert ea.betti(spec["curled torus (tube)"]) == (1, 1, 16)          # each ring is filled: one loop, 16 cavities
    assert ea.betti(spec["tube, re-glued once"]) == (1, 1, 16)
    assert ea.betti(spec["4 knots (4-cubes)"]) == (4, 0, 28)            # the 4-cube's squares enclose 7 cavities
    # the flat torus's node Laplacian is the lattice's: 4 - 2 cos(2 pi a / 8) - 2 cos(2 pi b / 8)
    lattice = sorted(4 - 2 * math.cos(2 * math.pi * i / 8) - 2 * math.cos(2 * math.pi * j / 8)
                     for i in range(8) for j in range(8))
    assert np.allclose(np.sort(spec["flat torus"][0]), lattice)


def test_vacuum():
    for sigma in (0.05, 0.1, 0.3):
        g = ea.vacuum_metric(sigma)
        assert 1 / math.e < g < 1 and abs(-g * math.log(g) - sigma) < 1e-12            # Eq. (65)
        action = lambda x: sigma * math.log(x) + x * math.log(x) - x                    # noqa: E731  S+ per cell
        assert abs((action(g + 1e-6) - action(g - 1e-6)) / 2e-6) < 1e-8                 # stationary there
        assert abs(ea.vacuum_action_per_cell(sigma) - action(g)) < 1e-12
    assert abs(ea.vacuum_action_per_cell(0.1) - (-1.005377)) < 1e-6


def test_the_recorded_numbers():
    """ASSUMPTIONS O111. More squares, a more negative action, in the vacuum and at the identity metric alike."""
    spec = {name: ea.spectra(adj) for name, adj in ARRANGED.items()}
    cells = {name: sum(len(mu) for mu in s) for name, s in spec.items()}
    assert [cells[k] for k in ARRANGED] == [256, 272, 272, 288]                         # 3N + S
    vacuum = [ea.vacuum_action_per_cell(0.1) * cells[k] for k in ARRANGED]
    assert np.allclose(vacuum, [-257.377, -273.463, -273.463, -289.549], atol=1e-3)
    at_one = [ea.action_at_identity(spec[k], 1.0) for k in ARRANGED]
    assert np.allclose(at_one, [-642.0339, -688.1332, -688.1332, -734.7802], atol=1e-4)
    assert at_one[0] > at_one[1] > at_one[3]
    # the re-glued tube is another network with another spectrum, and the action at the identity barely sees it:
    # the two agree in every closed walk shorter than the way round the tube
    tube, twin = spec["curled torus (tube)"], spec["tube, re-glued once"]
    assert max(float(np.abs(np.sort(x) - np.sort(y)).max()) for x, y in zip(tube, twin)) > 0.3
    assert all(abs(float((x ** p).sum() - (y ** p).sum())) < 1e-6 for x, y in zip(tube, twin) for p in range(1, 7))
    assert abs(ea.action_at_identity(tube, 1.0) - ea.action_at_identity(twin, 1.0)) < 1e-8


def test_across_saved_wirings_the_action_acts_like_the_energy_below_lambda_one():
    """ASSUMPTIONS O111, addendum. At the identity metric the action falls with every square and rises with every
    surplus square, as this model's energy does, with an effective coefficient well below 1."""
    wirings = ea.saved_wirings(per_kind=4)
    assert len(wirings) > 100 and len({(s, x) for s, x, _adj in wirings}) > 30
    for c0 in (0.1, 1.0, 10.0):
        per_square, per_surplus, lam, rms, _worst = ea.effective_rule(wirings, c0)
        assert per_square < 0 < per_surplus and 0 < lam < 1
    lam_small, lam_one, lam_ten = (ea.effective_rule(wirings, c0)[2] for c0 in (0.1, 1.0, 10.0))
    assert lam_small < lam_one < lam_ten                       # it grows with c0 and stays below 1
    assert abs(lam_one - 0.2755) < 0.002
    # the three rungs, at every c0 from a hundredth to a thousand: more squares, a more negative action
    spec = {name: ea.spectra(adj) for name, adj in ARRANGED.items()}
    for c0 in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        flat, tube, knots = (ea.action_at_identity(spec[k], c0) for k in ("flat torus", "curled torus (tube)", "4 knots (4-cubes)"))
        assert knots < tube < flat
