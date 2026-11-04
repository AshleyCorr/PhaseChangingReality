"""Body diagonals derived from the edges (graphity.diagonals): the first-order rule, the flat values, the light reader."""
import numpy as np

from graphity import diagonals, lengths

DIMS = (8, 8, 8)


def test_flat_diagonals_are_root_three_and_light_takes_root_three_per_step():
    delta = np.zeros((3,) + DIMS)
    diag = diagonals.diagonal_lengths(delta)
    assert np.allclose(diag, np.sqrt(3))
    # (0,0,0) to (4,0,0): four body-diagonal steps at least, e.g. (1,1,1),(1,-1,-1),(1,1,1),(1,-1,-1)
    assert np.isclose(diagonals.light_ticks(diag, (0, 0, 0), (4, 0, 0)), 4 * np.sqrt(3))
    assert np.isclose(diagonals.light_ticks(diag, (0, 0, 0), (1, 1, 1)), np.sqrt(3))


def test_one_edge_lengthens_its_four_cubes_by_a_quarter_over_root_three():
    delta = np.zeros((3,) + DIMS)
    d = 0.04
    delta[0, 2, 3, 4] = d                     # the x-link whose lower point is (2, 3, 4)
    diag = diagonals.diagonal_lengths(delta) - np.sqrt(3)
    touched = {(2, 3, 4), (2, 2, 4), (2, 3, 3), (2, 2, 3)}   # cubes with lower corners (2, 3 or 2, 4 or 3)
    for cube in touched:
        assert np.isclose(diag[cube], (d / 4) / np.sqrt(3), rtol=0.02)
    mask = np.ones(DIMS, dtype=bool)
    for cube in touched:
        mask[cube] = False
    assert np.all(diag[mask] == 0)


def test_cube_edge_means_are_means_of_the_four_edges():
    rng = np.random.default_rng(1)
    delta = rng.random((3,) + DIMS) * 0.1
    l = diagonals.cube_edge_lengths(delta)
    c = (1, 2, 3)
    expected = 1 + (delta[0][1, 2, 3] + delta[0][1, 3, 3] + delta[0][1, 2, 4] + delta[0][1, 3, 4]) / 4
    assert np.isclose(l[0][c], expected)


def test_a_relaxed_halo_lengthens_diagonals_near_curl_for_negative_alpha():
    f = np.zeros((3,) + DIMS); f[0, 4, 4, 4] = 8.0
    delta, _ = lengths.relax(f, -0.5)
    diag = diagonals.diagonal_lengths(delta)
    assert diag[4, 4, 4] > np.sqrt(3) > diag[0, 0, 0]
    near = diagonals.light_ticks(diag, (2, 4, 4), (6, 4, 4))
    far = diagonals.light_ticks(diag, (2, 0, 0), (6, 0, 0))
    assert near > far
