"""Tests of graphity.regge (ASSUMPTIONS Q27): angles against coordinates, flat space, Gauss and Bonnet on a torus and
a sphere, the kind of every diagonal, and the toy's own torus labels."""
import math
import random

import numpy as np
import pytest

from graphity import cqg, regge


def test_angles_match_coordinates():
    # brute force: angles of a triangle drawn in the plane, against the law of cosines from its side lengths
    rng = random.Random(1)
    for _ in range(200):
        pts = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(3)])
        la, lb, lc = (np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[0] - pts[1]))
        if min(la, lb, lc) < 1e-3:
            continue
        got = regge.triangle_angles(la, lb, lc)
        for i in range(3):
            u, v = pts[(i + 1) % 3] - pts[i], pts[(i + 2) % 3] - pts[i]
            want = math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            assert got[i] == pytest.approx(want, abs=1e-9)
        assert sum(got) == pytest.approx(math.pi, abs=1e-12)


def test_triangle_inequality_is_enforced():
    with pytest.raises(ValueError):
        regge.triangle_angles(1.0, 1.0, 2.5)


def test_flat_torus_has_no_curvature():
    tris, edges, diags = regge.torus_triangles(8)
    d = regge.deficits(tris, regge.flat_lengths(edges, diags))
    assert len(d) == 64
    assert max(abs(v) for v in d.values()) < 1e-12


def test_torus_is_a_torus_and_matches_the_toy():
    tris, edges, diags = regge.torus_triangles(6, 8)
    assert regge.euler_number(tris) == 0
    adj, part = cqg.torus(6, 8)
    toy_links = {regge.side(u, int(v)) for u in range(adj.shape[0]) for v in adj[u]}
    assert edges == toy_links
    for a, b in diags:                       # every diagonal joins two corners of kind 0
        assert part[a] == part[b] == 0
    assert len(diags) == 48                  # one per square


@pytest.mark.parametrize("seed", range(5))
def test_gauss_bonnet_on_the_torus_for_any_lengths(seed):
    rng = random.Random(seed)
    tris, edges, diags = regge.torus_triangles(8)
    length = {s: v * rng.uniform(0.85, 1.15) for s, v in regge.flat_lengths(edges, diags).items()}
    d = regge.deficits(tris, length)
    assert max(abs(v) for v in d.values()) > 1e-3          # curved somewhere
    assert sum(d.values()) == pytest.approx(0.0, abs=1e-9)  # but the total is the torus's: 2 pi x 0


def test_gauss_bonnet_on_a_sphere():
    # the octahedron: 6 points, 12 sides, 8 triangles; Euler number 2, so the deficits add to 4 pi for any lengths
    tris = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2), (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]
    assert regge.euler_number(tris) == 2
    rng = random.Random(7)
    sides = {regge.side(a, b) for p, q, r in tris for a, b in ((p, q), (q, r), (p, r))}
    for _ in range(20):
        length = {s: rng.uniform(0.9, 1.1) for s in sides}
        assert sum(regge.deficits(tris, length).values()) == pytest.approx(4 * math.pi, abs=1e-9)


def test_a_bump_is_paid_for_by_its_ring():
    # shorten the sides at one point of the flat torus: its angles open past 2 pi, a saddle (negative curvature), and its
    # neighbours take the opposite curvature, exactly cancelling (the balance of curl and uncurl, free in two directions)
    tris, edges, diags = regge.torus_triangles(8)
    length = regge.flat_lengths(edges, diags)
    centre = 3 * 8 + 3
    for s in list(length):
        if centre in s:
            length[s] *= 0.9
    d = regge.deficits(tris, length)
    assert abs(d[centre]) > 1e-3
    assert sum(v for p, v in d.items() if p != centre) == pytest.approx(-d[centre], abs=1e-12)
