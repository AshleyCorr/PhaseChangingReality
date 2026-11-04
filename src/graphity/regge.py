"""Soft space, stage B, first piece: lengths on links and diagonals, and the curvature they carry (Regge calculus).

ASSUMPTIONS Q27; the brief is docs/design/regge_stage_b_brief.md (the owner's answers of 21 October 2026: two
directions first, as a check, then three; one diagonal per square, joining the two corners of one kind).

A surface is given as a list of triangles (triples of point labels) and a length for each of their sides. Each
triangle is flat; its angles follow from its three side lengths by the law of cosines. The curvature at a point is its
deficit angle, 2 pi minus the angles of the triangles that meet there (Regge's hinge in two directions is the point;
[Ham09] Eq. 518 for the general form). Flat space has zero deficit everywhere.

Exact, and what the tests check: on a closed surface the deficits add up to 2 pi times the Euler number (2 for a
sphere, 0 for a torus), whatever the lengths (Gauss and Bonnet). So in two directions an energy that charges only for
curving, sum of deficits, is the same for every choice of lengths on one surface: it gives the balance of curl and
uncurl for free and no dynamics. That is why this two-direction code is a check and the aim is three directions.

The toy's flat four-link torus is triangulated by giving each square one diagonal, the one joining its two corners of
kind 0 (even coordinate sum); `torus_triangles` builds it from the same labels as `graphity.cqg.torus`.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def side(a: int, b: int) -> tuple[int, int]:
    """A side's key: its two points, smaller label first."""
    return (a, b) if a < b else (b, a)


def triangle_angles(la: float, lb: float, lc: float) -> tuple[float, float, float]:
    """Angles opposite sides of lengths la, lb, lc (law of cosines). Raises if the lengths make no triangle."""
    if not (la < lb + lc and lb < la + lc and lc < la + lb):
        raise ValueError("lengths violate the triangle inequality: %r" % ((la, lb, lc),))

    def opposite(x, y, z):
        c = (y * y + z * z - x * x) / (2.0 * y * z)
        return math.acos(max(-1.0, min(1.0, c)))

    return opposite(la, lb, lc), opposite(lb, la, lc), opposite(lc, la, lb)


def deficits(triangles, length) -> dict[int, float]:
    """Deficit angle at every point of a closed triangulated surface.

    triangles: iterable of (p, q, r); length: dict from side(p, q) to its length.
    """
    total = {}
    for p, q, r in triangles:
        # the angle at p is opposite the side qr, and so on
        a_p, a_q, a_r = triangle_angles(length[side(q, r)], length[side(p, r)], length[side(p, q)])
        for point, ang in ((p, a_p), (q, a_q), (r, a_r)):
            total[point] = total.get(point, 0.0) + ang
    return {point: TWO_PI - s for point, s in total.items()}


def euler_number(triangles) -> int:
    """Points minus sides plus triangles."""
    tris = [tuple(t) for t in triangles]
    points = {p for t in tris for p in t}
    sides = {side(a, b) for p, q, r in tris for a, b in ((p, q), (q, r), (p, r))}
    return len(points) - len(sides) + len(tris)


def torus_triangles(lx: int, ly: int | None = None):
    """The flat lx x ly torus of graphity.cqg.torus, each square cut by its kind-0 diagonal into two triangles.

    Returns (triangles, edges, diagonals): edges are the toy's links, diagonals the added sides, all as side keys.
    """
    if ly is None:
        ly = lx
    label = lambda x, y: (x % lx) * ly + (y % ly)  # noqa: E731
    triangles, edges, diagonals = [], set(), set()
    for x in range(lx):
        for y in range(ly):
            a, b, c, d = label(x, y), label(x + 1, y), label(x + 1, y + 1), label(x, y + 1)
            edges.update({side(a, b), side(b, c), side(c, d), side(d, a)})
            if (x + y) % 2 == 0:            # a and c are the kind-0 corners
                diagonals.add(side(a, c))
                triangles += [(a, b, c), (a, c, d)]
            else:                           # b and d are
                diagonals.add(side(b, d))
                triangles += [(a, b, d), (b, c, d)]
    return triangles, edges, diagonals


def flat_lengths(edges, diagonals) -> dict:
    """Lengths of flat space: links 1, diagonals sqrt 2."""
    out = {e: 1.0 for e in edges}
    out.update({d: math.sqrt(2.0) for d in diagonals})
    return out
