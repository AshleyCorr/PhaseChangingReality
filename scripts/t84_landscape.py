"""T84, stage 1 and the held-out stage: the exact energy landscape of two one-switch knots in flat six-link space with the
link lengths relaxed from their curl (graphity.lengths; PREREGISTRATION T84, Amendment 2). Usage:

    python scripts/t84_landscape.py L ALPHA SEED

Writes results/t84_landscape_L<L>_a<ALPHA>.csv and .meta.json: for each separation r along an axis and along a face
diagonal, the pair energy E_int(r) = E_relax(both) - 2 E_relax(one), the same for two unit point sources on family 0 with
the same solver (the torus's own Green's-function law), their ratio, and the light path past the knot against the same
path far from it. The knot is the defect of one random valid switch near the origin (O58: no six-link knot is a dip, so
this is a shape test of the constrained model, not a test of motion), copied by translation for the second knot.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__, lengths  # noqa: E402
from graphity.cqg_d import _switch, is_valid, torus  # noqa: E402
from graphity.results import ResultWriter, provenance  # noqa: E402

try:
    import numba
    NUMBA = numba.__version__
except ImportError:                                   # pragma: no cover
    NUMBA = ""


def light_path(delta, coords, idx, L, src, dst):
    import heapq
    dist = {src: 0.0}; heap = [(0.0, src)]
    while heap:
        d, p = heapq.heappop(heap)
        if p == dst:
            return d
        if d > dist[p]:
            continue
        c = coords[p]
        for axis in range(3):
            for s in (1, -1):
                q = idx[tuple((c + s * np.eye(3, dtype=int)[axis]) % L)]
                lower = c if s > 0 else coords[q]
                nd = d + 1 + delta[axis][tuple(lower % L)]
                if nd < dist.get(q, np.inf):
                    dist[q] = nd; heapq.heappush(heap, (nd, q))


def main(L, alpha, seed, out_dir="results"):
    dims = (L, L, L)
    adj0, part = torus(dims)
    n = adj0.shape[0]
    coords = np.array(np.unravel_index(np.arange(n), dims)).T
    idx = {tuple(c): i for i, c in enumerate(coords)}
    rng = np.random.default_rng(seed)
    cand = [p for p in range(n) if np.all(np.abs((coords[p] + L // 2) % L - L // 2) <= 1)]
    adjA = adj0.copy()
    for _ in range(100000):
        u1, u2 = rng.choice(cand, 2, replace=False)
        v1 = adjA[u1, rng.integers(6)]; v2 = adjA[u2, rng.integers(6)]
        if len({u1, v1, u2, v2}) < 4 or v1 in adjA[u2] or v2 in adjA[u1]:
            continue
        _switch(adjA, u1, v1, u2, v2)
        if is_valid(adjA):
            quad = (u1, v1, u2, v2)
            break
        _switch(adjA, u1, v2, u2, v1)
    else:
        raise RuntimeError("no valid switch found")
    fA = lengths.curl_fields(adjA, dims)
    deltaA, EA = lengths.relax(fA, alpha)
    one = np.zeros((3,) + dims); one[0, 0, 0, 0] = 1.0
    E1 = lengths.relax(one, alpha)[1]
    name = "t84_landscape_L%d_a%s" % (L, ("m" if alpha < 0 else "p") + "%03d" % round(abs(alpha) * 100))
    meta = dict(L=L, alpha=alpha, seed=seed, knot_switch=[int(p) for p in quad], family_charges=[float(x) for x in fA.sum(axis=(1, 2, 3))],
                package=__version__, python=platform.python_version(), numpy=np.__version__, numba=NUMBA,
                preregistration="PREREGISTRATION.md section T84, Amendment 2", provenance=provenance())
    with ResultWriter(name, meta, out_dir) as out:
        for r in range(2, L // 2 + 1):
            for kind, shift in (("axis", np.array([r, 0, 0])), ("face", np.array([r, r, 0]))):
                adjB = adjA.copy()
                _switch(adjB, *[idx[tuple((coords[p] + shift) % L)] for p in quad])
                assert is_valid(adjB)
                EAB = lengths.relax(lengths.curl_fields(adjB, dims), alpha)[1]
                two = one.copy(); two[0][tuple(shift % L)] += 1.0
                ref = lengths.relax(two, alpha)[1] - 2 * E1
                e_int = EAB - 2 * EA
                out.write(dict(L=L, alpha=alpha, r=r, kind=kind, euclid=float(np.linalg.norm(shift)), e_int=e_int,
                               e_ref=ref, ratio=(e_int / ref if ref != 0 else "")))
        near = light_path(deltaA, coords, idx, L, idx[((-3) % L, 0, 0)], idx[(3, 0, 0)])
        far = light_path(deltaA, coords, idx, L, idx[((-3) % L, L // 2, L // 2)], idx[(3, L // 2, L // 2)])
        out.write(dict(L=L, alpha=alpha, r="", kind="light", euclid=6.0, e_int=near, e_ref=far, ratio=near / far))
    print(name, "written; family charges", fA.sum(axis=(1, 2, 3)), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
