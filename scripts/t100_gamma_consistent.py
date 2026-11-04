"""T100: the gamma test in its consistent form, on fresh data (PREREGISTRATION T100; T85's proposed amendment). Exact, no chain.

    python scripts/t100_gamma_consistent.py L SEED [ALPHA_USED]

As T85 (scripts/t85_gamma.py) except that D and P are taken along ONE path: the fixed zigzag along body diagonals from
(-R, b, 0), alternating (1, 1, 1) and (1, -1, -1), 2R steps. D(b) is that path's ticks near the knot minus the same path's
ticks far from it (shifted by L/2 in y and z); P(b) is the sum of |U| over the 2R + 1 lattice points the path visits, times
sqrt 3, with U(x) = E_int(x) / E_knot; alpha*(b) = alpha_used * D / (2 P). Writes results/t100_gamma_L<L>_s<SEED>.csv.
"""
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__, diagonals, lengths  # noqa: E402
from graphity.cqg_d import _switch, hamiltonian, is_valid, torus  # noqa: E402
from graphity.results import ResultWriter, provenance  # noqa: E402
from t85_gamma import LAM, make_knot  # noqa: E402


def zigzag_points(start, steps):
    p = np.array(start); out = [tuple(p)]
    for k in range(steps):
        p = p + (np.array([1, 1, 1]) if k % 2 == 0 else np.array([1, -1, -1]))
        out.append(tuple(p))
    return out


def path_ticks(diag, pts, L):
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        s = np.array(b) - np.array(a)
        cube = tuple(np.mod(np.array(a) + np.minimum(s, 0), L))
        total += diag[cube]
    return total


def main(L, seed, alpha=-0.5, out_dir="results"):
    dims = (L, L, L)
    adj0, part = torus(dims)
    n = adj0.shape[0]
    coords = np.array(np.unravel_index(np.arange(n), dims)).T
    idx = {tuple(c): i for i, c in enumerate(coords)}
    rng = np.random.default_rng(seed)
    adjA = adj0.copy()
    quad = make_knot(adjA, coords, idx, L, rng)
    e_knot = float(hamiltonian(adjA, LAM) - hamiltonian(adj0, LAM))
    fA = lengths.curl_fields(adjA, dims)
    deltaA, EA = lengths.relax(fA, alpha)
    diag = diagonals.diagonal_lengths(deltaA)
    q = [float(x) for x in fA.sum(axis=(1, 2, 3))]

    def pair_energy_at(x):
        adjB = adjA.copy()
        _switch(adjB, *[idx[tuple((coords[p] + np.array(x)) % L)] for p in quad])
        if not is_valid(adjB):
            return None                        # Amendment 1: a copy overlapping the knot is skipped and counted
        return lengths.relax(lengths.curl_fields(adjB, dims), alpha)[1] - 2 * EA

    name = "t100_gamma_L%d_s%d" % (L, seed)
    meta = dict(L=L, seed=seed, alpha_used=alpha, lam=LAM, knot_switch=[int(p) for p in quad], charges=q, e_knot=e_knot,
                path="fixed zigzag (1,1,1)/(1,-1,-1) along body diagonals; D and P on the same path",
                package=__version__, python=platform.python_version(), numpy=np.__version__,
                preregistration="PREREGISTRATION.md section T100", provenance=provenance())
    R = L // 4
    with ResultWriter(name, meta, out_dir) as out:
        for b in range(2, L // 2 - 1, 2):
            near_pts = zigzag_points((-R, b, 0), 2 * R)
            far_pts = zigzag_points((-R, b + L // 2, L // 2), 2 * R)
            D = path_ticks(diag, near_pts, L) - path_ticks(diag, far_pts, L)
            P = 0.0
            skipped = 0
            for point in near_pts:
                if all(((coords[p] + np.array(point)) % L == coords[p]).all() for p in quad):
                    skipped += 1
                    continue
                e_int = pair_energy_at(point)
                if e_int is None:
                    skipped += 1
                    continue
                P += abs(e_int) / e_knot * np.sqrt(3)
            alpha_star = alpha * D / (2 * P) if P > 0 else float("nan")
            out.write(dict(L=L, seed=seed, b=b, D=D, P=P, alpha_star=alpha_star, skipped=skipped, q0=q[0], q1=q[1], q2=q[2], e_knot=e_knot))
            print("L=%d seed=%d b=%d  D=%.5f  P=%.5f  alpha*=%.4f" % (L, seed, b, D, P, alpha_star), flush=True)
    print(name, "written; charges", q, "E_knot", e_knot, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else -0.5)
