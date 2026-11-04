"""T85: the gamma test. Does one alpha make the toy's light delay and its pull agree as Einstein's gamma = 1 does, across
impact parameters and knots? (PREREGISTRATION T85; docs/design/diagonals_brief_2026-10-28.md, section 3.) Usage:

    python scripts/t85_gamma.py L SEED [ALPHA_USED]

Writes results/t85_gamma_L<L>_s<SEED>.csv and .meta.json. Exact, no chain.

The halo: one random valid switch near the origin (the knot, charges q by family; its energy above flat E_knot in toy
units, 16 (squares lost) + 4 lambda (surplus gained) at lambda = 1.25), its edge lengths relaxed by graphity.lengths with
the total fixed (alpha_used, sign negative so that links lengthen near curl). The body diagonals' lengths are derived
from the edges (graphity.diagonals). For each impact parameter b:
  D(b)  extra ticks of light along body diagonals from (-R, b, 0) to (R, b, 0), R = L/4, over the same displacement far
        from the knot (shifted by L/2 in y and z);
  P(b)  the line integral along that near path of |U|, U(x) = E_int(x) / E_knot, E_int(x) the exact pair energy of a
        copy of the knot placed with its origin at x (graphity.lengths), the step length sqrt(3) per point;
  alpha_star(b) = alpha_used * D / (2 P): the alpha at which D = 2 P, which is Einstein's weak-field relation between
        the Shapiro delay and the potential along the path with gamma = 1 (D is linear in alpha, P quadratic).
ONE alpha means alpha_star is the same across b and across knots (two seeds).
"""
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__, diagonals, lengths  # noqa: E402
from graphity.cqg_d import _switch, hamiltonian, is_valid, torus  # noqa: E402
from graphity.results import ResultWriter, provenance  # noqa: E402

LAM = 1.25


def make_knot(adj, coords, idx, L, rng):
    n = adj.shape[0]
    cand = [p for p in range(n) if np.all(np.abs((coords[p] + L // 2) % L - L // 2) <= 1)]
    for _ in range(100000):
        u1, u2 = rng.choice(cand, 2, replace=False)
        v1 = adj[u1, rng.integers(6)]; v2 = adj[u2, rng.integers(6)]
        if len({u1, v1, u2, v2}) < 4 or v1 in adj[u2] or v2 in adj[u1]:
            continue
        _switch(adj, u1, v1, u2, v2)
        if is_valid(adj):
            return (u1, v1, u2, v2)
        _switch(adj, u1, v2, u2, v1)
    raise RuntimeError("no valid switch")


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
        assert is_valid(adjB)
        return lengths.relax(lengths.curl_fields(adjB, dims), alpha)[1] - 2 * EA

    name = "t85_gamma_L%d_s%d" % (L, seed)
    meta = dict(L=L, seed=seed, alpha_used=alpha, lam=LAM, knot_switch=[int(p) for p in quad], charges=q, e_knot=e_knot,
                package=__version__, python=platform.python_version(), numpy=np.__version__,
                preregistration="PREREGISTRATION.md section T85", provenance=provenance())
    R = L // 4
    with ResultWriter(name, meta, out_dir) as out:
        for b in range(2, L // 2 - 1, 2):
            src, dst = (-R, b, 0), (R, b, 0)
            near = diagonals.light_ticks(diag, src, dst)
            far = diagonals.light_ticks(diag, (-R, b + L // 2, L // 2), (R, b + L // 2, L // 2))
            D = near - far
            # the line integral of |U| along the straight near path, sampled at its 2R + 1 lattice points
            P = 0.0
            for x in range(-R, R + 1):
                point = (x, b, 0)
                if all(((coords[p] + np.array(point)) % L == coords[p]).all() for p in quad):
                    continue
                e_int = pair_energy_at(point)
                P += abs(e_int) / e_knot * np.sqrt(3)
            alpha_star = alpha * D / (2 * P) if P > 0 else float("nan")
            out.write(dict(L=L, seed=seed, b=b, near_ticks=near, far_ticks=far, D=D, P=P, alpha_star=alpha_star,
                           q0=q[0], q1=q[1], q2=q[2], e_knot=e_knot))
            print("L=%d seed=%d b=%d  D=%.5f  P=%.5f  alpha*=%.4f" % (L, seed, b, D, P, alpha_star), flush=True)
    print(name, "written; charges", q, "E_knot", e_knot, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else -0.5)
