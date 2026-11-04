"""Link lengths relaxed from curl: the gravity link brief (docs/design/gravity_link_brief.md), steps 1 and 2.

WHAT IS EXACT HERE (the brief, section 4; sandbox/regge_step1_stiffness.py, L = 6, 8 and the held-out 10). With the
published curvature ([T25] Sec. II.2: uniform measure on the sphere of neighbours, transport cost by shortest paths)
taken over link lengths, the flat six-link torus is a minimum in the lengths, and to second order

    H_curv = (1/2) delta^T K delta,   K = 4 (6 I - A) on each axis family of links separately,

delta the length changes, A the cubic adjacency of link positions (a link along axis a is indexed by its lower
point). K is 4 times the cubic-lattice Laplacian: gapless, with uniform scaling as its zero mode.

THE SOURCE (the owner's decision of 28 October, VISION Update 136): a link's value is set by its own curl, coupled with
one constant alpha:  alpha * sum_e |c_e| delta_e,  c_e = S_e - 4 the curl on the link. Minimising over the lengths:

    delta = -alpha K^+ |c|,        E_relax = -(alpha^2 / 2) |c|^T K^+ |c|,

K^+ the pseudo-inverse with the zero mode (uniform scaling) set to zero mean. So two relics lower their joint energy by
alpha^2 |c_1|^T K^+ |c_2|, Newton's shape on the torus, for either sign of alpha; alpha < 0 lengthens links near curl
(light delayed, the owner's choice), alpha > 0 shortens them.

THE RESPONSE (the owner's, 28 October, 15:19 UTC): a move is judged by the total energy with the lengths relaxed,
H_toy + E_relax. The chain below is cqg_d.run_chain's proposal and Metropolis rule with that total; it recomputes
H_toy and E_relax in full at every proposal (exact, slow, for small tori), so every acceptance is checked against the
whole energy and nothing incremental can drift. alpha = 0 is the toy's own chain in distribution.

FAMILIES. After switches a link need not lie along an axis. Its family is the axis of its largest displacement on the
torus (ties to the lowest axis), and it is placed at the lower point along that axis. This is a bookkeeping choice
(ours); it affects nothing on the flat torus and only the placement of a defect's curl in the halo.

Detailed balance: the proposal is cqg_d's symmetric switch; Metropolis on the total energy, which is a function of the
wiring alone (the lengths are its minimiser), so the chain samples exp(-(H_toy + E_relax)/g).
"""
import numpy as np

from graphity.cqg_d import _new_edge_ok, _switch, has_edge, hamiltonian, squares_on_edge

FLAT_SQUARES = 4           # squares on an axis link of the flat six-link torus (2D - 2 at D = 3)
STIFFNESS = 4.0            # K = STIFFNESS * (6 I - A) per family


def curl_fields(adj, dims):
    """|c_e| = |S_e - 4| summed into (3, L0, L1, L2): family = axis of the link's largest displacement, placed at its
    lower point along that axis. Each link counted once."""
    dims = tuple(int(x) for x in dims)
    n = adj.shape[0]
    coords = np.array(np.unravel_index(np.arange(n), dims)).T
    f = np.zeros((3,) + dims)
    for u in range(n):
        for v in adj[u]:
            if v < u:
                continue
            c = abs(squares_on_edge(adj, u, v) - FLAT_SQUARES)
            if c == 0:
                continue
            d = (coords[v] - coords[u] + np.array(dims) // 2) % dims - np.array(dims) // 2
            axis = int(np.argmax(np.abs(d)))
            lower = coords[u] if d[axis] > 0 else coords[v]
            f[axis][tuple(lower % dims)] += c
    return f


def stiffness_eigenvalues(dims):
    """Eigenvalues of K on one family, in FFT order; the zero mode is set to inf so that 1/lam is its pseudo-inverse."""
    ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(L) for L in dims], indexing="ij")
    lam = STIFFNESS * 2 * sum(1 - np.cos(k) for k in ks)
    lam[(0,) * len(dims)] = np.inf
    return lam


def apply_stiffness(field):
    """K f for one family: 24 f minus 4 times the sum over the six neighbouring link positions."""
    out = 6 * field.copy()
    for axis in range(field.ndim):
        out -= np.roll(field, 1, axis) + np.roll(field, -1, axis)
    return STIFFNESS * out


def relax(fields, alpha):
    """(delta, E_relax): delta = -alpha K^+ f per family, E_relax = -(alpha^2 / 2) sum_f f^T K^+ f."""
    lam = stiffness_eigenvalues(fields.shape[1:])
    delta = np.empty_like(fields)
    energy = 0.0
    for a in range(fields.shape[0]):
        g = np.real(np.fft.ifftn(np.fft.fftn(fields[a]) / lam))     # K^+ f
        delta[a] = -alpha * g
        energy += -(alpha ** 2 / 2) * np.sum(fields[a] * g)
    return delta, energy


def relaxed_energy(adj, dims, alpha):
    return relax(curl_fields(adj, dims), alpha)[1]


def total_energy(adj, dims, lam, alpha):
    return float(hamiltonian(adj, lam)) + relaxed_energy(adj, dims, alpha)


def run_chain_relaxed(adj, side_u, dims, inv_g, n_equil, n_meas, seed, lam=1.0, alpha=0.0, on_sweep=None):
    """Metropolis on H_toy + E_relax with cqg_d's switch proposal; modifies adj in place.

    Returns (H_toy per measured sweep, E_relax per measured sweep, acceptance). on_sweep(adj, sweep) is called after
    every measured sweep. Full recomputation at every proposal: exact and slow (small tori only).
    """
    rng = np.random.default_rng(seed)
    n, deg = adj.shape
    h_toy = float(hamiltonian(adj, lam))
    e_rel = relaxed_energy(adj, dims, alpha)
    out_h = np.zeros(n_meas)
    out_e = np.zeros(n_meas)
    attempted = accepted = 0
    for sweep in range(n_equil + n_meas):
        for _ in range(2 * n):
            attempted += 1
            u1 = side_u[rng.integers(len(side_u))]
            v1 = adj[u1, rng.integers(deg)]
            u2 = side_u[rng.integers(len(side_u))]
            v2 = adj[u2, rng.integers(deg)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            _switch(adj, u1, v1, u2, v2)
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                h_new = float(hamiltonian(adj, lam))
                e_new = relaxed_energy(adj, dims, alpha) if alpha != 0.0 else 0.0
                d = (h_new + e_new) - (h_toy + e_rel)
                if d > 0.0 and rng.random() >= np.exp(-inv_g * d):
                    ok = False
            if ok:
                h_toy, e_rel = h_new, e_new
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        if sweep >= n_equil:
            out_h[sweep - n_equil] = h_toy
            out_e[sweep - n_equil] = e_rel
            if on_sweep is not None:
                on_sweep(adj, sweep - n_equil)
    return out_h, out_e, accepted / max(attempted, 1)
