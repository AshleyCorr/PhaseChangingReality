"""A phase on every link of the toy's graph: the published lattice photon (compact U(1) lattice gauge theory).

ASSUMPTIONS Q29; VISION Update 96 (the owner's decision of 24 October 2026 to build it into the toy, from her
particle handoff, `docs/geometric_particle_model_handoff_2026-10-24.md`, idea 1: "a flat path/link exists without a
photon; a photon is an excitation traveling along a pre-existing path"). Wilson's lattice gauge theory (Wilson 1974;
review Kogut, RMP 51, 659 (1979); both to read, general knowledge until then) puts a phase U_l = exp(i theta_l) on
every link of a fixed lattice; the field's curvature is the holonomy round each small loop.

What this module fixes (ours, recorded in Q29):
* The graph is any of the toy's bipartite neighbour arrays (`graphity.cqg.torus`, `graphity.cqg_d.torus`, or a
  chain's state); it is held fixed here. Riding the field on a graph whose wiring switches is not in this piece.
* A link is oriented from its point of kind 0 to its point of kind 1, so the toy's two kinds of point give every
  link its direction; theta_l is the phase in that direction.
* The loops are the toy's squares: every 4-cycle, found from the wiring alone. On a flat torus with every side at
  least 6 these are exactly the lattice's plaquettes; on a side of 4 (a curled direction) the cycle round the curl is
  a square too, so the field sees it (`graphity.cqg_d` counts it the same way).
* The square u0 -> v0 -> u1 -> v1 -> u0 (u of kind 0, v of kind 1) has holonomy
      phi_p = theta(u0 v0) - theta(u1 v0) + theta(u1 v1) - theta(u0 v1).
  A gauge transformation, theta(u v) -> theta(u v) + alpha_u - alpha_v, leaves every phi_p unchanged.
* Energy of a field with link velocities thetadot (unit inertia per link):
      H = sum_l thetadot_l^2 / 2 + kappa sum_p (1 - cos phi_p).
  With kappa = 1 the linear waves on a flat torus travel at most one link per tick (tested).
* Charge at a point (Gauss's law): Q_x = sum of E over links leaving x minus over links arriving, with E the field
  conjugate to theta (here thetadot). On any closed graph the charges add to zero, real-valued or mod 3 (tested).

Monte Carlo (Euclidean weight exp(beta sum_p cos phi_p), the standard Wilson action):
  a single link's phase is moved by delta drawn uniformly from [-step, step] and wrapped into (-pi, pi]; the proposal
  density from theta to theta' equals that from theta' to theta, so Metropolis acceptance min(1, exp(beta dS)), with
  dS the change of sum_p cos phi_p over the squares through that link, satisfies detailed balance with respect to the
  weight. One sweep visits every link once in order; each single update leaves the weight invariant, so the sweep does
  too. Ergodic: every link can reach every phase.

Reproduction gate (TASKS, Gate U1): on a flat torus in two directions the average square is known exactly. Changing
variables to the squares' holonomies, which on a torus obey one constraint (summed with alternating signs e_p, because
neighbouring squares are walked in opposite senses, they make a multiple of 2 pi; cos is even, so the signs drop out
of the answer; corrected 24 October, the first version said a plain sum), gives
      Z = sum_n I_n(beta)^P,   <cos phi> = sum_n I_n^(P-1) (I_(n-1) + I_(n+1)) / 2  /  sum_n I_n^P,
P the number of squares and I_n the modified Bessel functions (ours, exact; the character expansion of 2D lattice
gauge theory, textbook, to verify at its source). `exact_mean_cos_2d` computes it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.special import iv


def links(adj, part):
    """Oriented links (tail of kind 0, head of kind 1) and, for every point and neighbour slot, the link's index."""
    n, deg = adj.shape
    tail, head = [], []
    slot_link = -np.ones((n, deg), dtype=np.int64)
    index = {}
    for u in range(n):
        if part[u] != 0:
            continue
        for k in range(deg):
            v = int(adj[u, k])
            index[(u, v)] = len(tail)
            tail.append(u)
            head.append(v)
    for x in range(n):
        for k in range(deg):
            y = int(adj[x, k])
            slot_link[x, k] = index[(x, y)] if part[x] == 0 else index[(y, x)]
    return np.array(tail, dtype=np.int64), np.array(head, dtype=np.int64), slot_link, index


def squares(adj, part):
    """Every 4-cycle u0-v0-u1-v1 once, as (u0, v0, u1, v1) with u0 < u1 of kind 0."""
    n, deg = adj.shape
    nbr = [set(int(y) for y in adj[x]) for x in range(n)]
    out = []
    for u0 in range(n):
        if part[u0] != 0:
            continue
        for a in range(deg):
            for b in range(a + 1, deg):
                v0, v1 = int(adj[u0, a]), int(adj[u0, b])
                for u1 in nbr[v0]:
                    if u1 > u0 and v1 in nbr[u1]:
                        out.append((u0, v0, u1, v1))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


class Field:
    """The fixed graph's links, squares and the matrices that join them."""

    def __init__(self, adj, part):
        self.adj = np.asarray(adj, dtype=np.int64)
        self.part = np.asarray(part, dtype=np.int64)
        self.n = len(self.adj)
        self.tail, self.head, self.slot_link, self.index = links(self.adj, self.part)
        self.n_links = len(self.tail)
        self.squares = squares(self.adj, self.part)
        self.n_squares = len(self.squares)
        sq_links = np.empty((self.n_squares, 4), dtype=np.int64)
        sq_sign = np.array([1, -1, 1, -1], dtype=np.int64)
        for p, (u0, v0, u1, v1) in enumerate(self.squares):
            sq_links[p] = (self.index[(u0, v0)], self.index[(u1, v0)], self.index[(u1, v1)], self.index[(u0, v1)])
        self.sq_links = sq_links
        self.sq_sign = np.tile(sq_sign, (self.n_squares, 1))
        rows = np.repeat(np.arange(self.n_squares), 4)
        self.B = sp.csr_matrix((self.sq_sign.ravel(), (rows, sq_links.ravel())),
                               shape=(self.n_squares, self.n_links))
        r = np.concatenate([self.tail, self.head])
        c = np.concatenate([np.arange(self.n_links)] * 2)
        v = np.concatenate([np.ones(self.n_links), -np.ones(self.n_links)])
        self.div = sp.csr_matrix((v, (r, c)), shape=(self.n, self.n_links))
        # squares through each link, padded with -1
        count = np.zeros(self.n_links, dtype=np.int64)
        for p in range(self.n_squares):
            for j in range(4):
                count[sq_links[p, j]] += 1
        m = max(1, int(count.max()) if self.n_squares else 1)
        self.link_sq = -np.ones((self.n_links, m), dtype=np.int64)
        self.link_sgn = np.zeros((self.n_links, m), dtype=np.int64)
        fill = np.zeros(self.n_links, dtype=np.int64)
        for p in range(self.n_squares):
            for j in range(4):
                l = sq_links[p, j]
                self.link_sq[l, fill[l]] = p
                self.link_sgn[l, fill[l]] = self.sq_sign[p, j]
                fill[l] += 1

    def holonomy(self, theta):
        return self.B @ theta

    def potential(self, theta, kappa=1.0):
        return kappa * float(np.sum(1.0 - np.cos(self.B @ theta)))

    def energy(self, theta, vel, kappa=1.0):
        return 0.5 * float(np.dot(vel, vel)) + self.potential(theta, kappa)

    def charge(self, E):
        return self.div @ E

    def gauge(self, theta, alpha):
        """theta(u v) -> theta(u v) + alpha_u - alpha_v."""
        return theta + alpha[self.tail] - alpha[self.head]

    def mode_matrix(self):
        """B^T B: the squared frequencies of the linear waves (kappa = 1, unit inertia)."""
        return (self.B.T @ self.B).toarray()

    def force(self, theta, kappa=1.0):
        return -kappa * (self.B.T @ np.sin(self.B @ theta))

    def evolve(self, theta, vel, kappa=1.0, dt=0.05, steps=100):
        """Velocity Verlet for the nonlinear equations; returns the final (theta, vel)."""
        theta, vel = theta.copy(), vel.copy()
        f = self.force(theta, kappa)
        for _ in range(steps):
            vel += 0.5 * dt * f
            theta += dt * vel
            f = self.force(theta, kappa)
            vel += 0.5 * dt * f
        return theta, vel

    def square_signs(self):
        """The signs e_p with sum_p e_p B_pl = 0 for every link, on a closed two-direction torus.

        Squares are walked u0 -> v0 -> u1 -> v1 with links oriented by kind, so neighbouring squares run in opposite
        senses; the signed sum of holonomies is identically zero, and with each holonomy wrapped into (-pi, pi] it is
        2 pi times the winding number Q. Raises unless B^T has exactly one null direction (a closed orientable
        surface).
        """
        u, s, vt = np.linalg.svd(self.B.toarray().T)
        null = int(np.sum(s < 1e-9)) + (self.n_squares - len(s))
        if null != 1:
            raise ValueError(f"square signs need exactly one null direction of B^T, found {null}")
        e = vt[-1] / np.abs(vt[-1]).max()
        if not np.allclose(np.abs(e), 1.0, atol=1e-9):
            raise ValueError("the null direction of B^T is not a sign vector")
        return np.sign(e).astype(np.int64)

    def winding(self, theta):
        """Q = sum_p e_p wrap(phi_p) / (2 pi), a whole number on a closed two-direction torus."""
        q = float(np.sum(self.square_signs() * (_wrap_array(self.B @ theta)))) / (2.0 * np.pi)
        return int(round(q))

    def instanton(self):
        """A link field whose wrapped holonomies all equal 2 pi / P in the sense e_p (winding Q = 1).

        Target holonomies h_p = e_p 2 pi / P on every square except square 0, which gets e_0 (2 pi / P - 2 pi), so
        that sum_p e_p h_p = 0 and h lies in the range of B; solved by least squares and checked.
        """
        e = self.square_signs()
        P = self.n_squares
        h = e * (2.0 * np.pi / P)
        h[0] -= e[0] * 2.0 * np.pi
        x = np.linalg.lstsq(self.B.toarray(), h, rcond=None)[0]
        if not np.allclose(self.B @ x, h, atol=1e-8):
            raise ValueError("instanton target not reached")
        return x

    def metropolis(self, theta, beta, step, n_equil, n_meas, seed, instanton=None):
        """Wilson-action Monte Carlo. Returns (final theta, mean cos phi after each measured sweep, acceptance).

        With `instanton` (a link field from `Field.instanton`), each sweep ends with one global move
        theta -> theta + s * instanton, s = +1 or -1 with equal chance, accepted by Metropolis on the whole action:
        the proposal is symmetric (s and -s equally likely, and each undoes the other), so detailed balance holds.
        It is a topology-changing proposal: it changes the winding by exactly one for smooth fields, but where square
        holonomies sit near +-pi the wrapped count can change by another amount (Codex audit, 24 October, O154). Without it the chain is draw for draw what it was before the
        option existed. The returned acceptance counts single-link moves only.
        """
        theta = np.asarray(theta, dtype=np.float64).copy()
        out = np.empty(n_meas, dtype=np.float64)
        use = instanton is not None
        inst = np.asarray(instanton, dtype=np.float64) if use else np.zeros(self.n_links)
        acc = _metropolis(theta, self.sq_links, self.sq_sign, self.link_sq, self.link_sgn,
                          beta, step, n_equil, n_meas, seed, out, use, inst)
        return theta, out, acc


@njit(cache=True)
def _hol(theta, sq_links, sq_sign, p):
    h = 0.0
    for j in range(4):
        h += sq_sign[p, j] * theta[sq_links[p, j]]
    return h


@njit(cache=True)
def delta_action(theta, sq_links, sq_sign, link_sq, link_sgn, l, d):
    """Change of sum_p cos phi_p when theta_l moves by d (only the squares through l)."""
    ds = 0.0
    for k in range(link_sq.shape[1]):
        p = link_sq[l, k]
        if p < 0:
            break
        h = _hol(theta, sq_links, sq_sign, p)
        ds += np.cos(h + link_sgn[l, k] * d) - np.cos(h)
    return ds


@njit(cache=True)
def _wrap(x):
    return x - 2.0 * np.pi * np.floor((x + np.pi) / (2.0 * np.pi))


def _wrap_array(x):
    return x - 2.0 * np.pi * np.floor((x + np.pi) / (2.0 * np.pi))


@njit(cache=True)
def _metropolis(theta, sq_links, sq_sign, link_sq, link_sgn, beta, step, n_equil, n_meas, seed, out,
                use_instanton, inst):
    np.random.seed(seed)
    n_links = theta.shape[0]
    n_sq = sq_links.shape[0]
    accepted = 0
    tried = 0
    for sweep in range(n_equil + n_meas):
        for l in range(n_links):
            d = step * (2.0 * np.random.random() - 1.0)
            ds = delta_action(theta, sq_links, sq_sign, link_sq, link_sgn, l, d)
            tried += 1
            if ds >= 0.0 or np.random.random() < np.exp(beta * ds):
                theta[l] = _wrap(theta[l] + d)
                accepted += 1
        if use_instanton:
            s = 1.0 if np.random.random() < 0.5 else -1.0
            ds = 0.0
            for p in range(n_sq):
                h = _hol(theta, sq_links, sq_sign, p)
                hi = 0.0
                for j in range(4):
                    hi += sq_sign[p, j] * inst[sq_links[p, j]]
                ds += np.cos(h + s * hi) - np.cos(h)
            if ds >= 0.0 or np.random.random() < np.exp(beta * ds):
                for l in range(n_links):
                    theta[l] = _wrap(theta[l] + s * inst[l])
        if sweep >= n_equil:
            s = 0.0
            for p in range(n_sq):
                s += np.cos(_hol(theta, sq_links, sq_sign, p))
            out[sweep - n_equil] = s / n_sq
    return accepted / tried


def exact_mean_cos_2d(beta, n_squares, n_max=60):
    """<cos phi> for compact U(1) on a closed two-direction torus with n_squares squares (character expansion)."""
    ns = np.arange(-n_max, n_max + 1)
    In = iv(ns, beta)
    lead = iv(0, beta)
    r = In / lead                      # scale out I_0^P to avoid overflow
    num = np.sum(r ** (n_squares - 1) * 0.5 * (iv(ns - 1, beta) + iv(ns + 1, beta)) / lead)
    den = np.sum(r ** n_squares)
    return float(num / den)
