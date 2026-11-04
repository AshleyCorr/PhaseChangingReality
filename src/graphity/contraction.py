"""The contraction model, layer form (VISION Updates 63 and 65; docs/design/contraction_model_brief.md).

Naming: at lambda != 1, and with a varying count of points, this is our family around the published model, never
combinatorial quantum gravity (VISION Update 22).

What this module adds to the six-link toy (cqg_d.py): two moves that change the number of points, acting on a whole
layer pair of a torus at once, so that no point is ever left broken (the layer form the author chose on 18 October
2026, Update 65). The energy is the toy's, H = 16 (3N - S) + 4 lam X, read on whatever count the graph has, plus an
optional cost per point mu_N * N (the brief, section 2.2; mu_N = 0 pending her decision).

The state is a torus with coordinates kept per point: `adj` (N, 6), `part` (N,), `coords` (N, 3), `dims` (3,), and
one store per point. States are kept CANONICAL: a point's index is the ravel order of its coordinates, so that a
torus is cqg_d.torus's up to the order of the slots. Names are a convenience (VISION Update 12); a move renames.
The torus wiring is the lattice's unless a switch has moved it; the layer moves check the links they need and
refuse the move otherwise, so they never guess.

* `contract_layers(state, axis, x0)`: the layers x0 and x0 + 1 along `axis` are removed and every surviving point
  (x0 - 1, p) is linked to (x0 + 2, p). Requires the lattice links (x0-1,p)-(x0,p), (x0,p)-(x0+1,p),
  (x0+1,p)-(x0+2,p) for every transverse position p, no other link between the removed layers and the rest, and a
  width of at least 6 along the axis before the move (the result is at least 4). The removed layers' stores go to the
  spliced pair at the same p, half to each end. Exact (sandbox, 18 October; tests): on the flat torus this takes
  6 x L x L to the slab 4 x L x L with every point still closed-or-open and H rising by exactly 4 (lam - 1) per
  surviving point, and every step above six around is free.
* `split_layers(state, axis, x0)`: the reverse. The lattice links (x0,p)-(x0+1,p) are cut and two new layers of
  points are inserted between them with lattice wiring; their stores start at zero. Requires every (x0,p)-(x0+1,p)
  link present.
Both return None when refused, and never leave the state half-changed. Both are checked against the hard-core rule.
After a contraction at x0 the spliced pair sits at (x0 - 1) and x0 of the new width; after a split at x0 the two
old layers sit at x0 and x0 + 3 of the new width (both mod the new width).

Two protocols:
* `run_feed`: the driven protocol of T67/T68/the brief (no temperature). Each sweep adds `rate` to every store (the
  "all" place), then proposes `proposals_per_sweep` layer moves at random (an axis, a position, split or contract).
  A move's energy change dF = dH + mu_N dN is paid from the pooled stores of the points whose links the move touches
  (the two layers on either side of the cut, and for a contraction the two removed layers too), each in proportion to
  what it holds; a move the pool cannot pay is refused; a release is credited evenly to the two layers that end up
  beside the cut. A `gate` (an energy) is the one new ingredient the layer form needs to make X stable for now: a
  move that costs nothing (a release, or a free move between flat widths) fires only when its pool already holds at
  least `gate`, which is NOT spent (an activation, a hill not a toll). Without it the slab splits at the first
  proposal, since the split is downhill; and once split, the lump sitting in the stores can pay the contraction
  back at the next proposal, so nothing in this protocol holds the product open unless the lump leaves (the leak):
  the layer form's version of T9's "room". Its value is a knob of the protocol, declared in the config; gate = 0 means the slab
  splits at the first proposal. Energy is conserved exactly in the sense H + mu_N N + sum(stores) - fed + leaked =
  constant; a `leak` removes that fraction of every store each sweep and is counted. This protocol is not an
  equilibrium chain and claims no detailed balance; conservation is its invariant, and the tests check it to 1e-9.
* `run_thermal_widths`: a Metropolis chain over the layer moves alone at coupling g, for reading mu_N off the
  width's fluctuations (the brief, E5). Because states are canonical, every split position from a given width leads
  to the same state and so does every contraction, so the proposal (split or contract with probability 1/2, a
  position chosen uniformly) is symmetric between neighbouring widths and the acceptance is min(1, exp(-dF/g)).
  With layer moves alone every flat width at or above six has H = 0, so the stationary distribution over widths is
  exactly Boltzmann in H + mu_N N (the slab's extra 4 (lam - 1) N at width 4), which the tests check against the
  exact sum (`exact_width_distribution`). A maximum width `max_width` is a hard wall.
"""
from dataclasses import dataclass

import numpy as np

from .cqg_d import hamiltonian, is_valid, torus

DEG = 6


@dataclass
class LayerState:
    adj: np.ndarray
    part: np.ndarray
    coords: np.ndarray
    dims: tuple
    stores: np.ndarray

    @property
    def n(self):
        return int(self.adj.shape[0])

    def copy(self):
        return LayerState(self.adj.copy(), self.part.copy(), self.coords.copy(), tuple(self.dims), self.stores.copy())

    def index(self, c):
        """The point at coordinates c (canonical order)."""
        return int(np.ravel_multi_index(tuple(int(x) % d for x, d in zip(c, self.dims)), self.dims))


def build(dims, stores=None):
    """A flat (or, at width 4, curled) three-direction torus with coordinates and empty stores."""
    dims = tuple(int(x) for x in dims)
    if len(dims) != 3:
        raise ValueError("three directions only")
    adj, part = torus(dims)
    coords = np.array(np.unravel_index(np.arange(adj.shape[0]), dims)).T.astype(np.int64)
    st = np.zeros(adj.shape[0]) if stores is None else np.asarray(stores, dtype=float).copy()
    return LayerState(np.sort(adj, axis=1), part, coords, dims, st)


def energy(state, lam, mu_n=0.0):
    """H + mu_N N on the current count."""
    return float(hamiltonian(state.adj, lam)) + mu_n * state.n


def _has(adj, a, b):
    return bool(np.any(adj[a] == b))


def _canonical(adj, part, coords, dims, stores):
    """Renumber points by the ravel order of their coordinates; sort slots."""
    key = np.ravel_multi_index(tuple(coords.T % np.array(dims)[:, None]), dims)
    order = np.argsort(key, kind="stable")
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    new_adj = np.sort(inv[adj[order]], axis=1)
    return LayerState(new_adj, part[order].copy(), coords[order] % np.array(dims), tuple(dims), stores[order].copy())


def _rotate(state, axis, by):
    """The same torus with coordinates shifted along `axis` (a renaming)."""
    coords = state.coords.copy()
    coords[:, axis] = (coords[:, axis] + by) % state.dims[axis]
    return _canonical(state.adj, state.part, coords, state.dims, state.stores)


def contract_layers(state, axis, x0):
    """Remove the layers x0 and x0 + 1 along `axis`; splice (x0-1, p) to (x0+2, p). None if refused."""
    L = state.dims[axis]
    if L < 6:
        return None
    x0 %= L
    if x0 == L - 1:                       # the pair wraps; rename so that it does not
        state = _rotate(state, axis, 1)
        x0 = L - 2
    xa, xb, xm, xp = x0, x0 + 1, (x0 - 1) % L, (x0 + 2) % L
    cx = state.coords[:, axis]
    is_removed = (cx == xa) | (cx == xb)
    for i in np.flatnonzero(cx == xa):
        c = state.coords[i]
        j = state.index(np.r_[c[:axis], xb, c[axis + 1:]])
        m = state.index(np.r_[c[:axis], xm, c[axis + 1:]])
        q = state.index(np.r_[c[:axis], xp, c[axis + 1:]])
        if not (_has(state.adj, i, j) and _has(state.adj, i, m) and _has(state.adj, j, q)):
            return None
    survivors = np.flatnonzero(~is_removed)
    cross = is_removed[state.adj[survivors]].sum(axis=1)
    border = (cx[survivors] == xm) | (cx[survivors] == xp)
    if np.any(cross[border] != 1) or np.any(cross[~border] != 0):
        return None
    new_index = np.full(state.n, -1, dtype=np.int64)
    new_index[survivors] = np.arange(len(survivors))
    adj = np.empty((len(survivors), DEG), dtype=np.int64)
    stores = state.stores[survivors].copy()
    for k, i in enumerate(survivors):
        row = state.adj[i]
        keep = row[~is_removed[row]]
        if cx[i] == xm or cx[i] == xp:
            c = state.coords[i]
            other = xp if cx[i] == xm else xm
            partner = state.index(np.r_[c[:axis], other, c[axis + 1:]])
            ra = state.index(np.r_[c[:axis], xa, c[axis + 1:]])
            rb = state.index(np.r_[c[:axis], xb, c[axis + 1:]])
            stores[k] += 0.5 * (state.stores[ra] + state.stores[rb])
            keep = np.append(keep, partner)
        if len(keep) != DEG or len(set(keep.tolist())) != DEG:
            return None
        adj[k] = new_index[keep]
    coords = state.coords[survivors].copy()
    coords[coords[:, axis] > xb, axis] -= 2
    new_dims = tuple(L - 2 if a == axis else state.dims[a] for a in range(3))
    out = _canonical(adj, state.part[survivors], coords, new_dims, stores)
    return out if is_valid(out.adj) else None


def split_layers(state, axis, x0, max_width=None):
    """Insert two lattice layers between x0 and x0 + 1 along `axis`. None if refused."""
    L = state.dims[axis]
    if max_width is not None and L + 2 > max_width:
        return None
    x0 %= L
    xb = (x0 + 1) % L
    cx = state.coords[:, axis]
    lower = np.flatnonzero(cx == x0)
    n_trans = len(lower)
    uppers = np.empty(n_trans, dtype=np.int64)
    for k, i in enumerate(lower):
        c = state.coords[i]
        j = state.index(np.r_[c[:axis], xb, c[axis + 1:]])
        if not _has(state.adj, i, j):
            return None
        uppers[k] = j
    new_dims = tuple(L + 2 if a == axis else state.dims[a] for a in range(3))
    coords = state.coords.copy()
    coords[coords[:, axis] > x0, axis] += 2
    n_old = state.n
    new_a = np.arange(n_old, n_old + n_trans)
    new_b = np.arange(n_old + n_trans, n_old + 2 * n_trans)
    ca = state.coords[lower].copy(); ca[:, axis] = x0 + 1
    cb = state.coords[lower].copy(); cb[:, axis] = x0 + 2
    coords = np.vstack([coords, ca, cb])
    # the new layers' sides follow their neighbours, not the coordinates (a renaming may have shifted parity)
    part = np.concatenate([state.part, 1 - state.part[lower], state.part[lower]]).astype(np.int64)
    stores = np.concatenate([state.stores, np.zeros(2 * n_trans)])
    adj = np.full((n_old + 2 * n_trans, DEG), -1, dtype=np.int64)
    adj[:n_old] = state.adj
    pos_a = {tuple(int(x) for x in ca[k]): int(new_a[k]) for k in range(n_trans)}
    pos_b = {tuple(int(x) for x in cb[k]): int(new_b[k]) for k in range(n_trans)}
    for k, i in enumerate(lower):
        j = uppers[k]
        adj[i, np.flatnonzero(adj[i] == j)[0]] = new_a[k]
        adj[j, np.flatnonzero(adj[j] == i)[0]] = new_b[k]
        a, b = int(new_a[k]), int(new_b[k])
        row_a = [int(i), b]
        row_b = [a, int(j)]
        for ax in range(3):
            if ax == axis:
                continue
            for step in (1, -1):
                qa = ca[k].copy(); qa[ax] = (qa[ax] + step) % new_dims[ax]
                qb = cb[k].copy(); qb[ax] = (qb[ax] + step) % new_dims[ax]
                row_a.append(pos_a[tuple(int(x) for x in qa)])
                row_b.append(pos_b[tuple(int(x) for x in qb)])
        adj[a] = row_a
        adj[b] = row_b
    out = _canonical(adj, part, coords, new_dims, stores)
    return out if is_valid(out.adj) else None


def pool_mask(state, axis, x0, contract):
    """Before a move: the points whose stores pay for it (the two layers either side of the cut, plus the removed
    pair for a contraction)."""
    L = state.dims[axis]
    x0 %= L
    xs = [x0, (x0 + 1) % L] if not contract else [x0, (x0 + 1) % L, (x0 - 1) % L, (x0 + 2) % L]
    return np.isin(state.coords[:, axis], xs)


def receiving_mask(new_state, axis, x0, contract):
    """After a move: the two layers beside the cut, which receive a release evenly."""
    Ln = new_state.dims[axis]
    if contract:
        L = Ln + 2
        x0 %= L
        if x0 == L - 1:        # the pair wrapped and the state was renamed by one before the move
            x0 = L - 2
        xs = [(x0 - 1) % Ln, x0 % Ln]
    else:
        xs = [x0 % Ln, (x0 + 3) % Ln]
    return np.isin(new_state.coords[:, axis], xs)


def run_feed(state, lam, rate, n_sweeps, seed, mu_n=0.0, gate=0.0, leak=0.0, proposals_per_sweep=1,
             max_width=None, axes=(0,), log_every=1):
    """The driven protocol of the layer form (module docstring). Returns the final state and rows
    (sweep, N, dims, F, stores_sum, fed, leaked, n_split, n_contract), F = H + mu_N N."""
    rng = np.random.default_rng(seed)
    st = state.copy()
    fed = leaked = 0.0
    n_split = n_contract = 0
    rows = []
    f = energy(st, lam, mu_n)
    for sweep in range(1, n_sweeps + 1):
        st.stores += rate
        fed += rate * st.n
        if leak > 0:
            out = leak * st.stores
            st.stores -= out
            leaked += float(out.sum())
        for _ in range(proposals_per_sweep):
            axis = int(axes[rng.integers(len(axes))])
            x0 = int(rng.integers(st.dims[axis]))
            contract = bool(rng.integers(2))
            cand = contract_layers(st, axis, x0) if contract else split_layers(st, axis, x0, max_width)
            if cand is None:
                continue
            f_new = energy(cand, lam, mu_n)
            df = f_new - f
            mask = pool_mask(st, axis, x0, contract)
            pool = float(st.stores[mask].sum())
            if df > 0:
                if pool < df:  # the whole bill, no tolerance, so no store goes negative (O158 (b), fixed 27 October 2026)
                    continue
                st.stores[mask] -= st.stores[mask] * (df / pool)
                cand = contract_layers(st, axis, x0) if contract else split_layers(st, axis, x0, max_width)
            else:
                if pool + 1e-12 < gate:
                    continue
                rec = receiving_mask(cand, axis, x0, contract)
                cand.stores[rec] += (-df) / int(rec.sum())
            st, f = cand, f_new
            if contract:
                n_contract += 1
            else:
                n_split += 1
        if sweep % log_every == 0 or sweep == n_sweeps:
            rows.append((sweep, st.n, tuple(st.dims), f, float(st.stores.sum()), fed, leaked, n_split, n_contract))
    return st, rows


def conservation_residual(f0, rows):
    """F + stores - fed + leaked - F0 at the last logged row; zero to rounding if energy is conserved."""
    _, _, _, f, s, fed, leaked, _, _ = rows[-1]
    return f + s - fed + leaked - f0


def run_thermal_widths(state, lam, g, n_sweeps, seed, mu_n=0.0, max_width=12, axis=0):
    """Metropolis over the layer moves alone along one axis (module docstring). Returns the width after each step."""
    rng = np.random.default_rng(seed)
    st = state.copy()
    f = energy(st, lam, mu_n)
    widths = np.empty(n_sweeps, dtype=np.int64)
    for sweep in range(n_sweeps):
        x0 = int(rng.integers(st.dims[axis]))
        contract = bool(rng.integers(2))
        cand = contract_layers(st, axis, x0) if contract else split_layers(st, axis, x0, max_width)
        if cand is not None:
            f_new = energy(cand, lam, mu_n)
            if rng.random() < np.exp(-(f_new - f) / g):
                st, f = cand, f_new
        widths[sweep] = st.dims[axis]
    return st, widths


def exact_width_distribution(lam, g, mu_n, dims_transverse, max_width, min_width=4):
    """The stationary distribution of run_thermal_widths, exact: Boltzmann in H + mu_N N over the torus family,
    where H = 0 for every width >= 6 and 4 (lam - 1) N at width 4. Valid for transverse widths >= 6 (flat); with a
    transverse width of 4 the states are the two-curled rod and the 6-cube, whose energies are not these."""
    area = int(np.prod(dims_transverse))
    ws = np.arange(min_width, max_width + 1, 2)
    fs = np.array([(4.0 * (lam - 1.0) * w * area if w == 4 else 0.0) + mu_n * w * area for w in ws])
    p = np.exp(-(fs - fs.min()) / g)
    return ws, p / p.sum()
